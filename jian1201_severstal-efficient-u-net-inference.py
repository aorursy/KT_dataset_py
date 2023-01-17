import cv2

import keras

import keras.backend as K

from keras.losses import binary_crossentropy

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import pandas as pd

import tensorflow as tf

from tqdm import tqdm
def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)



class FixedDropout(keras.layers.Dropout):

    def _get_noise_shape(self, inputs):

        if self.noise_shape is None:

            return self.noise_shape



        symbolic_shape = K.shape(inputs)

        noise_shape = [symbolic_shape[axis] if shape is None else shape

                       for axis, shape in enumerate(self.noise_shape)]

        return tuple(noise_shape)

def dice_coef_rounded(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred = K.cast(y_pred, 'float32')

    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')

    intersection = y_true_f * y_pred_f

    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

    return score

def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
densenet = load_model('../input/severstal-predict-missing-masks/model.h5')

densenet.summary()
custom_objects = custom_objects={

    'swish': tf.nn.swish,

    'FixedDropout': FixedDropout,

    'dice_coef': dice_coef,

    'bce_dice_loss': bce_dice_loss,

    'dice_coef_rounded': dice_coef_rounded

    

}

unet_model_path = '../input/severstal-u-net-with-efficientnetb4/model.h5'

unet = load_model(unet_model_path, custom_objects=custom_objects)



unet.summary()
sub_df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')

sub_df['ImageId'] = sub_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

test_imgs.head()
def create_test_gen(batch_size=64):

    return ImageDataGenerator(rescale=1/255.).flow_from_dataframe(

        test_imgs,

        directory='../input/severstal-steel-defect-detection/test_images',

        x_col='ImageId',

        class_mode=None,

        target_size=(256, 256),

        batch_size=batch_size,

        shuffle=False

    )
test_gen = create_test_gen()



test_missing_pred = densenet.predict_generator(

    test_gen,

    steps=len(test_gen),

    verbose=1

)



test_imgs['allMissing'] = test_missing_pred



filtered_test_imgs = test_imgs[test_imgs['allMissing'] < 0.5]

print(filtered_test_imgs.shape)

filtered_test_imgs.head()
filtered_mask = sub_df['ImageId'].isin(filtered_test_imgs["ImageId"].values)

filtered_sub_df = sub_df[filtered_mask].copy()

null_sub_df = sub_df[~filtered_mask].copy()

null_sub_df['EncodedPixels'] = null_sub_df['EncodedPixels'].apply(lambda x: ' ')



filtered_sub_df.reset_index(drop=True, inplace=True)

filtered_test_imgs.reset_index(drop=True, inplace=True)



print(filtered_sub_df.shape)

print(null_sub_df.shape)



filtered_sub_df.head()
def np_resize(img, input_shape):

    """

    Reshape a numpy array, which is input_shape=(height, width), 

    as opposed to input_shape=(width, height) for cv2

    """

    height, width = input_shape

    return cv2.resize(img, (width, height))

    

def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle2mask(rle, input_shape):

    width, height = input_shape[:2]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return mask.reshape(height, width).T



def build_masks(rles, input_shape, reshape=None):

    depth = len(rles)

    if reshape is None:

        masks = np.zeros((*input_shape, depth))

    else:

        masks = np.zeros((*reshape, depth))

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            if reshape is None:

                masks[:, :, i] = rle2mask(rle, input_shape)

            else:

                mask = rle2mask(rle, input_shape)

                reshaped_mask = np_resize(mask, reshape)

                masks[:, :, i] = reshaped_mask

    

    return masks



def build_rles(masks, reshape=None):

    width, height, depth = masks.shape

    

    rles = []

    

    for i in range(depth):

        mask = masks[:, :, i]

        

        if reshape:

            mask = mask.astype(np.float32)

            mask = np_resize(mask, reshape).astype(np.int64)

        

        rle = mask2rle(mask)

        rles.append(rle)

        

    return rles
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path='../input/severstal-steel-defect-detection/train_images',

                 batch_size=32, dim=(256, 1600), n_channels=3, reshape=None,

                 n_classes=4, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df

        self.list_IDs = list_IDs

        self.reshape = reshape

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.random_state = random_state

        

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        

        X = self.__generate_X(list_IDs_batch)

        

        if self.mode == 'fit':

            y = self.__generate_y(list_IDs_batch)

            return X, y

        

        elif self.mode == 'predict':

            return X



        else:

            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

        

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.seed(self.random_state)

            np.random.shuffle(self.indexes)

    

    def __generate_X(self, list_IDs_batch):

        'Generates data containing batch_size samples'

        # Initialization

        if self.reshape is None:

            X = np.empty((self.batch_size, *self.dim, self.n_channels))

        else:

            X = np.empty((self.batch_size, *self.reshape, self.n_channels))

        

        # Generate data

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            img_path = f"{self.base_path}/{im_name}"

            img = self.__load_rgb(img_path)

            

            if self.reshape is not None:

                img = np_resize(img, self.reshape)

            

            # Store samples

            X[i,] = img



        return X

    

    def __generate_y(self, list_IDs_batch):

        if self.reshape is None:

            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        else:

            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)

        

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            

            rles = image_df['EncodedPixels'].values

            

            if self.reshape is not None:

                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)

            else:

                masks = build_masks(rles, input_shape=self.dim)

            

            y[i, ] = masks



        return y

    

    def __load_grayscale(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.

        img = np.expand_dims(img, axis=-1)



        return img

    

    def __load_rgb(self, img_path):

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.



        return img
test_df = []



for i in range(0, filtered_test_imgs.shape[0], 300):

    batch_idx = list(

        range(i, min(filtered_test_imgs.shape[0], i + 300))

    )

    

    test_generator = DataGenerator(

        batch_idx,

        df=filtered_test_imgs,

        shuffle=False,

        mode='predict',

        base_path='../input/severstal-steel-defect-detection/test_images',

        target_df=filtered_sub_df,

        reshape=(256, 512),

        batch_size=1,

        n_classes=4

    )

    

    batch_pred_masks = unet.predict_generator(

        test_generator, 

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )

    

    for j, b in tqdm(enumerate(batch_idx)):

        filename = filtered_test_imgs['ImageId'].iloc[b]

        image_df = filtered_sub_df[filtered_sub_df['ImageId'] == filename].copy()

        

        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks, reshape=(256, 1600))

        

        image_df['EncodedPixels'] = pred_rles

        test_df.append(image_df)
test_df = pd.concat(test_df)

final_submission_df = pd.concat([test_df, null_sub_df])



print(test_df.shape)

print(final_submission_df.shape)



final_submission_df.head()
final_submission_df[['ImageId_ClassId', 'EncodedPixels']].to_csv('submission.csv', index=False)