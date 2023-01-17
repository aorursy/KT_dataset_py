!pip install git+https://github.com/qubvel/efficientnet
import gc

import os

import warnings

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm



import cv2

import PIL

from PIL import ImageOps, ImageFilter, ImageDraw



from keras import backend as K

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger



from keras.preprocessing.image import ImageDataGenerator



from efficientnet.keras import EfficientNetB3



from sklearn.model_selection import train_test_split



from imgaug import augmenters as iaa

import imgaug as ia



from keras.utils import Sequence, to_categorical



from sklearn.utils import shuffle





warnings.filterwarnings(action='ignore')



K.image_data_format()
# Parameter

application = EfficientNetB3

img_size=300

net_name='efficientnetb3'

learning_rate=0.001

min_learning_rate=0.00001

patience=4



epochs = 100

batch_size = 32
def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def crop_boxing_img(img_name, margin=16) :

    if img_name.split('_')[0] == "train" :

        PATH = TRAIN_IMG_PATH

        data = df_train

    elif img_name.split('_')[0] == "test" :

        PATH = TEST_IMG_PATH

        data = df_test

        

    img = PIL.Image.open(os.path.join(PATH, img_name))

    pos = data.loc[data["img_file"] == img_name, \

                   ['bbox_x1','bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)



    width, height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(pos[2] + margin, width)

    y2 = min(pos[3] + margin, height)

    

    if abs(pos[2] - pos[0]) > width or abs(pos[3] - pos[1]) > height:

        print(f'{img_name} is wrong bounding box, img size: {img.size},  bbox_x1: {pos[0]}, bbox_x2: {pos[2]}, bbox_y1: {pos[1]}, bbox_y2: {pos[3]}')

        return img



    return img.crop((x1,y1,x2,y2))
def get_model(model_name, image_size):

    base_model = model_name(weights='imagenet', input_shape=(image_size,image_size,3), include_top=False)

    

    model = Sequential()

    model.add(base_model)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(2048, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(196, activation='softmax'))

    model.summary()



    optimizer = Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1_m, precision_m, recall_m])



    return model
def get_steps(num_samples, batch_size):

    if (num_samples % batch_size) > 0 :

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size
# 혹 다른 데이터 셋 추가(Pretrained Model Weights)로 인해 PATH가 변경된다면 아래 PATH를 수정

DATA_PATH = '../input/2019-3rd-ml-month-with-kakr/'

os.listdir(DATA_PATH)
# 이미지 폴더 경로

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')



CROPPED_IMG_PATH = '../cropped'



# CSV 파일 경로

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))
# Data 누락 체크

if set(list(df_train.img_file)) == set(os.listdir(TRAIN_IMG_PATH)) :

    print("Train file 누락 없음!")

else : 

    print("Train file 누락")



if set(list(df_test.img_file)) == set(os.listdir(TEST_IMG_PATH)) :

    print("Test file 누락 없음!")

else : 

    print("Test file 누락")

    

# Data 갯수

print("Number of Train Data : {}".format(df_train.shape[0]))

print("Number of Test Data : {}".format(df_test.shape[0]))



print("타겟 클래스 총 갯수 : {}".format(df_class.shape[0]))

print("Train Data의 타겟 종류 갯수 : {}".format(df_train['class'].nunique()))
if (os.path.isdir(CROPPED_IMG_PATH) == False):

    os.mkdir(CROPPED_IMG_PATH)

    

for i, row in df_train.iterrows():

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(os.path.join(CROPPED_IMG_PATH, row['img_file']))



for i, row in df_test.iterrows():

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(os.path.join(CROPPED_IMG_PATH, row['img_file']))
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(

        [

            # apply the following augmenters to most images

            iaa.Fliplr(0.5), # horizontally flip 50% of all images

            #iaa.Flipud(0.2), # vertically flip 20% of all images

            # crop images by -5% to 10% of their height/width

            sometimes(iaa.CropAndPad(

                percent=(-0.05, 0.1),

                pad_mode=ia.ALL,

                pad_cval=(0, 255)

            )),

            sometimes(iaa.Affine(

                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis

                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)

                rotate=(-45, 45), # rotate by -45 to +45 degrees

                shear=(-16, 16), # shear by -16 to +16 degrees

                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

                cval=(0, 255), # if mode is constant, use a cval between 0 and 255

                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

            )),

            # execute 0 to 5 of the following (less important) augmenters per image

            # don't execute all of them, as that would often be way too strong

            iaa.SomeOf((0, 5),

                [

                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation

                    iaa.OneOf([

                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0

                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7

                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7

                    ]),

                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images

                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images

                    # search either for all edges or for directed edges,

                    # blend the result with the original image using a blobby mask

                    iaa.SimplexNoiseAlpha(iaa.OneOf([

                        iaa.EdgeDetect(alpha=(0.5, 1.0)),

                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),

                    ])),

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images

                    iaa.OneOf([

                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels

                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),

                    ]),

                    iaa.Invert(0.05, per_channel=True), # invert color channels

                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation

                    # either change the brightness of the whole image (sometimes

                    # per channel) or change the brightness of subareas

                    iaa.OneOf([

                        iaa.Multiply((0.5, 1.5), per_channel=0.5),

                        iaa.FrequencyNoiseAlpha(

                            exponent=(-4, 0),

                            first=iaa.Multiply((0.5, 1.5), per_channel=True),

                            second=iaa.ContrastNormalization((0.5, 2.0))

                        )

                    ]),

                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast

                    iaa.Grayscale(alpha=(0.0, 1.0)),

                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)

                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around

                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))

                ],

                random_order=True

            )

        ],

        random_order=True)
class My_Generator(Sequence):



    def __init__(self, image_filenames, labels,

                 batch_size, is_train=True,

                 mix=False, augment=False):

        self.image_filenames, self.labels = image_filenames, labels

        self.batch_size = batch_size

        self.is_train = is_train

        self.is_augment = augment

        if(self.is_train):

            self.on_epoch_end()

        self.is_mix = mix



    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        if(self.is_train):

            return self.train_generate(batch_x, batch_y)

        return self.valid_generate(batch_x, batch_y)



    def on_epoch_end(self):

        if(self.is_train):

            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

        else:

            pass

    

    def mix_up(self, x, y):

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y



    def train_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            path = os.path.join(CROPPED_IMG_PATH,sample)

            img = cv2.imread(path)

            img = cv2.resize(img, (img_size, img_size))

            if(self.is_augment):

                img = seq.augment_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        if(self.is_mix):

            batch_images, batch_y = self.mix_up(batch_images, batch_y)

        return batch_images, batch_y



    def valid_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            path = os.path.join(CROPPED_IMG_PATH,sample)

            img = cv2.imread(path)

            img = cv2.resize(img, (img_size, img_size))

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y
df_train["class"] = df_train["class"].astype('str')



df_train = df_train[['img_file', 'class']]

df_test = df_test[['img_file']]



its = np.arange(df_train.shape[0])

train_idx, val_idx = train_test_split(its, train_size = 0.8, random_state=42, stratify=df_train["class"])



X_train = df_train.iloc[train_idx, :]

X_val = df_train.iloc[val_idx, :]



nb_train_samples = len(X_train)

nb_validation_samples = len(X_val)

    

train_y = to_categorical(pd.to_numeric(X_train['class'], errors='coerce')-1, num_classes=196)

valid_y = to_categorical(pd.to_numeric(X_val['class'], errors='coerce')-1, num_classes=196) 
model_path = './'

        

train_generator = My_Generator(X_train['img_file'], train_y, batch_size, is_train=True, mix=False, augment=True)

valid_generator = My_Generator(X_val['img_file'], valid_y, batch_size, is_train=False)



model_name = model_path + net_name + '.hdf5'



model = get_model(application, img_size)



try:

    model.load_weights(model_name)

except:

    pass



es = EarlyStopping(monitor='val_f1_m', min_delta=0, patience=patience, verbose=1, mode='max')

rr = ReduceLROnPlateau(monitor = 'val_f1_m', factor = 0.5, patience = patience/2, min_lr=min_learning_rate, verbose=1, mode='max')

cl = CSVLogger(filename='../working/training_log.csv', separator=',', append=True)

mc = ModelCheckpoint(filepath=model_name, monitor='val_f1_m', verbose=1, save_best_only=True, mode='max')



callbackList = [cl, es, rr, mc]



model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['acc', f1_m, precision_m, recall_m])



history = model.fit_generator(

    train_generator,

    steps_per_epoch=get_steps(nb_train_samples, batch_size),

    epochs=epochs,

    validation_data=valid_generator,

    validation_steps=get_steps(nb_validation_samples, batch_size),

    callbacks = callbackList

)
submit = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))



model_name = model_path + net_name + '.hdf5'

model = get_model(application, img_size)

model.load_weights(model_name)



score_predict = []

for i, img_name in tqdm(enumerate(submit['img_file'])):

    path = os.path.join(CROPPED_IMG_PATH, img_name)

    image = cv2.imread(path)

    image = cv2.resize(image, (img_size, img_size))

    X = np.array((image[np.newaxis])/255)

    score_predict.append(model.predict(X).ravel())

    

label_predict = np.argmax(score_predict, axis=1)

submit['class'] = label_predict+1

submit.to_csv('submission.csv', index=False)

submit.head()