import os



import numpy as np

from numpy.random import (

    choice

)

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib.patches import (

    Rectangle

)



import tensorflow as tf

from tensorflow.keras import (

    Input, 

    Model

)

from tensorflow.keras.layers import (

    Conv2D,

    BatchNormalization,

    LeakyReLU,

    Add  

)

from tensorflow.keras.optimizers import (

    Adam

)

from tensorflow.keras.callbacks import (

    ReduceLROnPlateau,

    EarlyStopping,

    TensorBoard

)

from tensorflow.keras.losses import (

    BinaryCrossentropy,

    Reduction

)

from PIL import Image, ImageDraw, ImageEnhance



import albumentations as A



from tqdm import tqdm



from pathlib import Path



from importlib import reload as reload_lib



data_root_dir = Path('../input/global-wheat-detection')

data_root_dir.is_dir()
train_csv_file = data_root_dir / 'train.csv'

train_csv_file.is_file()
data_df = pd.read_csv(train_csv_file)

data_df.head()
def extract_bbox_from_str(df_line):

    bbox = df_line['bbox'].str.split(',', expand=True)

    bbox[0] = bbox[0].str.slice(start=1)

    bbox[3] = bbox[3].str.slice(stop=-1)

    

    return bbox.values.astype(float)
data_df = data_df.groupby('image_id').apply(extract_bbox_from_str)

data_df['b6ab77fd7'][0:5]
N = data_df.shape[0]  # total number of samples

test_n = 10  # number of samples for final test



train_image_ids = np.unique(data_df.index.values)[:N-test_n]

val_image_ids = np.unique(data_df.index.values)[N-test_n:]

print(f'Number of train images: {train_image_ids.shape[0]}\nNumber of validation images: {val_image_ids.shape[0]}')
def load_images(data_df, data_root_dir, image_ids, data_type, resize_shape=None):

    def _load_image(img_root_dir, img_id, resize_shape):

        return np.asarray(Image.open(str(img_root_dir / (img_id+'.jpg'))).resize(resize_shape))



    images_dict = {}

    bboxes_dict = {}



    for img_id in tqdm(image_ids):

        images_dict[img_id] = _load_image(img_root_dir=data_root_dir / data_type, img_id=img_id, resize_shape=(256, 256))

        bboxes_dict[img_id] = data_df[img_id].copy() / 4

        

    return images_dict, bboxes_dict
resize_shape = (256, 256)



train_images_dict, train_bboxes_dict = load_images(data_df=data_df, data_root_dir=data_root_dir, image_ids=train_image_ids, data_type='train', resize_shape=resize_shape)

val_images_dict, val_bboxes_dict = load_images(data_df=data_df, data_root_dir=data_root_dir, image_ids=val_image_ids, data_type='train', resize_shape=resize_shape)
def show_image_sample(images, bboxes, sample_size=5):

    def _image_bbox_viz(ax, img, bboxes):

        ax.imshow(img)

        

        for bbox in bboxes:

            x, y, w, h = bbox

            ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.5, color='red'))

            

        return np.asarray(ax)

    fig, axs = plt.subplots(1, sample_size, figsize=(20, 20))

    if sample_size > 1:

        for idx, img in enumerate(images):

            _image_bbox_viz(axs[idx], img, bboxes[idx])

    else:

        _image_bbox_viz(axs, images[0], bboxes)
N = len(train_images_dict.values())

sample_size = 6



rand_sample_idx = choice(N, sample_size)

sample_images = np.array(list(train_images_dict.values()))[rand_sample_idx]

sample_bboxes = np.array(list(train_bboxes_dict.values()))[rand_sample_idx]



show_image_sample(

    images=sample_images, 

    bboxes=sample_bboxes, 

    sample_size=sample_size

)
def clean_bboxes(images_dict, bboxes_dict, min_bbox_area, max_bbox_area, clean=False, excluded_bboxes=None):

    small_bbox_area_cnt, large_bbox_area_cnt = (0, 0)

    for img_id in tqdm(bboxes_dict):

        bboxes = bboxes_dict[img_id]

        delete_bbox_idx = []

        for bbox_idx, bbox in enumerate(bboxes):

            if excluded_bboxes is not None:

                if (img_id, bbox_idx) in excluded_bboxes:

                    continue

            _, _, w, h = bbox

            if w * h <= min_bbox_area or w * h >= max_bbox_area:

                if w * h >= max_bbox_area:

                    # print(f'w * h = {w * h}')

                    print(f'image id: {img_id}, bbox index: {bbox_idx}')

                    show_image_sample(

                        images=[images_dict[img_id]], 

                        bboxes=[bbox], 

                        sample_size=1

                    )

                    large_bbox_area_cnt += 1

                else:

                    # print(f'w * h = {w * h}')

                    small_bbox_area_cnt += 1

                delete_bbox_idx.append(bbox_idx)

        if clean:

            bboxes_dict[img_id] = np.delete(bboxes, delete_bbox_idx, axis=0)

                

    print(f'Small area bboxes: {small_bbox_area_cnt}\nLarge area bboxes: {large_bbox_area_cnt}')

    return bboxes_dict
clean_train_bboxes_dict = clean_bboxes(

    images_dict=train_images_dict,

    bboxes_dict=train_bboxes_dict.copy(), 

    min_bbox_area=10, 

    max_bbox_area=8000,

    clean=True,

    excluded_bboxes=[('51f2e0a05', 5), ('69fc3d3ff', 1)]

)



# check

# image id: 51f2e0a05, bbox index: 5

image_id = '51f2e0a05'

bbox_idx = 5

show_image_sample(

    images=[train_images_dict[image_id]], 

    bboxes=[train_bboxes_dict[image_id][bbox_idx]], 

    sample_size=1

) 
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_ids, image_pixels, labels=None, batch_size=1, shuffle=False, augment=False):

        self.image_ids = image_ids

        self.image_pixels = image_pixels

        self.labels = labels

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.augment = augment

        

        self.on_epoch_end()

        

        self.image_grid = self.form_image_grid()

            

    def on_epoch_end(self):

        self.indexes  = np.arange(len(self.image_ids))

        

        if self.shuffle:

            np.random.shuffle(self.indexes)

            

    def form_image_grid(self):

        image_grid = np.zeros((32, 32, 4))

        

        # initial cell coordinates

        cell = [0, 0, 256 / 32, 256 / 32]

            

        for i in range(32):

            for j in range(32):

                image_grid[i, j] = cell

                cell[0] = cell[0] + cell[2]

            cell[0] = 0

            cell[1] = cell[1] + cell[3]



        return image_grid

    

    def __len__(self):

        return int(np.floor(len(self.image_ids) / self.batch_size))

    

    def __getitem__(self, index):

        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        

        batch_image_ids = [self.image_ids[i] for i in indexes]



        return self.__get_batch(batch_image_ids)

    

    def __get_batch(self, batch_image_ids):

        X, y = [], []

        

        for idx, image_id in enumerate(batch_image_ids):

            batch_image_pixels = self.image_pixels[image_id]

            batch_image_bboxes = self.labels[image_id]

            

            if self.augment:

                batch_image_pixels, batch_image_bboxes = self.augment_image(batch_image_pixels, batch_image_bboxes)

            else:

                batch_image_pixels = self.contrast_image(batch_image_pixels)

                batch_image_bboxes = self.form_label_grid(batch_image_bboxes)

            X.append(batch_image_pixels)

            y.append(batch_image_bboxes)

        return np.array(X), np.array(y)

    

    def augment_image(self, image_pixels, image_bboxes):

        bbox_labels = np.ones(len(image_bboxes))

        

        aug_result = self.train_aug(image=image_pixels, bboxes=image_bboxes, labels=bbox_labels)

        

        image_bboxes = self.form_label_grid(aug_result['bboxes'])

        

        return np.array(aug_result['image']) / 255, image_bboxes

    

    def contrast_image(self, image_pixels):

        aug_result = self.val_aug(image=image_pixels)

        return np.array(aug_result['image']) / 255

    

    

    def form_label_grid(self, bboxes):

        label_grid = np.zeros((32, 32, 10))

        

        for i in range(32):

            for j in range(32):

                cell = self.image_grid[i, j]

                label_grid[i, j] = self.rect_intersect(cell, bboxes)

        return label_grid

    

    def rect_intersect(self, cell, bboxes):

        cell_x, cell_y, cell_width, cell_height = cell

        cell_x_max = cell_x + cell_width

        cell_y_max = cell_y + cell_height

        

        anchor_one = np.array([0, 0, 0, 0, 0])

        anchor_two = np.array([0, 0, 0, 0, 0])

        

        for bbox in bboxes:

            bbox_x, bbox_y, bbox_width, bbox_height = bbox

            bbox_center_x = bbox_x + (bbox_width / 2)

            bbox_center_y = bbox_y + (bbox_height / 2)

            

            if bbox_center_x >= cell_x and bbox_center_x < cell_x_max and bbox_center_y >= cell_y and bbox_center_y < cell_y_max:

                if anchor_one[0] == 0:  # if there is no object present in the anchor 1 cell

                    anchor_one = self.yolo_shape(

                        [bbox_x, bbox_y, bbox_width, bbox_height],

                        [cell_x, cell_y, cell_width, cell_height]

                    )

                elif anchor_two[0] == 0:  # if there is no object present in the anchor 2 cell

                    anchor_two = self.yolo_shape(

                        [bbox_x, bbox_y, bbox_width, bbox_height],

                        [cell_x, cell_y, cell_width, cell_height]

                    )

                else:

                    break

        return np.concatenate((anchor_one, anchor_two), axis=None)

    

    def yolo_shape(self, bbox, cell):

        bbox_x, bbox_y, bbox_width, bbox_height = bbox

        cell_x, cell_y, cell_width, cell_height = cell

        

        # move the top left x, y coordinates to the center

        bbox_x = bbox_x + (bbox_width / 2)

        bbox_y = bbox_y + (bbox_height / 2)

        

        # x, y relative to cell

        bbox_x = (bbox_x - cell_x) / cell_width

        bbox_y = (bbox_y - cell_y) / cell_height

        

        # change the bbox width and height relative to the cell width and height

        bbox_width = bbox_width / 256

        bbox_height = bbox_height / 256

        

        return [1, bbox_x, bbox_y, bbox_width, bbox_height]
DataGenerator.train_aug = A.Compose([

        A.RandomSizedCrop(

            min_max_height=(200, 200), 

            height=256, 

            width=256, 

            p=0.8

        ),

        A.OneOf([

            A.Flip(),

            A.RandomRotate90(),

        ], p=1),

        A.OneOf([

            A.HueSaturationValue(),

            A.RandomBrightnessContrast()

        ], p=1),

        A.OneOf([

            A.GaussNoise(),

            A.GlassBlur(),

            A.ISONoise(),

            A.MultiplicativeNoise(),

        ], p=0.5),

        A.Cutout(

            num_holes=8, 

            max_h_size=16, 

            max_w_size=16, 

            fill_value=0, 

            p=0.5

        ),

        A.CLAHE(p=1),

        A.ToGray(p=1),

    ], 

    bbox_params={'format': 'coco', 'label_fields': ['labels']})

    

DataGenerator.val_aug = A.Compose([

    A.CLAHE(p=1),

    A.ToGray(p=1),

])
train_generator = DataGenerator(

    train_image_ids,

    train_images_dict,

    train_bboxes_dict,

    batch_size=6,

    shuffle=True,

    augment=True

)

image_grid = train_generator.image_grid



val_generator = DataGenerator(

    val_image_ids,

    val_images_dict,

    val_bboxes_dict,

    batch_size=10,

    shuffle=False, 

    augment=False

)
class YOLOv3:

    def __init__(self):

        self.darknet_53 = None

        self.build_net()



    def build_net(self):

        # == INPUT ==   

        print(f'Working on:\n\t>Input layers')

        X_input = Input(shape=(256, 256, 3))



        X = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(X_input)

        X = BatchNormalization()(X)

        X = LeakyReLU(alpha=0.1)(X)



        # == BLOCK 1 ==

        print(f'Working on:\n\t>Block 1')



        X = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(X)

        X = BatchNormalization()(X)

        X = LeakyReLU(alpha=0.1)(X)



        X_sc = X



        for layer_idx in tqdm(range(2)):

            X = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(X)

            X = BatchNormalization()(X)

            X = LeakyReLU(alpha=0.1)(X)

            

            X = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(X)

            X = BatchNormalization()(X)

            X = LeakyReLU(alpha=0.1)(X)

            

            X = Add()([X_sc, X])

            X = LeakyReLU(alpha=0.1)(X)

            

            X_sc = X

            

        # == BLOCK 2 ==

        print(f'Working on:\n\t>Block 2')



        X = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(X)

        X = BatchNormalization()(X)

        X = LeakyReLU(alpha=0.1)(X)



        X_sc = X



        for layer_idx in tqdm(range(2)):

            X = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(X)

            X = BatchNormalization()(X)

            X = LeakyReLU(alpha=0.1)(X)

            

            X = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(X)

            X = BatchNormalization()(X)

            X = LeakyReLU(alpha=0.1)(X)

            

            X = tf.keras.layers.Add()([X_sc, X])

            X = LeakyReLU(alpha=0.1)(X)

            

            X_sc = X

            

        # == BLOCK 3 ==

        print(f'Working on:\n\t>Block 3')

        X = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(X)

        X = BatchNormalization()(X)

        X = LeakyReLU(alpha=0.1)(X)



        X_sc = X



        for layer_idx in tqdm(range(8)):

            X = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(X)

            X = BatchNormalization()(X)

            X = LeakyReLU(alpha=0.1)(X)

            

            X = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(X)

            X = BatchNormalization()(X)

            X = LeakyReLU(alpha=0.1)(X)

            

            X = Add()([X_sc, X])

            X = LeakyReLU(alpha=0.1)(X)

            

            X_sc = X

            

        # == BLOCK 4 ==

        print(f'Working on:\n\t>Block 4')

        X = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(X)

        X = BatchNormalization()(X)

        X = LeakyReLU(alpha=0.1)(X)



        X_sc = X



        for layer_idx in tqdm(range(8)):

            X = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(X)

            X = BatchNormalization()(X)

            X = LeakyReLU(alpha=0.1)(X)

            

            X = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(X)

            X = BatchNormalization()(X)

            X = LeakyReLU(alpha=0.1)(X)

            

            X = Add()([X_sc, X])

            X = LeakyReLU(alpha=0.1)(X)

            

            X_sc = X

            

        # == BLOCK 5 ==

        print(f'Working on:\n\t>Block 5')



        X = Conv2D(1024, (3, 3), strides=(2, 2), padding='same')(X)

        X = BatchNormalization()(X)

        X = LeakyReLU(alpha=0.1)(X)



        X_sc = X



        for layer_idx in tqdm(range(4)):

            X = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(X)

            X = BatchNormalization()(X)

            X = LeakyReLU(alpha=0.1)(X)

            

            X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(X)

            X = BatchNormalization()(X)

            X = LeakyReLU(alpha=0.1)(X)

            

            X = Add()([X_sc, X])

            X = LeakyReLU(alpha=0.1)(X)

            

            X_sc = X



        # == OUTPUT ==

        print(f'Working on:\n\t>Output layers')



        X = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(X)

        X = BatchNormalization()(X)

        X = LeakyReLU(alpha=0.1)(X)



        X = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(X)

        X = BatchNormalization()(X)

        X = LeakyReLU(alpha=0.1)(X)



        X = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(X)

        X = BatchNormalization()(X)

        X = LeakyReLU(alpha=0.1)(X)



        preds = Conv2D(10, (1, 1), strides=(1, 1), activation='sigmoid')(X)



        self.darknet_53 = Model(inputs=X_input, outputs=preds)



        print(f'\n===\nModel was build successfully!\n===\n')



    def compile_model(self, optimizer, loss):

        self.darknet_53.compile(

            optimizer=optimizer,

            loss=loss

        )

def loss_func(y_true, y_pred):

    def _mask_by_y_true(y_true):

        anchor_one_mask = tf.where(

            y_true[..., 0]==0,

            0.5,

            5.0

        )



        anchor_two_mask = tf.where(

            y_true[..., 5]==0,

            0.5,

            5.0

        )

    

        return tf.concat([anchor_one_mask, anchor_two_mask], axis=0)



    binary_crossentropy = prob_loss = BinaryCrossentropy(

        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE

    )

    

    prob_loss = binary_crossentropy(

        tf.concat([y_true[..., 0], y_true[..., 5]], axis=0),

        tf.concat([y_pred[..., 0], y_pred[..., 5]], axis=0)

    )



    xy_loss = tf.keras.losses.MSE(

        tf.concat([y_true[..., 1:3], y_true[..., 6:8]], axis=0),

        tf.concat([y_pred[..., 1:3], y_pred[..., 6:8]], axis=0)

    )

    

    wh_loss = tf.keras.losses.MSE(

        tf.concat([y_true[..., 3:5], y_true[..., 8:10]], axis=0),

        tf.concat([y_pred[..., 3:5], y_pred[..., 8:10]], axis=0)

    )

    

    bboxes_mask = _mask_by_y_true(y_true)

    

    xy_loss = xy_loss * bboxes_mask

    wh_loss = wh_loss * bboxes_mask

    

    return prob_loss + xy_loss + wh_loss
yolo_v3 = YOLOv3()

yolo_v3.darknet_53.summary()
yolo_v3.compile_model(

    optimizer=Adam(learning_rate=1e-4),

    loss=loss_func

)
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),

    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True),

    tf.keras.callbacks.TensorBoard(log_dir='./logs', write_graph=True, write_images=True, update_freq='epoch')

]
epochs = 80

history = yolo_v3.darknet_53.fit(

    train_generator,

    validation_data=val_generator,

    epochs=epochs,

    callbacks=callbacks

)
def format_yolo_2_coco(yolo_bboxes, image_grid):

    bboxes = yolo_bboxes.copy()

    

    im_width = (image_grid[:,:,2] * 32)

    im_height = (image_grid[:,:,3] * 32)

    

    # descale x,y

    bboxes[:,:,1] = np.floor(bboxes[:,:,1] * image_grid[:,:,2]) + image_grid[:,:,0]

    bboxes[:,:,2] = np.floor(bboxes[:,:,2] * image_grid[:,:,3]) + image_grid[:,:,1]

    bboxes[:,:,6] = np.floor(bboxes[:,:,6] * image_grid[:,:,2]) + image_grid[:,:,0]

    bboxes[:,:,7] = np.floor(bboxes[:,:,7] * image_grid[:,:,3]) + image_grid[:,:,1]

    

    # descale width,height

    bboxes[:,:,3] = bboxes[:,:,3] * im_width 

    bboxes[:,:,4] = bboxes[:,:,4] * im_height

    bboxes[:,:,8] = bboxes[:,:,8] * im_width 

    bboxes[:,:,9] = bboxes[:,:,9] * im_height

    

    # centre x,y to top left x,y

    bboxes[:,:,1] = bboxes[:,:,1] - np.floor(bboxes[:,:,3] / 2)

    bboxes[:,:,2] = bboxes[:,:,2] - np.floor(bboxes[:,:,4] / 2)

    bboxes[:,:,6] = bboxes[:,:,6] - np.floor(bboxes[:,:,8] / 2)

    bboxes[:,:,7] = bboxes[:,:,7] - np.floor(bboxes[:,:,9] / 2)

    

    # width,heigth to x_max,y_max

    bboxes[:,:,3] = bboxes[:,:,1] + bboxes[:,:,3]

    bboxes[:,:,4] = bboxes[:,:,2] + bboxes[:,:,4]

    bboxes[:,:,8] = bboxes[:,:,6] + bboxes[:,:,8]

    bboxes[:,:,9] = bboxes[:,:,7] + bboxes[:,:,9]

    

    return bboxes
def clear_low_conf_bboxes(preds, top_n):

    def _switch_x_y(bboxes):

        x1 = bboxes[:, 0].copy()

        y1 = bboxes[:, 1].copy()

        x2 = bboxes[:, 2].copy()

        y2 = bboxes[:, 3].copy()

        

        bboxes[:, 0] = y1

        bboxes[:, 1] = x1

        bboxes[:, 2] = y2

        bboxes[:, 3] = x2

        

        return bboxes

    

    def _top_n_preds(probs, bboxes, top_n):

        bboxes = _switch_x_y(bboxes)

        top_n_indices = tf.image.non_max_suppression(

            boxes=bboxes,

            scores=probs,

            max_output_size=top_n,

            iou_threshold=0.3,

            score_threshold=0.3

        ).numpy()

        bboxes = _switch_x_y(bboxes)

        

        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]

        

        top_n_preds = list(zip(probs[top_n_indices], bboxes[top_n_indices]))

    

        res =  np.array(list(map(lambda pred: np.concatenate([[pred[0]], pred[1]]), top_n_preds)))



        return res

    

    

    probs = np.concatenate((preds[:, :, 0].flatten(), preds[:, :, 5].flatten()), axis=None)

    

    first_anchors = preds[:, :, 1:5].reshape((32*32, 4))

    second_anchors = preds[:, :, 6:10].reshape((32*32, 4))

    

    bboxes = np.concatenate((first_anchors, second_anchors), axis=0)

    

    preds = _top_n_preds(probs, bboxes, top_n=top_n)



    return preds
def get_final_preds(yolo_bboxes, image_ids, image_grid):

    preds = {}

    coco_bboxes = yolo_bboxes.copy()

    

    for idx, img_id in enumerate(image_ids):

        

        coco_bboxes[idx] = format_yolo_2_coco(

            yolo_bboxes=yolo_bboxes[idx], 

            image_grid=image_grid

        )



        preds[img_id] = clear_low_conf_bboxes(

            preds=coco_bboxes[idx], 

            top_n=100

        )

    return preds
test_dir = data_root_dir / 'test'

test_dir.is_dir()
test_image_ids = os.listdir(test_dir)

test_image_ids = [img_id[:-4] for img_id in test_image_ids]

print(f'Test image ids:')

for img_id in test_image_ids:

    print(f'\t- {img_id}')
test_preds = []



for idx, img_id in enumerate(test_image_ids):

    img = Image.open(str(test_dir) + f'/{img_id}' + '.jpg')

    img = img.resize((256, 256))

    

    img = np.asarray(img)

    

    aug = A.Compose([A.CLAHE(p=1), A.ToGray(p=1)])

    

    aug_img = aug(image=img)

    

    img = np.array(aug_img['image']) / 255

    

    img = np.expand_dims(img, axis=0)

    

    pred_yolo_bboxes = yolo_v3.darknet_53.predict(img)

    

    test_preds.append(pred_yolo_bboxes)

    

test_yolo_preds = np.concatenate(test_preds)

test_coco_preds = get_final_preds(

    yolo_bboxes=test_yolo_preds, 

    image_ids=test_image_ids, 

    image_grid=image_grid

)
test_preds = {}

model_scale = 256

original_scale = 1024

for key in test_coco_preds:

    pred_line = ''

    for bbox_idx, bbox_pred in enumerate(test_coco_preds[key]):

        for idx, pred in enumerate(bbox_pred):

            if not idx:

                pred_line += str(pred)

            else:

                pred_line += str(int(pred * original_scale / model_scale))

                

            if idx < len(bbox_pred):

                pred_line += ' '



#     print(f'line = {pred_line}')

    test_preds[key] = pred_line

    
final_preds_df = pd.DataFrame(dict(image_id=list(test_preds.keys()), PredictionString=list(test_preds.values())))

final_preds_df
final_preds_df.to_csv('submission.csv', index=False)
yolo_v3.darknet_53.save_weights(f'yolov3_{epochs}_epochs_weights')
def extract_bboxes(bboxes_csv_file, model_scale, original_scale):

    def _parse_str_line(str_line):

        bboxes = []

        data_line = np.array(str_line.split(' '))

        print(data_line)

        for idx, data in enumerate(data_line):

            bbox_num_data = []

            start_idx = idx * 5

            if start_idx >= len(data_line) - 5:

                break

            # print(f'{start_idx}, {len(data_line) + 5}')

            data_idx = np.arange(start_idx, start_idx + 5)

            # print(data_idx)

            bbox_str_data = data_line[data_idx]

            bbox_num_data.append(bbox_str_data[0])

            for bbox_str in bbox_str_data[1:]:

                bbox_num_data.append(int(bbox_str) * original_scale / model_scale)

            # print(bbox_data)

            bboxes.append(bbox_num_data)

        return np.array(bboxes, dtype=np.float32)

    bbox_df = pd.read_csv(bboxes_csv_file)

    bbox_df.PredictionString = bbox_df.PredictionString.apply(_parse_str_line)

    bbox_df.rename(columns={'PredictionString': 'PredictionArray'}, inplace=True)

    return bbox_df
def load_images(data_dir, image_ids):

    images = {}

    for image_id in image_ids:

        images[image_id] = np.asarray(Image.open(str(data_dir / image_id) + '.jpg'))

    return images

def show_test_image_sample(images, bboxes):

    def _image_bbox_viz(ax, image, image_bboxes):

        ax.imshow(image)

        

        for image_bbox in image_bboxes:

            c, x, y, w, h = image_bbox

            print(f'c = {c}, x = {x}, y = {y}, w = {w}, h = {h}')

            ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.5, color='red'))

            

        return np.asarray(ax)

    

    fig, axs = plt.subplots(1, len(images), figsize=(200, 200))



    for idx, image_bbox in enumerate(zip(images, bboxes)):

        _image_bbox_viz(axs[idx], image_bbox[0], image_bbox[1])

bbox_preds_data_df = extract_bboxes(

    bboxes_csv_file='submission.csv', 

    model_scale=1, 

    original_scale=1

)

bbox_preds_data_df


images = load_images(data_dir=Path(data_root_dir / 'test'), image_ids=bbox_preds_data_df.image_id.values)

show_test_image_sample(

    images=images.values(), 

    bboxes=bbox_preds_data_df.PredictionArray.values

)