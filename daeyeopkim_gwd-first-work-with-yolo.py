import numpy as np 

import pandas as pd 

import tensorflow as tf

import os

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageEnhance

import albumentations as albu

from tqdm.notebook import tqdm



DIR_INPUT = '../input/global-wheat-detection'

labels = pd.read_csv(f'{DIR_INPUT}/train.csv')

input_size = (256,256)

labels.head()
def group_boxes(group):

    

    """groups bbox by image_id and removes all without numbers

    

    Args:

        group: Series of pandas

    Returns:

        Arrays grouped by the same image_id

    """

    

    boundaries = group['bbox'].str.split(',', expand=True)

    

    # To get rid of '[' , ']'.

    boundaries[0] = boundaries[0].str.slice(start=1)

    boundaries[3] = boundaries[3].str.slice(stop=-1)

    

    return boundaries.to_numpy().astype(float)



labels = labels.groupby('image_id').apply(group_boxes)

len(labels)
train_image_ids = labels.index.to_numpy()[:-33]

valid_image_ids = labels.index.to_numpy()[-33:]
def load_image(image_id):

    """loads and resizes image to input size

    Args:

        image_id: An image id in train data

    return:

        resized image as array

    """

    global input_size

    image = Image.open(f'{DIR_INPUT}/train/{image_id}.jpg').resize(input_size)

    return np.asarray(image)



def reorganize(image_ids):

    """separates image data to pixels and bboxes

    Args:

        image_ids: An iterator that contains ids of image

    return:

        resized image as array, bboxes

    """

    

    images = {}

    bboxes = {}

    for image_id in tqdm(image_ids):

        #images[image_id] = np.expand_dims(load_image(image_id), axis = 0)

        images[image_id] = load_image(image_id)

        bboxes[image_id] = labels[image_id]

        

    return images, bboxes



train_images, train_bboxes = reorganize(train_image_ids)

valid_images, valid_bboxes = reorganize(valid_image_ids)
def draw_boxes_on_image(image, bboxes, color = 'red'):

    """draws lines on the picture where there are wheat

    Args:

        image: An image. Not array

        bboxes: an iterator of box data. (x,y,w,h)

        color : color of line

    return:

        image that lines are drawn on

    """

    draw = ImageDraw.Draw(image)

    for bbox in bboxes:

        draw.rectangle([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], 

                       width=4, outline=color)

    return image
example_id = valid_image_ids[2]

example_image = Image.open(f'{DIR_INPUT}/train/{example_id}.jpg')

example_detection = draw_boxes_on_image(example_image, valid_bboxes[example_id])

plt.figure(figsize=(8,8))

plt.imshow(example_detection)
class DataGenerator(tf.keras.utils.Sequence):

    """DataGenerator is input data going into model.fit and validation_data

    Every Sequence must implement the __getitem__ and the __len__ methods. 

    The method __getitem__ should return a complete batch.

    If you want to modify your dataset between epochs you may implement on_epoch_end. 

    """

    def __init__(self, image_ids, image_pixels, labels, 

                 batch_size=1, shuffle = False, augment = False):

        self.image_ids = image_ids

        self.image_pixels = image_pixels

        self.labels = labels

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.augment = augment

        self.image_grid = self.form_image_grid()

        

        self.on_epoch_end() 

        

    def __len__(self):

        """ is used to determine how many images there are in dataset.

        Python len() function returns the length of the object.

        This function internally calls __len__() function of the object. 

        So we can use len() function with any object that defines __len__() function. 

        """

        return int(np.floor(len(self.image_ids)/self.batch_size))

          

    def __getitem__(self, index):

        """When the batch corresponding to a given index is called, 

        the generator executes the __getitem__ method to generate it.

        i.e To get batch at position 'index'

        """

        

        # Generate indices of the batch

        indices = self.indices[index * self.batch_size : (index+1) * self.batch_size]

        

        # Find list of ids

        batch_ids = [self.image_ids[k] for k in indices]

        self.batch_ids = batch_ids

        

        # Generate data

        X, y = self.__data_generation(batch_ids)

        

        return X, y

        

    def on_epoch_end(self):

        """If you want to modify your dataset between epochs you may implement on_epoch_end"""

        

        self.indices = np.arange(len(self.image_ids))

        

        if self.shuffle:

            np.random.shuffle(self.indices)

        

    def __data_generation(self, batch_ids):

        """Produces batch-size of data """

        

        X, y = [], []

        

        # Generate data

        for image_id in batch_ids:

            pixels = self.image_pixels[image_id]

            bboxes = self.labels[image_id]

            

            if self.augment:

                pixels, bboxes = self.augment_image(pixels, bboxes)

                

            else:

                pixels = self.contrast_image(pixels)

                bboxes = self.form_label_grid(bboxes)

                

            X.append(pixels)

            y.append(bboxes)

        

        X = np.array(X)

        y = np.array(y)

        

        return X, y

    

    def form_image_grid(self):    

        """creates image grid cells which indicate the information about the location where a cell is """

        

        image_grid = np.zeros((16, 16, 4))

        cell = [0, 0, 16,  16] 



        for i in range(0, 16):

            for j in range(0, 16):

                image_grid[i,j] = cell



                cell[0] = cell[0] + cell[2]



            cell[0] = 0

            cell[1] = cell[1] + cell[3]



        return image_grid

    

    

    def augment_image(self, pixels, bboxes):

        """augments image

        

        Args:

            pixels: a batch size of images as array

            bboxes: a batch size of bboxes as array

        retruns:

            augmented images and bboxes scaled down 0 to 1,

        """

        

        # from 1024 to 256

        downsized_bboxes = bboxes / 4

        

        bbox_labels = np.ones(len(bboxes))

        aug_result = self.train_augmentations(image=pixels, bboxes=downsized_bboxes, labels=bbox_labels)

        bboxes = self.form_label_grid(aug_result['bboxes'])

        

        return aug_result['image']/256, bboxes

    

    def contrast_image(self, pixels):

        """converts images into grayscale"""

        

        aug_result = self.val_augmentations(image=pixels)

        return aug_result['image']/256

        

    def form_label_grid(self, bboxes):

        """returns Yolo shape of a label grid"""

        

        label_grid = np.zeros((16, 16, 5))

        

        for i in range(16):

            for j in range(16):

                cell = self.image_grid[i,j]

                label_grid[i,j] = self.rect_intersect(cell, bboxes) 

        

        return label_grid

    

    def rect_intersect(self, cell, bboxes):

        """puts all boundary boxes into appropriate cells in the grid."""

        

        cell_x, cell_y, cell_width, cell_height = cell

        cell_x_max = cell_x + cell_width

        cell_y_max = cell_y + cell_height



        anchor_one = np.zeros(5)

        anchor_two = np.zeros(5)

        

        for bbox in bboxes:

            if self.augment:

                bbox_ = bbox

            else :

                bbox_ = bbox/4

            box_x, box_y, box_width, box_height = bbox_

            box_x_centre = box_x + box_width/2

            box_y_centre = box_y + box_height/2

            

            # If the centre of box is in the cell, 

            if (box_x_centre >= cell_x and box_x_centre < cell_x_max

               and box_y_centre >= cell_y and box_y_centre < cell_y_max):

                

                if anchor_one[0] == 0:

                    anchor_one = self.yolo_shape(bbox_, cell)



                else:

                    break

            

        return anchor_one

    

    def yolo_shape(self, bbox, cell):

        """converts the shape of boundary boxes into the shape of Yolo """

    

            

        box_x, box_y, box_width, box_height = bbox 

        cell_x, cell_y, cell_width, cell_height = cell

        

        box_x_centre = box_x + box_width / 2

        box_y_centre = box_y + box_height / 2

        

        resized_box_x = (box_x_centre - cell_x) / cell_width

        resized_box_y = (box_y_centre - cell_y) / cell_height

        resized_box_width = box_width / 256 

        resized_box_height = box_height / 256

        

        return [1, resized_box_x, resized_box_y, resized_box_width, resized_box_height]



    
DataGenerator.train_augmentations = albu.Compose([

    

    albu.RandomSizedCrop(

        min_max_height = (200,200),

        height = 256,

        width = 256,

        p=0.8

    ),

    albu.OneOf([

        albu.Flip(),

        albu.RandomRotate90()

    ], p=1),

    albu.OneOf([

        albu.HueSaturationValue(),

        albu.RandomBrightnessContrast()

    ], p=1),

    albu.OneOf([

        albu.GaussNoise(),

        albu.GaussianBlur(),

        albu.ISONoise(),

        albu.MultiplicativeNoise()

    ], p=1),

    albu.Cutout(

        num_holes = 8,

        max_h_size = 16,

        max_w_size = 16,

        fill_value = 0,

        p = 0.5

    ),

    albu.CLAHE(p=1),

    albu.ToGray(p=1)

    

], bbox_params = {'format':'coco', 'label_fields': ['labels']})



DataGenerator.val_augmentations = albu.Compose([

    albu.CLAHE(p=1),

    albu.ToGray(p=1)

])
'To generate data_generator'





train_generator = DataGenerator(

    train_image_ids,

    train_images,

    train_bboxes,

    batch_size = 6,

    shuffle = True,

    augment = True

)



val_generator = DataGenerator(

    valid_image_ids,

    valid_images,

    valid_bboxes,

    batch_size = 1

)
x_input = tf.keras.Input(shape=(256,256,3))



x = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x_input)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



########## block 1 ##########

x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



x_shortcut = x



for i in range(2):

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x = tf.keras.layers.Add()([x_shortcut, x])

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x_shortcut = x





########## block 2 ##########

x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



x_shortcut = x



for i in range(2):

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x = tf.keras.layers.Add()([x_shortcut, x])

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x_shortcut = x



########## block 3 ##########

x = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



x_shortcut = x



for i in range(8):

    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x = tf.keras.layers.Add()([x_shortcut, x])

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x_shortcut = x



    

########## block 4 ##########

x = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



x_shortcut = x



for i in range(8):

    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x = tf.keras.layers.Add()([x_shortcut, x])

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x_shortcut = x



########## block 5 ##########

x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same')(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



x_shortcut = x



for i in range(4):

    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x = tf.keras.layers.Add()([x_shortcut, x])

    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



    x_shortcut = x



########## output layers ##########

x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)



predictions = tf.keras.layers.Conv2D(5, (1, 1), strides=(2, 2), activation='sigmoid')(x)



model = tf.keras.Model(inputs=x_input, outputs=predictions)
'loss function'



def loss_function(y_true, y_pred):

    """

    modified loss function that is refered to Yolo v1.

    Some coefficents are adjusted for having a proper loss value by empirical try

    """



    

    OBJ_SCALE = 5

    NO_OBJ_SCALE = 0.8

    WH_SCALE = 1.5

    

    true_conf = y_true[...,0:1]

    true_xy   = y_true[...,1:3]

    true_wh   = y_true[...,3:] * WH_SCALE

    

    pred_conf = y_pred[...,0:1]

    pred_xy   = y_pred[...,1:3]

    pred_wh   = y_pred[...,3:] * WH_SCALE



    obj_mask = tf.expand_dims(y_true[..., 0], axis = -1) * OBJ_SCALE

    noobj_mask = (1 - obj_mask) * NO_OBJ_SCALE

    

    loss_xy    = tf.reduce_sum(tf.square((true_xy - pred_xy) * obj_mask))

    loss_wh    = tf.reduce_sum(tf.square((true_wh - pred_wh) * obj_mask))

    loss_obj   = tf.reduce_sum(tf.square((true_conf - pred_conf) * obj_mask))

    loss_noobj = tf.reduce_sum(tf.square((true_conf - pred_conf) * noobj_mask))

    

    loss = loss_xy + loss_wh + loss_obj + loss_noobj

    

    tf.print('loss_xy', loss_xy)

    tf.print('loss_wh', loss_wh)

    tf.print('loss_obj', loss_obj)

    tf.print('loss_noobj', loss_noobj)

    tf.print('loss', loss)

 

    return loss    
optimizers = tf.keras.optimizers.Adam(learning_rate=0.00001)



model.compile(

    optimizer = optimizers,

    loss = loss_function,

    metrics = 'accuracy'

)



#history = model.fit(train_generator, validation_data = val_generator, epochs=5)

model.load_weights('../input/first-model-yolov1/first model.h5')
def prediction_to_submission(prediction, ids, threshold = 0.2):

    """ The result of prediction doesn't have the same shape of submission.

    So It scales up to 1024, 

    converts (centre_x, centre_y, width ,height) into (x, y, width, height), 

    groups boundary boxes of images by a relevant id.

    

    Args:

        prediction: the result of prediction from the model

        ids: Ids of the test images

    returns:

        prediction modified as the form of submission

    """

    

    grid_x = prediction.shape[1]

    grid_y = prediction.shape[2]

    

    submission = {}

    

    for i, Id in enumerate(ids):

        List = []

        for j in range(grid_x):

            for k in range(grid_y):

                pred_ = prediction[i,j,k]

                if pred_[0] > threshold:

                    

                    confidence = pred_[0]

                    cell_x = 64 * k

                    cell_y = 64 * j

                    

                    box_width = pred_[3] * 1024

                    box_height = pred_[4] * 1024 

                    

                    box_x = cell_x + (pred_[1] * 64) - (box_width/2)

                    box_y = cell_y + (pred_[2] * 64) - (box_height/2)

                    

                    List.append([confidence, box_x, box_y, box_width, box_height])

                    

        submission[Id] = List

        

    return submission

'To get predictions. Because it takes long time, it needs to be separate for a cell below'



predictions_val = model.predict(val_generator)
'To try to visualize the val_predictions'



id_in_valid = valid_image_ids[2]

submission_val = prediction_to_submission(predictions_val, valid_image_ids, threshold=0.95)

image_1 = Image.open(f'{DIR_INPUT}/train/{id_in_valid}.jpg')

image_2 = image_1.copy()



bbox_true = valid_bboxes[id_in_valid]

image_true = draw_boxes_on_image(image_1, bbox_true)



bbox_pred = np.array(submission_val[id_in_valid])[:,1:]

image_pred = draw_boxes_on_image(image_2, bbox_pred)
fig, ax = plt.subplots(1,2, figsize = (13,13))



ax[1].set_title('pred', fontsize = 17)

ax[1].set_xticks([])

ax[1].set_yticks([])

ax[1].imshow(image_pred)



ax[0].set_title('true', fontsize = 17)

ax[0].set_xticks([])

ax[0].set_yticks([])

ax[0].imshow(image_true)
'submission'



test_albu = albu.Compose([

    albu.CLAHE(p=1),

    albu.ToGray(p=1)

])



test_image_ids = os.listdir(f'{DIR_INPUT}/test/')

test_image_ids = [ Id[:-4] for Id in test_image_ids ]

test_images = []



for test_id in test_image_ids:

    test_image = Image.open(f'{DIR_INPUT}/test/{test_id}.jpg').resize((256,256))

    test_image = np.asarray(test_image)

    test_augment = test_albu(image = test_image)

    test_images.append(test_augment['image'])



test_images = np.asarray(test_images)/256

prediction = model.predict(test_images)

submission = prediction_to_submission(prediction, test_image_ids, threshold = 0.95)
'To try to visualize the predictions'



id_in_test = test_image_ids[0]

bbox = np.array(submission[id_in_test])[:,1:]

image = Image.open(f'{DIR_INPUT}/test/{id_in_test}.jpg')

image = draw_boxes_on_image(image, bbox)

plt.figure(figsize=(8,8))

plt.imshow(image)

submission_list = []

for test_id in test_image_ids:

    prediction_string = []

    for pixel in submission[test_id]:

        c,x,y,w,h = pixel

        prediction_string.append(f'{c} {x} {y} {w} {h}')

    prediction_string = ' '.join(prediction_string)

    submission_list.append([test_id, prediction_string])



final_submission = pd.DataFrame(submission_list , columns = ['image_id', 'PredictionString'])        

#final_submission.to_csv('submission.csv', index = False)