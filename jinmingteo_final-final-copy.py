import os

import PIL

import pickle

import numpy as np

from tqdm import tqdm

from math import log, exp

from random import shuffle

from skimage.transform import resize

from IPython.display import Image, display

from PIL import ImageEnhance, ImageFont, ImageDraw



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.python.keras.utils.data_utils import Sequence



tf.keras.backend.clear_session()  # For easy reset of notebook state.



input_shape = (224,224,3)



# Edit these to suit your environment

base_folder = '/kaggle/input/til2020-test/'

output_folder = '/kaggle/working/'

model_save_folder = os.path.join( output_folder, 'saved_models' )

data_folder = os.path.join( base_folder, 'voc-cat-dog' )

# These are how my pickled data chunks are stored. If you have a different way, change it here.

data_split_template = '{}-voc-catdog-data-pil-set'.format( tuple(input_shape[:2]) )

data_split_template = data_split_template + '{}.p'
'''

Augmentation methods. We need to implement our own augmentation because native support in keras does not change the bounding box 

labels for us as the image is altered. We need to do it ourselves.

'''

# Helper method: Computes the boundary of the image that includes all bboxes

def compute_reasonable_boundary(labels):

    bounds = [ (x-w/2, x+w/2, y-h/2, y+h/2) for _,x,y,w,h in labels]

    xmin = min([bb[0] for bb in bounds])

    xmax = min([bb[1] for bb in bounds])

    ymin = min([bb[2] for bb in bounds])

    ymax = min([bb[3] for bb in bounds])

    return xmin, xmax, ymin, ymax



def aug_horizontal_flip(img, labels):

    flipped_labels = []

    for c,x,y,w,h in labels:

        flipped_labels.append( (c,1-x,y,w,h) )

    return img.transpose(PIL.Image.FLIP_LEFT_RIGHT), np.array(flipped_labels)



def aug_crop(img, labels):

    # Compute bounds such that no boxes are cut out

    xmin, xmax, ymin, ymax = compute_reasonable_boundary(labels)

    # Choose crop_xmin from [0, xmin]

    crop_xmin = max( np.random.uniform() * xmin, 0 )

    # Choose crop_xmax from [xmax, 1]

    crop_xmax = min( xmax + (np.random.uniform() * (1-xmax)), 1 )

    # Choose crop_ymin from [0, ymin]

    crop_ymin = max( np.random.uniform() * ymin, 0 )

    # Choose crop_ymax from [ymax, 1]

    crop_ymax = min( ymax + (np.random.uniform() * (1-ymax)), 1 )

    # Compute the "new" width and height of the cropped image

    crop_w = crop_xmax - crop_xmin

    crop_h = crop_ymax - crop_ymin

    cropped_labels = []

    for c,x,y,w,h in labels:

        c_x = (x - crop_xmin) / crop_w

        c_y = (y - crop_ymin) / crop_h

        c_w = w / crop_w

        c_h = h / crop_h

        cropped_labels.append( (c,c_x,c_y,c_w,c_h) )



    W,H = img.size

    # Compute the pixel coordinates and perform the crop

    impix_xmin = int(W * crop_xmin)

    impix_xmax = int(W * crop_xmax)

    impix_ymin = int(H * crop_ymin)

    impix_ymax = int(H * crop_ymax)

    return img.crop( (impix_xmin, impix_ymin, impix_xmax, impix_ymax) ), np.array( cropped_labels )



def aug_translate(img, labels):

    # Compute bounds such that no boxes are cut out

    xmin, xmax, ymin, ymax = compute_reasonable_boundary(labels)

    trans_range_x = [-xmin, 1 - xmax]

    tx = trans_range_x[0] + (np.random.uniform() * (trans_range_x[1] - trans_range_x[0]))

    trans_range_y = [-ymin, 1 - ymax]

    ty = trans_range_y[0] + (np.random.uniform() * (trans_range_y[1] - trans_range_y[0]))



    trans_labels = []

    for c,x,y,w,h in labels:

        trans_labels.append( (c,x+tx,y+ty,w,h) )



    W,H = img.size

    tx_pix = int(W * tx)

    ty_pix = int(H * ty)

    return img.crop( (0, 0, W-tx_pix, H-ty_pix) ), np.array( trans_labels )



def aug_colorbalance(img, labels, color_factors=[0.2,2.0]):

    factor = color_factors[0] + np.random.uniform() * (color_factors[1] - color_factors[0])

    enhancer = ImageEnhance.Color(img)

    return enhancer.enhance(factor), labels



def aug_contrast(img, labels, contrast_factors=[0.2,2.0]):

    factor = contrast_factors[0] + np.random.uniform() * (contrast_factors[1] - contrast_factors[0])

    enhancer = ImageEnhance.Contrast(img)

    return enhancer.enhance(factor), labels



def aug_brightness(img, labels, brightness_factors=[0.2,2.0]):

    factor = brightness_factors[0] + np.random.uniform() * (brightness_factors[1] - brightness_factors[0])

    enhancer = ImageEnhance.Brightness(img)

    return enhancer.enhance(factor), labels



def aug_sharpness(img, labels, sharpness_factors=[0.2,2.0]):

    factor = sharpness_factors[0] + np.random.uniform() * (sharpness_factors[1] - sharpness_factors[0])

    enhancer = ImageEnhance.Sharpness(img)

    return enhancer.enhance(factor), labels



# Performs no augmentations and returns the original image and bbox. Used for the validation images.

def aug_identity(pil_img, label_arr):

    return np.array(pil_img), label_arr



# This is the default augmentation scheme that we will use for each training image.

def aug_default(img, labels, p={'flip':0.5, 'crop':0.2, 'translate':0.2, 'color':0.2, 'contrast':0.2, 'brightness':0.2, 'sharpness':0.2}):

    if p['color'] > np.random.uniform():

        img, labels = aug_colorbalance(img, labels)

    if p['contrast'] > np.random.uniform():

        img, labels = aug_contrast(img, labels)

    if p['brightness'] > np.random.uniform():

        img, labels = aug_brightness(img, labels)

    if p['sharpness'] > np.random.uniform():

        img, labels = aug_sharpness(img, labels)

    if p['flip'] > np.random.uniform():

        img, labels = aug_horizontal_flip(img, labels)

    if p['crop'] > np.random.uniform():

        img, labels = aug_crop(img, labels)

    if p['translate'] > np.random.uniform():

        img, labels = aug_translate(img, labels)

    return np.array(img), labels
from PIL import Image

im = Image.open(base_folder + "train/train/1000.jpg")

display(im)
W,H = im.size

im.crop( (0, 0, W-100, H-200))
im.rotate(0, translate=(100,200))
im2 = Image.open(base_folder + "train/train/1000.jpg")

draw=ImageDraw.Draw(im2)

draw.rectangle(((150, 300), (300, 400)))

display(im2)
im2 = Image.open(base_folder + "train/train/1000.jpg")

im2 = im2.crop( (0, 0, W-100, H-0) )

draw=ImageDraw.Draw(im2)

draw.rectangle(((150, 300), (300, 400)))

display(im2)
# Each y label has shape of (batch,i,j,7)

def custom_loss(ytrue, ypred):

    obj_loss_weight = 1.0

    cat_loss_weight = 1.0

    loc_loss_weight = 1.0

    # ytrue's first channel is objectness, and it signals where gradients should be considered.

    # So ypred should only take it's predictions seriously where ytrue has a positive, otherwise it should not learn from the negatives.

    objectness_loss = tf.keras.losses.BinaryCrossentropy()( ytrue[:,:,:,:1], ypred[:,:,:,:1] )

    ypred = tf.where( ytrue[:,:,:,:1] != 0, ypred, 0 )

    category_loss = tf.keras.losses.CategoricalCrossentropy() ( ytrue[:,:,:,1:3], ypred[:,:,:,1:3] )

    localisation_loss = tf.keras.losses.Huber() ( ytrue[:,:,:,3:], ypred[:,:,:,3:] )

    return obj_loss_weight*objectness_loss + cat_loss_weight*category_loss + loc_loss_weight*localisation_loss
def basic_detection_model( input_shape, model_name='basic_detection_model' ):

    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same')(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, padding='same')(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, padding='same')(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    block_1_output = layers.MaxPooling2D(2)(x) # 112



    x = layers.Conv2D(64, 3, padding='same')(block_1_output)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, padding='same')(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.add([x, block_1_output])

    x = layers.Conv2D(128, 3, padding='same')(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    block_2_output = layers.MaxPooling2D(2)(x) #56



    x = layers.Conv2D(128, 3, padding='same')(block_2_output)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 3, padding='same')(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.add([x, block_2_output])

    x = layers.Conv2D(256, 3, padding='same')(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    block_3_output = layers.MaxPooling2D(2)(x) #28



    x = layers.Conv2D(256, 3, padding='same')(block_3_output)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 3, padding='same')(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.add([x, block_3_output])

    x = layers.Conv2D(512, 3, padding='same')(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    block_4_output = layers.MaxPooling2D(2)(x) #14



    x = layers.Conv2D(512, 3, padding='same')(block_4_output)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(512, 3, padding='same')(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.add([x, block_4_output])

    x = layers.Conv2D(1024, 3, padding='same')(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    block_5_output = layers.MaxPooling2D(2)(x) #7



    x = layers.Conv2D(512, 3, padding='same')(block_5_output)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(512, 3, padding='valid')(x) #5

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(512, 3, padding='valid')(x) #3

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)



    x = layers.Dropout(0.5)(x)



    objectness_preds = layers.Conv2D(1, 1, activation='sigmoid')(x)

    class_preds = layers.Conv2D(2, 1, activation='softmax')(x)

    bbox_preds = layers.Conv2D(4, 1, activation='tanh')(x)

    predictions = layers.Concatenate()( [objectness_preds, class_preds, bbox_preds] )



    model = keras.Model(inputs, predictions, name=model_name)

    model.compile( optimizer=tf.keras.optimizers.Adam(0.001),

                 loss=custom_loss )

    return model
# Choose whether to start a new model or load a previously trained one

model_context = 'object-detection-tutorial-tanh'

# load_model_path = os.path.join( base_folder, model_save_folder, '{}-best_val_loss.h5'.format(model_context) )

load_model_path = None

if load_model_path is not None:

    model = tf.keras.models.load_model( load_model_path , custom_objects={'custom_loss':custom_loss})

else:

    model = basic_detection_model(input_shape=input_shape, model_name=model_context)



model.summary()
## To implement my own one ##



def reload_data(set_indices, num_sets=None):

    if num_sets is not None:

        shuffle( set_indices )

        selected_indices = set_indices[:num_sets]

        acc = []

    for index in selected_indices:

        set_fp = os.path.join( data_folder, data_split_template.format(index) )

    with open(set_fp, 'rb') as f:

        mini_dataset = pickle.load(f)

        acc.extend(mini_dataset)

    shuffle(acc)

    return acc
class CatDogVocSequence(Sequence):

    def __init__(self, dataset, batch_size, augmentations, dims, input_size=(224,224,3)):

        self.x, self.y = zip(*dataset)

        self.x_acc, self.y_acc = [], []

        self.batch_size = batch_size

        self.augment = augmentations

        self.dims = dims

        self.input_size = input_size



    def __len__(self):

        return int(np.ceil(len(self.x) / float(self.batch_size)))



    # Computes the intersection-over-union (IoU) of two bounding boxes

    def iou(self, bb1, bb2):

        x1,y1,w1,h1 = bb1

        xmin1 = x1 - w1/2

        xmax1 = x1 + w1/2

        ymin1 = y1 - h1/2

        ymax1 = y1 + h1/2



        x2,y2,w2,h2 = bb2

        xmin2 = x2 - w2/2

        xmax2 = x2 + w2/2

        ymin2 = y2 - h2/2

        ymax2 = y2 + h2/2



        area1 = w1*h1

        area2 = w2*h2



        # Compute the boundary of the intersection

        xmin_int = max( xmin1, xmin2 )

        xmax_int = min( xmax1, xmax2 )

        ymin_int = max( ymin1, ymin2 )

        ymax_int = min( ymax1, ymax2 )

        intersection = max(xmax_int - xmin_int, 0) * max( ymax_int - ymin_int, 0 )



        # Remove the double counted region

        union = area1+area2-intersection



        return intersection / union



    '''

    labels: A numpy array of shape (num_labels, 5). num_labels is the number of bounding boxes for the image.

    Each bounding box has entry: c x y w h (class, center-x, center-y, width, height). 



    All numbers are normalized wrt image size: they are in the range [0,1]



    This function inspects each bbox entry and decides how to generate a corresponding array format that the CNN understands.

    '''

    def convert_labels_cxywh_to_arrays(self, labels, iou_threshold=0.5, exceed_thresh_positive=True):

        num_entries = 7 # objectness, p_cat, p_dog, dx, dy, dw, dh

        kx,ky = self.dims

        labels_arr = np.zeros( (kx, ky, num_entries) ) # For this basic model, this is of shape (3,3,7)



        for label in labels:

            # Retrieve the ground-truth class label and bbox

            gtclass, gtx, gty, gtw, gth = label

            gtclass = int(gtclass)

            gt_bbox = [gtx, gty, gtw, gth]



            iou_scores = []



            '''

            There are kx x ky cells. In the basic model, this is 3x3.

            Each cell is of width=gapx and height=gapy

            For the (i,j)-th tile, center-x = (0.5+i)*gapx | center-y = (0.5+j)*gapy

            '''

            gapx = 1.0 / kx

            gapy = 1.0 / ky

          # In this loop, we run through all cells of the 3x3 grid, compute the intersection-over-union w the ground-truth and also the targets to predict.

        for i in range(kx):

            for j in range(ky):

                x = (0.5+i)*gapx

                y = (0.5+j)*gapy



                # These are fixed to the width and height of the square-cell at the moment. However, if we want more anchor boxes of varying aspect ratios, this is the place to change it.

                w = gapx

                h = gapy

                candidate_bbox = [x,y,w,h]



                # Based on the SSD training regime. These are the targets we wish for the CNN to predict at the end.

                # Read the SSD paper: https://arxiv.org/pdf/1512.02325.pdf, for more details.

                dx = (gtx - x) / w 

                dy = (gty - y) / h

                dw = log( gtw / w )

                dh = log( gth / h )



                IoU = self.iou( candidate_bbox, gt_bbox )

                iou_scores.append( (IoU, i, j, dx, dy, dw, dh) )

            

        # Sort by IoU: only the highest IoU scores get included into the resulting label array. Cutoff at threshold.

        iou_scores.sort( key=lambda x: x[0], reverse=True )

        # Always take the top IoU entry

        iou_scores = [iou_scores[0]] + [iou_score for iou_score in iou_scores[1:] if iou_score[0] >= iou_threshold]



        for iou_score in iou_scores:

            # The top IoU score is always included

            IoU, i, j, dx, dy, dw, dh = iou_score

            payload = [IoU, 0, 0, dx,dy,dw,dh]

            payload[gtclass + 1] = 1

            labels_arr[i,j,:] = payload

            if not exceed_thresh_positive:

                break

        return labels_arr



    # Basic preprocessing that can be replaced, if you want to try

    def preprocess_npimg(self, x):

        return x * 1./255.



    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]



        self.x_acc.clear()

        self.y_acc.clear()

        for x,y in zip( batch_x, batch_y ):

            x_aug, y_aug = self.augment( x, y )

            self.x_acc.append( x_aug if x_aug.shape == self.input_size else resize( x_aug, self.input_size[:2] ) )

            y_arr = self.convert_labels_cxywh_to_arrays( y_aug )

            self.y_acc.append( y_arr )



        return self.preprocess_npimg( np.array( self.x_acc ) ), np.array( self.y_acc )
# I broke up my dataset into 10 pickle files because I had issues at the start getting all of them into one giant set. 

# Everything seems okay now, but I have kept this format.

# You can set up your own way of reading in the data.

all_set_ids = list(range(10))

val_set_ids = [0]

train_set_ids = all_set_ids[len(val_set_ids):]



dims = (3,3)

bs = 32

n_epochs = 100



train_dataset = reload_data(train_set_ids)

# Set up training sequence data generator with default augmentation

train_sequence = CatDogVocSequence(train_dataset, bs, aug_default, dims)

val_dataset = reload_data(val_set_ids)

# Set up validation sequence data generator with no augmentation

val_sequence = CatDogVocSequence(val_dataset, bs, aug_identity, dims)



# 3 checkpoints in use - one to save the best val_loss model, one to stop early if no improvement, and one to reduce the learning rate if no improvement.

save_model_path = os.path.join( base_folder, model_save_folder, '{}-best_val_loss.h5'.format(model_context) )

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(

    filepath=save_model_path,

    save_weights_only=False,

    monitor='val_loss',

    mode='auto',

    save_best_only=True)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)



model.fit(x=train_sequence, 

          epochs=n_epochs, 

          batch_size=bs, 

          validation_data=val_sequence, 

          callbacks=[model_checkpoint_callback, earlystopping, reduce_lr],

          )



# Save the final one, if you want

model.save(os.path.join(base_folder, model_save_folder, '{}-final.h5'.format(model_context)))
# Load a previously saved model

model_context = 'object-detection-tutorial-tanh'

saved_model_path = os.path.join( base_folder, model_save_folder, '{}-best_val_loss.h5'.format(model_context) )

model = tf.keras.models.load_model(saved_model_path, custom_objects={'custom_loss':custom_loss})

model.summary()
# Load some data to use on the trained model.

dataset = reload_data([1])
# Converts model tensor outputs into c,x,y,w,h format so that we can display results.

def convert_array_to_cxywh(label_arr, det_threshold=0.1, top=None):

    kx, ky = label_arr.shape[:2]

    gapx = 1. / kx

    gapy = 1. / ky

    # Find the locations of the label_arr where the objectness-score (detection confidence) exceeds the threshold. 

    # These are the detections we will visualize.

    # The lower the threshold, the more false positives we are likely to get.

    eyes, jays = np.where( label_arr[:,:,0] > det_threshold ) #i's and j's are the coordinates of the tensor.

    labels = []

    for i,j in zip(eyes,jays):

        cx = (0.5+i)*gapx

        cy = (0.5+j)*gapy

        w = gapx

        h = gapy



        det_score, p_cat, p_dog, dx, dy, dw, dh = label_arr[i,j]



        # Reverse the targets based on the SSD formulation, to obtain proper x,y,w,h information

        predx = (dx * w) + cx

        predy = (dy * h) + cy

        predw = w * exp( dw )

        predh = h * exp( dh )

        class_str = 'cat' if p_cat > p_dog else 'dog'

        labels.append( (det_score, class_str, predx, predy, predw, predh, i, j) )

    labels.sort( key=lambda x:x[0], reverse=True )

    labels = [(class_str, predx, predy, predw, predh, i, j) for _, class_str, predx, predy, predw, predh, i, j in labels[:top]]

    return labels
# This snippet visualises the result.

colors = ['green', 'blue', 'yellow', 'red', 'cyan', 'magenta', 'white', 'orange', 'brown']

for k in range(10,20):

    pil_img, label_cxywh = dataset[k]

    img_arr = np.array(pil_img) / 255.

    W,H = pil_img.size

    model_pred = model(np.array([img_arr]))[0]

    preds = convert_array_to_cxywh( model_pred, det_threshold=0.9 )



    draw = ImageDraw.Draw(pil_img)

    for cls, x,y,w,h,i,j in preds:

        bb_x = int(x * W)

        bb_y = int(y * H)

        bb_w = int(w * W)

        bb_h = int(h * H)

        left = int(bb_x - bb_w / 2)

        top = int(bb_y - bb_h / 2)

        right = int(bb_x + bb_w / 2)

        bot = int(bb_y + bb_h / 2)

        color = colors[i*3 + j]



        draw.rectangle(((left, top), (right, bot)), outline=color)

        draw.text((bb_x, bb_y), cls, fill=color)



    display(pil_img)