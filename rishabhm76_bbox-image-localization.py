import os

import glob
image_path = glob.glob('/kaggle/input/image-localization-dataset/training_images/*.jpg')
import numpy as np

from PIL import Image, ImageDraw
input_dimen = 228

images = []



for imagefile in image_path:

    image = Image.open(imagefile).resize((input_dimen, input_dimen))

    image = np.asarray(image) / 255.0

    images.append(image)
import xmltodict



bboxes = []

classes_raw = []

annotations_path = glob.glob('/kaggle/input/image-localization-dataset/training_images/*.xml')



for xmlfile in annotations_path:

    x = xmltodict.parse(open(xmlfile, 'rb'))

    bndbox = x['annotation']['object']['bndbox']

    bndbox = np.array([ int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])])

    bndbox2 = [None] * 4

    bndbox2[0] = bndbox[0]

    bndbox2[1] = bndbox[1]

    bndbox2[2] = bndbox[2]

    bndbox2[3] = bndbox[3]

    bndbox2 = np.array(bndbox2)/input_dimen

    bboxes.append(bndbox2)

    classes_raw.append( x['annotation']['object']['name'])
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
boxes = np.array(bboxes)

encoder = LabelBinarizer()

classes_onehot = encoder.fit_transform(classes_raw)



y = np.concatenate( [boxes, classes_onehot], axis=1)

x = np.array(images)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
from keras import backend as K
def calculate_iou( target_boxes, pred_boxes ):

    xA = K.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )

    yA = K.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )

    xB = K.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )

    yB = K.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )

    

    interArea = K.maximum( 0.0, xB - xA ) * K.maximum( 0.0, yB - yA )

    boxA_Area = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])

    boxB_Area = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])

    

    iou = interArea / ( boxA_Area + boxB_Area - interArea)

    return iou
import tensorflow as tf



def custom_loss( y_true, y_pred):

    mse = tf.losses.mean_squared_error( y_true, y_pred)

    iou = calculate_iou( y_true, y_pred)

    return mse + ( 1 - iou )
def iou_metric( y_true, y_pred):

    return calculate_iou(y_true, y_pred)
import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.optimizers import Adam

from keras.constraints import max_norm

from keras.initializers import he_uniform

from keras.regularizers import l2
num_classes = 3

pred_vector_length = 4 + num_classes

alpha = 0.25

input_shape = (input_dimen, input_dimen, 3)



model = keras.Sequential()



'''



strides=1, padding = 'same', kernel_initializer = he_uniform(), kernel_regularizer = l2(0.001)



'''



model.add(keras.layers.Conv2D(16, kernel_size = (3,3), input_shape = input_shape))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(16, kernel_size = (3,3)))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

model.add(keras.layers.Dropout(0.2))



model.add(keras.layers.Conv2D(32, kernel_size = (3,3)))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(32, kernel_size = (3,3)))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

model.add(keras.layers.Dropout(0.2))



model.add(keras.layers.Conv2D(64, kernel_size = (3,3)))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, kernel_size = (3,3)))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

model.add(keras.layers.Dropout(0.2))



model.add(keras.layers.Conv2D(128, kernel_size = (3,3)))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(128, kernel_size = (3,3)))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

model.add(keras.layers.Dropout(0.2))



model.add(keras.layers.Conv2D(256, kernel_size = (3,3)))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(256, kernel_size = (3,3)))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

model.add(keras.layers.Dropout(0.2))



model.add(keras.layers.Flatten())



model.add(keras.layers.Dense(1280))

model.add(keras.layers.LeakyReLU(alpha=alpha))

model.add(keras.layers.Dense(640))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(480))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(120))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(62))

model.add(keras.layers.LeakyReLU(alpha=alpha))

#model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Dense(pred_vector_length))

model.add(keras.layers.LeakyReLU(alpha=alpha))
adam = Adam(learning_rate=0.0001, beta_1 = 0.9, beta_2 = 0.999, amsgrad=False)



model.compile(optimizer = adam, loss = custom_loss, metrics = [iou_metric])
'''

from keras.preprocessing.image import ImageDataGenerator



datagenerator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

gen = datagenerator.flow(x_train, y_train, batch_size=3)

steps = int(x_train.shape[0] / 3)



es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience=20)

mc = ModelCheckpoint('best_model.h5', monitor='val_iou_metric', mode='max', verbose=1, save_best_only=True)



history = model.fit_generator(gen, steps_per_epoch = steps, validation_data = (x_test, y_test), epochs = 120, callbacks=[es, mc], verbose=2)



_, acc = model.evaluate_generator(gen, steps = steps, callbacks=[es, mc], verbose=2)

print('> %.3f' % (acc*100.0))



'''
es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=0, patience=20)

mc = ModelCheckpoint('best_model.h5', monitor='val_iou_metric', mode='max', verbose=0, save_best_only=True)
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 80, batch_size = 2, callbacks=[es, mc], verbose=2)
_, acc = model.evaluate(x_train, y_train, callbacks=[es, mc], verbose=0)

print('> %.3f' % (acc*100.0))
import matplotlib.pyplot as plt



def summary_plot(history):

    # plot loss

    plt.subplot(211)

    plt.title('Custom Loss')

    plt.plot(history.history['loss'], color='blue', label='train')

    plt.plot(history.history['val_loss'], color='red', label='test')

    # plot Accuracy

    plt.subplot(212)

    plt.title('Iou Metric')

    plt.plot(history.history['iou_metric'], color='blue', label='train')

    plt.plot(history.history['val_iou_metric'], color='red', label='test')





summary_plot(history)
!rm -r inference_images

!mkdir -v inference_images



boxes = model.predict( x_test )

for i in range( boxes.shape[0] ):

    b = boxes[ i , 0 : 4 ] * input_dimen

    img = x_test[i] * 255

    source_img = Image.fromarray( img.astype( np.uint8 ) , 'RGB' )

    draw = ImageDraw.Draw( source_img )

    draw.rectangle( b , outline="black" )

    source_img.save( 'inference_images/image_{}.png'.format( i + 1 ) , 'png' )
def calculate_avg_iou( target_boxes , pred_boxes ):

    xA = np.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )

    yA = np.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )

    xB = np.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )

    yB = np.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )

    interArea = np.maximum(0.0, xB - xA ) * np.maximum(0.0, yB - yA )

    boxAArea = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])

    boxBArea = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])

    iou = interArea / ( boxAArea + boxBArea - interArea )

    return iou



target_boxes = y_test * input_dimen

pred = model.predict( x_test )

pred_boxes = pred[ ... , 0 : 4 ] * input_dimen



iou_scores = calculate_avg_iou( target_boxes , pred_boxes )

print( 'Mean IOU score {}'.format( iou_scores.mean() ) )