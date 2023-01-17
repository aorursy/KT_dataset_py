import tensorflow as tf

from absl import flags, app



from tensorflow.keras import layers, models, optimizers



import numpy as np

import pandas as pd

import os

import math

import uuid

from tqdm import tqdm

from sklearn.metrics import r2_score, confusion_matrix



from matplotlib import pyplot as plt

import seaborn as sns



import cv2

from PIL import Image
CAMERA_fx = 2304.5479

CAMERA_fy = 2305.8757

CAMERA_cx = 1686.2379

CAMERA_cy = 1354.9849



CAMERA_FOV = (CAMERA_cx / CAMERA_fx)

RADS_PIXEL_X = CAMERA_FOV / 3384



NUM_BINS = 8

IMAGE_INP_SIZE = 64



PATH_DATAFRAME = "../input/car-rotation-crops/rotation-dataset/data.csv"

PATH_IMGS = "../input/car-rotation-crops/rotation-dataset/img/"
def to_angle(radian):

    """Takes a +-π radian and transforms in [0, 2π) range"""

    if radian < 0:

        radian = (2*math.pi) + radian

    return radian



def to_rotation(radian):

    """Takes an angle and transforms it in +-π range """

    base_angle = 0

    

    if math.sin(radian) < 0:

        base_angle = -(2 * math.pi - radian)

    else:

        base_angle = radian - base_angle

    

    return base_angle



def get_local_rot(ray_angle, global_angle):

    return global_angle - ray_angle



def get_global_rot(ray_angle, local_angle):

    return ray_angle + local_angle



def get_bin(angle):

    """Gets bin nb and offset from that number.

    params: 

        - angle: Angle in radians [0, 2π)

    """

    bin_size = 360 / NUM_BINS

    total_bins = 360//bin_size

    

    degrees = math.degrees(angle) + bin_size/2  #Shift the bins

    bin_number = (degrees // bin_size) % total_bins

    offset = (degrees - (bin_number*bin_size))

    

    if degrees > 360:  #Correct offset if in last semi bin (8 == 0)

        offset = degrees - ((total_bins) * bin_size)

    

    offset = math.radians(offset)



    return bin_number, offset



def prediction_to_yaw(bin_nb, offset, ray_angle):

    """ Takes bin + offset and using the ray angle 

    returns the global rotation of the car """

    bin_size = 2*math.pi / NUM_BINS

    

    # Local rotation of the car in [0, 2π)

    angle = bin_nb * bin_size + offset - bin_size/2  # shift bins

    

    # Global rotation of the car (taking into account the camera ray angle)

    angle = get_global_rot(ray_angle, angle)

    

    if angle < 0:

        angle = 2*math.pi + angle

    

    # Represent the angle as a rotation of [0, π] or [0, -π)

    #angle = to_rotation(angle)

    

    return angle



def angle_distance(angle1, angle2):

    """Returns the shortest distance in degrees between two angles.

    Parameters:

        - angle1, angle2: (Degrees)

    """

    diff = ( angle2 - angle1 + 180 ) % 360 - 180;

    diff =  diff + 360  if diff < -180 else diff

    return diff
dataset = pd.read_csv(PATH_DATAFRAME).drop('Unnamed: 0', axis=1)

dataset['Name'] = dataset.apply(lambda r: PATH_IMGS + r['Name'] +'.jpeg', axis=1)



# Get global angle expressed in [0, 2π] range

dataset['Global_angle'] = dataset.apply(lambda r: to_angle(r['Global']), axis=1)



# Get ray angle expressed in [0, 2π] range

dataset['Ray_Angle_angle'] = dataset.apply(lambda r: to_angle(r['Ray_Angle']), axis=1)



# Calculate Local rotation

dataset['Local'] = dataset['Global_angle'] - dataset['Ray_Angle_angle']

# Correct the local angle in case the substraction is < 0 (express it in range 0,2π)

dataset['Local_corrected'] = dataset.apply(lambda r: to_angle(r['Local']), axis=1)



# Get Bins + Offsets

dataset[['Bin_nb', 'Bin_offset']] = dataset.apply(lambda r: pd.Series(get_bin(r['Local_corrected'])), axis=1)

dataset['Bin_nb'] = dataset['Bin_nb'].astype('int')



#Normalize bin offset

max_off = dataset['Bin_offset'].max()

dataset['Bin_offset_norm'] = dataset['Bin_offset'] / max_off# if max_off > 1 else dataset['Bin_offset']
# Balance bin number to reduce (as much as possible) biases. Cars in 0/4 (facing frontwards/backwards) are the majority

bins0 = dataset.loc[dataset['Bin_nb'] == 0].head(5000)

bins4 = dataset.loc[dataset['Bin_nb'] == 4].head(5000)

others = dataset.loc[(dataset['Bin_nb'] != 0) & (dataset['Bin_nb'] != 4)]



# Over represent the classes

new_dataset_rows = [bins0, bins4]

new_dataset_rows.extend([others] * 10)



dataset = pd.concat(new_dataset_rows).sample(frac=1)
# A crop example

Image.open(dataset.iloc[115,0])
fig, ax = plt.subplots(3, figsize=(12,7))



sns.distplot(dataset['Ray_Angle_angle'], ax=ax[0], bins=360)

ax[0].set_title("Ray_Angle_angle")



sns.distplot(dataset['Bin_nb'], ax=ax[1], kde=False)

ax[1].set_title("Bin_nb")



sns.distplot(dataset['Bin_offset_norm'], ax=ax[2])

ax[2].set_title("Bin_offset_norm")
# Train / Validation / Test split

train_mask = np.random.rand(len(dataset)) < 0.8



df_train = dataset[train_mask]

df_test = dataset[~train_mask]



valid_mask = np.random.rand(len(df_train)) < 0.1

df_val = df_train[valid_mask]

df_train = df_train[~valid_mask]
def proc_image(path, bin_nb, offset):

    image = tf.io.read_file(path)

    oh_bin = tf.one_hot(bin_nb, NUM_BINS)

    #offset /= 2 #Normalize

    

    #image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.decode_jpeg(image, channels=1)

    image = tf.image.convert_image_dtype(image, tf.float32)

    # resize the image to the desired size.

    image = tf.image.resize(image, [IMAGE_INP_SIZE, IMAGE_INP_SIZE])

    

    return image, (oh_bin, offset)



dataset = tf.data.Dataset.from_tensor_slices((df_train['Name'].values, df_train['Bin_nb'].values, df_train['Bin_offset_norm'].values))

dataset = dataset.map(proc_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.batch(8).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)





dataset_valid = tf.data.Dataset.from_tensor_slices((df_val['Name'].values, df_val['Bin_nb'].values, df_val['Bin_offset_norm'].values))

dataset_valid = dataset_valid.map(proc_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset_valid = dataset_valid.batch(8).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



steps_per_epoch_train = len(df_train) // 8

validation_steps = len(df_val) // 8
data_test = tf.data.Dataset.from_tensor_slices((df_test['Name'].values, df_test['Bin_nb'].values, df_test['Bin_offset_norm'].values))

data_test = data_test.map(proc_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

data_test = data_test.batch(8).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



steps_per_epoch_test = len(df_test) // 8
tf.keras.backend.clear_session()
def res_block(previous):

    conv1 = layers.Conv2D(256, (1,1), activation='relu', kernel_regularizer='l2', padding='same')(previous)

    conv2 = layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer='l2', padding='same')(conv1)

    conv3 = layers.Conv2D(128, (1,1), activation='relu', kernel_regularizer='l2', padding='same')(conv2)

    concat = layers.Concatenate()([previous, conv3])

    concat = layers.BatchNormalization()(concat)

    return concat



inp = layers.Input(shape=(IMAGE_INP_SIZE,IMAGE_INP_SIZE,1))

conv1 = layers.Conv2D(128, (4,4), activation='relu', strides=2)(inp)

pool1 = layers.MaxPool2D()(conv1)

block1 = res_block(pool1)

pool2 = layers.MaxPool2D((2,2))(block1)

block2 = res_block(pool2)

pool3 = layers.MaxPool2D((2,2))(block2)

#block3 = res_block(pool3)



flat = layers.Flatten()(pool3)

model_output = layers.Dense(500, activation='linear')(flat)



out_offset = layers.Dense(500, activation='relu')(model_output)

out_offset = layers.Dropout(.2)(out_offset)

out_offset = layers.Dense(300, activation='linear')(out_offset)

out_offset = layers.Dense(1, activation='sigmoid', name='out_offset')(out_offset)



out_bin = layers.Dense(500, activation='relu')(model_output)

out_bin = layers.Dropout(.2)(out_bin)

out_bin = layers.Dense(200, activation='linear')(out_bin)

out_bin = layers.Dense(NUM_BINS, activation='softmax', name='out_bin')(out_bin)





model = models.Model(inputs=[inp], outputs=[out_bin, out_offset])



callbacks = [

    tf.keras.callbacks.TerminateOnNaN(),

    tf.keras.callbacks.EarlyStopping(monitor='val_out_bin_acc', patience=2, min_delta=0.001, mode='max', restore_best_weights=True)

]



metrics = {

    'out_bin': ['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

}



model.compile(optimizers.Adam(lr=1e-4), loss=['categorical_crossentropy', 'mse'], loss_weights=[0.5, 0.5], 

              metrics=metrics)
hist = model.fit(dataset,

                 epochs = 20,

                 callbacks = callbacks,

                 steps_per_epoch=steps_per_epoch_train, 

                 validation_data=dataset_valid, 

                 validation_steps=validation_steps)
fig, ax = plt.subplots(2, figsize=(8,5))

ax[0].plot(hist.history['val_out_bin_acc'], label='val_acc')

ax[0].plot(hist.history['out_bin_acc'], label='train_acc')

ax[0].grid(True)

ax[0].legend()



ax[1].plot(hist.history['val_out_offset_loss'], label='val_offset')

ax[1].plot(hist.history['out_offset_loss'], label='train_offset')

ax[1].grid(True)

ax[1].legend()



fig.suptitle("Train history")
test_results = model.predict(data_test, steps=steps_per_epoch_test, verbose=True)
#Denormalize offset 

test_results[1] *= max_off
index = 29

image = Image.open(df_test.iloc[index]['Name'])

p_off = test_results[1][index]

p_bin = test_results[0][index]



real_off = df_test.iloc[index]['Bin_offset']

real_bin = df_test.iloc[index]['Bin_nb']





print(f"Real offset: {real_off}")

print(f"Real bin: {real_bin}")

print(f"REAL ANGLE {math.degrees(df_test.iloc[index]['Global_angle'])}")



print("-"*30)

bin_nb = np.argmax(p_bin)

offset = p_off

ray_angle = df_test.iloc[index]['Ray_Angle']



print(f"Predicted bin: {bin_nb}")

print(f"Predicted offset: {offset}")

print(f"PREDICTED ANGLE {math.degrees(prediction_to_yaw(bin_nb, offset, ray_angle)[0])}")



display(image)
pred_angles = []

real_angles = []



real_bins = []

predicted_bins = []



for i in range(len(test_results[1])):

    p_off = test_results[1][i]

    p_bin = test_results[0][i]

    ray_angle = df_test.iloc[i]['Ray_Angle']

    

    bin_nb = np.argmax(p_bin)

    offset = p_off

    

    ang = math.degrees(prediction_to_yaw(bin_nb, offset, ray_angle)[0])

    

    pred_angles.append(ang)

    real_angles.append(math.degrees(df_test.iloc[i]['Global_angle']))

    

    real_bins.append(df_test.iloc[i]['Bin_nb'])

    predicted_bins.append(bin_nb)
cm = confusion_matrix(real_bins, predicted_bins)

df_cm = pd.DataFrame(cm)

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=(8, 8))

ax.set_title('Confusion Matrix')

sns.heatmap(cm, cmap="Blues", annot=True, ax=ax, annot_kws={"size": 9})

ax.set_ylabel('Actual bin')

ax.set_xlabel('Predicted bin');
fig, ax = plt.subplots(1)

ax.scatter(real_angles, pred_angles, alpha=0.5, s=2);
errors = [angle_distance(a1, a2) for a1,a2 in zip(pred_angles, real_angles)]
fig, ax = plt.subplots(1, figsize=(11,8))

sns.distplot(errors, ax=ax)

ax.set_title("Error distribution")

ax.set_xlabel("Degrees")



print(f"Avg. error: {np.mean(np.abs(errors))}")

print(f"Median error: {np.median(errors)}")

print(f"Max. error: {np.max(np.abs(errors))}")

print(f"SD error: {np.std(errors)}")
# Save model if needed

model.save('yaw.h5')