import pandas as pd
import numpy as np
import pydicom
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.layers import SpatialDropout2D
from keras.models import Model
import cv2
import random
input_size = 256
batch_size = 8
df_labels = pd.read_csv("../input/stage_1_train_labels.csv")
df_class  = pd.read_csv("../input/stage_1_detailed_class_info.csv")
df = df_labels
df["class"] = df_class["class"]
ages = []
sexes = []
for p_id in df_labels["patientId"]:
    ds = pydicom.read_file("../input/stage_1_train_images/{}.dcm".format(str(p_id)))
    ages.append(ds.PatientAge)
    sexes.append(ds.PatientSex)
df["age"] = ages
df["sex"] = sexes

ids = df["patientId"].values
ids_train, ids_test = train_test_split(ids)
df.head()
input_img = Input((input_size, input_size, 1))

x = Conv2D(32, (3, 3), activation="relu")(input_img)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation="relu")(x)
# x = GaussianNoise(0.2)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Dropout(0.1)(x)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = GaussianNoise(0.2)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Dropout(0.1)(x)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = BatchNormalization()(x)
# x = GaussianNoise(0.2)(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)

input_extra = Input((3,))
x = Concatenate()([input_extra, x])
x = Dropout(0.2)(x)
x = Dense(1024, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)

extra_model = Model([input_img, input_extra], x)
extra_model.compile(optimizer="ADAM", loss="binary_crossentropy", metrics=["accuracy"])

def random_crop(img, crop_max=0.1, shape=input_size):

    crop_max = 128*crop_max
    crop_x1 = int(random.uniform(0, crop_max))
    crop_x2 = int(random.uniform(0, crop_max-crop_x1))
    crop_y1 = int(random.uniform(0, crop_max))
    crop_y2 = int(random.uniform(0, crop_max-crop_y1))
    # cv2.imshow("", img)
    # cv2.waitKey()
    img = img[crop_x1:128-crop_x2, crop_y1:128-crop_y2]
    img = cv2.resize(img, (shape, shape))
    # cv2.imshow("", img)
    # cv2.waitKey()
    return img

def train_generator_extra(ids, train = True):
    while True:
        for start in range(0, len(ids), batch_size):
            x_batch = []
            y_batch = []
            age_sex = []
            end = min(start + batch_size, len(ids))
            ids_train_batch = ids[start:end]
            for id in ids_train_batch:
                ds = pydicom.read_file("../input/stage_1_train_images/{}.dcm".format(id))
                img = ds.pixel_array
                img = cv2.resize(img, (input_size, input_size))
                img = random_crop(img)
                img = np.expand_dims(img, axis=2)
                
                row = df.loc[df["patientId"]==id].values[0]
                y_val = int(row[5])
                
                age_sex_val = [1, 0] if row[8] == "F" else [0, 1]
                age_sex_val.append(int(row[7])/100)
                age_sex.append(age_sex_val)
                x_batch.append(img)
                y_batch.append(y_val)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32)
            age_sex_batch = np.array(age_sex, np.float32)
            if train:
                yield [x_batch, age_sex_batch], y_batch
            else:
                yield [x_batch, age_sex_batch]
extra_model.fit_generator(train_generator_extra(ids_train), steps_per_epoch=int(len(ids_train)/batch_size), epochs=2,
                          validation_data=train_generator_extra(ids_test), validation_steps=int(len(ids_test)/batch_size))
def u_generator_extra(ids, train = True):
    while True:
        for start in range(0, len(ids), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids))
            ids_train_batch = ids[start:end]
            for id in ids_train_batch:
                ds = pydicom.read_file("../input/stage_1_train_images/{}.dcm".format(id))
                img = ds.pixel_array
                img = cv2.resize(img, (input_size, input_size))
                img = random_crop(img)
                img = np.expand_dims(img, axis=2)
                
                row = df.loc[df["patientId"]==id]
                mask = np.zeros((input_size, input_size))
                if not row.isnull().values.any():
                    for box in row.values:
                        mask[int(box[2]/4):int(box[2]/4+round(box[4]/4)), int(box[1]/4):int(box[1]/4+round(box[3]/4))] = np.ones((int(round(box[4]/4)), int(round(box[3]/4))))
                
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32)
            if train:
                yield x_batch, y_batch
            else:
                yield x_batch
from keras.losses import binary_crossentropy
import keras.backend as K


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def get_unet_256(input_shape=(256, 256, 1),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer="ADAM", loss=bce_dice_loss, metrics=[dice_coeff])

    return model
pred = extra_model.predict_generator(train_generator_extra(ids_train, train=False), steps=np.ceil(float(len(ids_train)) / float(batch_size)),)
hard_ids = [ids_train[index] for index in range(len(ids_train)) if pred[index]<0.5]
model = get_unet_256()
model.fit_generator(u_generator_extra(hard_ids), steps_per_epoch = int(len(hard_ids)/batch_size))
model.fit_generator(u_generator_extra(hard_ids), steps_per_epoch = int(len(hard_ids)/batch_size), epochs=1)
import cv2

# load and shuffle filenames
folder = '../input/stage_1_test_images'
test_filenames = os.listdir(folder)
print('n test samples:', len(test_filenames))
test_ids = [name[:-4] for name in test_filenames]

submission_dict = {}

pred = extra_model.predict_generator(train_generator_extra(ids_train, train=False), steps=np.ceil(float(len(ids_train)) / float(batch_size)),)
for index, id_ in enumerate(test_ids):
    
    if pred[index] < 0.5:
        submission_dict[id_] = ""
    else:
        mask = model.predict([folder+test_filenames[index]])[0]
        im_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
        _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        boxstring = ""
        for rect in rects:
            boxstring += str[rect[0]]+ " " + str[rect[1]]+ " " + str[rect[2]]+ " " + str[rect[3]]
        submission_dict[id_]  = boxstring
# save dictionary as csv file
sub = pd.DataFrame.from_dict(submission_dict,orient='index')
sub.index.names = ['patientId']
sub.columns = ['PredictionString']
sub.to_csv('submission.csv')
