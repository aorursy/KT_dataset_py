#import

import glob, pylab, pandas as pd

import pydicom, numpy as np

import os 

import sys
import matplotlib.pyplot as plt

import keras

from keras.applications import DenseNet121

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam, Nadam

from keras import layers

import tensorflow as tf

from tqdm import tqdm

import cv2

from sklearn.metrics import roc_auc_score, roc_curve
!pip install --upgrade imgaug

import imgaug.augmenters as iaa
data_dir = "../input/rsna-pneumonia-detection-challenge/"

train_image_dir = os.path.join(data_dir, "stage_2_train_images")

test_image_dir = os.path.join(data_dir, "stage_2_test_images")

model_dir = "../output/kaggle/working/"
df = pd.read_csv(os.path.join(data_dir, 'stage_2_train_labels.csv'))

print(df.iloc[0])
#parse data

def parse_data(df):

    """

    Method to read a CSV file (Pandas dataframe) and parse the 

    data into the following nested dictionary:



      parsed = {

        

        'patientId-00': {

            'dicom': path/to/dicom/file,

            'label': either 0 or 1 for normal or pnuemonia, 

            'boxes': list of box(es)

        },

        'patientId-01': {

            'dicom': path/to/dicom/file,

            'label': either 0 or 1 for normal or pnuemonia, 

            'boxes': list of box(es)

        }, ...



      }



    """

    # --- Define lambda to extract coords in list [y, x, height, width]

    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]



    parsed = {}

    for n, row in df.iterrows():

        # --- Initialize patient entry into parsed 

        pid = row['patientId']

        if pid not in parsed:

            parsed[pid] = {

                'dicom': '../input/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % pid,

                'label': row['Target'],

                'boxes': []}



        # --- Add box if opacity is present

        if parsed[pid]['label'] == 1:

            parsed[pid]['boxes'].append(extract_box(row))



    return parsed
parsed = parse_data(df)
#visualize box

def draw(data):

    """

    Method to draw single patient with bounding box(es) if present 



    """

    # --- Open DICOM file

    d = pydicom.read_file(data['dicom'])

    im = d.pixel_array



    # --- Convert from single-channel grayscale to 3-channel RGB

    im = np.stack([im] * 3, axis=2)



    # --- Add boxes with random color if present

    for box in data['boxes']:

        rgb = np.floor(np.random.rand(3) * 256).astype('int')

        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)



    pylab.imshow(im, cmap=pylab.cm.gist_gray)

    pylab.axis('off')



def overlay_box(im, box, rgb, stroke=1):

    """

    Method to overlay single box on image



    """

    # --- Convert coordinates to integers

    box = [int(b) for b in box]

    

    # --- Extract coordinates

    y1, x1, height, width = box

    y2 = y1 + height

    x2 = x1 + width



    im[y1:y1 + stroke, x1:x2] = rgb

    im[y2:y2 + stroke, x1:x2] = rgb

    im[y1:y2, x1:x1 + stroke] = rgb

    im[y1:y2, x2:x2 + stroke] = rgb



    return im
draw(parsed['00436515-870c-4b36-a041-de91049b9ab4'])
# classes

df_detailed = pd.read_csv(os.path.join(data_dir, 'stage_2_detailed_class_info.csv'))

summary = {}

for n, row in df_detailed.iterrows():

    if row['class'] not in summary:

        summary[row['class']] = 0

    summary[row['class']] += 1
#count

print("numbers of train samples:", len([name for name in os.listdir(train_image_dir)]))

print("numbers of test samples:", len([name for name in os.listdir(test_image_dir)]))

print("numbers of train labels:", df.shape[0])

print("numbers of train detailed labels:", df_detailed.shape[0])

print("classes:", summary)
#number of unique df

df_unique = df["patientId"].unique()

df_unique.shape
df.head()
#visualize numbers of patients with variable targets

df_target_num = df.groupby("patientId").agg("sum")

df_target_num.reset_index(inplace=True)

df_target_num["Target"].value_counts()
#create dataset for classifier

df_target_num["hasMask"] = df_target_num["Target"] != 0

df_target_num["hasMask"].value_counts()

# df_target_num.loc[df_target_num["hasMask"], "hasMask"] = "Yes"

# df_target_num.loc[df_target_num["hasMask"] == False, "hasMask"] = "No"
df_target_num.head()
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(df_target_num, test_size=0.15, stratify=df_target_num["hasMask"])
df_test = pd.DataFrame([name for name in os.listdir(test_image_dir)], columns=["patientId"])

df_test.head()
# #read data for passing to generator

# def load_image(data, directory):

#     d = pydicom.read_file(directory + "/" + data + ".dcm")

#     im = d.pixel_array



#     # --- Convert from single-channel grayscale to 3-channel RGB

#     im = np.stack([im] * 3, axis=2)

#     return im

# train_data = [load_image(d, train_image_dir) for d in df_unique]

# test_data = [load_image(d, test_image_dir) for d in df_unique]
#create datagenerator





# def create_train_generator():

#     return ImageDataGenerator(

#         zoom_range=0.1,

#         fill_mode="constant",#One of {"constant", "nearest", "reflect" or "wrap"}. Points outside the boundaries of the input are filled

#         cval=0.,#Float or Int. Value used for points outside the boundaries when fill_mode = "constant".

#         rotation_range=10,

#         height_shift_range=0.1,

#         width_shift_range=0.1,

#         horizontal_flip=True,

#         vertical_flip=True,

#         rescale=1/255.,

#         validation_split=0.15        

#     )



# def create_train_flow(datagen, df, seed, **dflow_args):

#     flow = datagen.flow_from_directory(

#         directory=train_image_dir,

#         class_mode = 'sparse',

#         seed = seed,

#         **dflow_args

#     )

#     flow.filenames = train_image_dir + "/" + df["patientId"] + ".dcm"

#     print(flow.filenames.values[1])

#     print(df["patientId"].values[1])

#     flow.classes = df["hasMask"]

#     flow.samples = df.shape[0]

#     flow.n = df.shape[0]

#     flow._set_index_array()

#     #flow.directory = '' # since we have the full path

#     print('Reinserting dataframe: {} images'.format(df.shape[0]))

#     return flow





# def create_train_flow(datagen, subset, seed):

#     return datagen.flow_from_dataframe(

#         df_target_num,

#         directory=os.path.join(data_dir, "stage_2_train_images"),

#         x_col="patientId",

#         y_col="hasMask",

#         color_mode='grayscale',

#         class_mode="sparse",

#         target_size=(256, 256),

#         batch_size=BATCH_SIZE,

#         subset=subset,

#         seed=seed

#     )



# def create_test_flow():

#     return ImageDataGenerator(rescale=1/255.).flow_from_dataframe(

#         df_test,

#         directory=os.path.join(data_dir, "stage_2_test_images"),

#         x_col="patientId",

#         color_mode='grayscale',

#         class_mode=None,

#         target_size=(256, 256),

#         batch_size=BATCH_SIZE,

#         shuffle=False

#     )



# data_generator = create_train_generator()

# train_gen = create_train_flow(data_generator, train_df, None, color_mode='rgb', batch_size=BATCH_SIZE)

#val_gen = create_train_flow(data_generator, df_target_num, "validation", None, color_mode='rgb', batch_size=BATCH_SIZE)

# test_gen = create_test_flow()
# t_x, t_y = next(train_gen)

# print(t_x.shape, t_y.shape)

# fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))

# for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):

#     c_ax.imshow(c_x[:,:,0], cmap = 'bone')

#     c_ax.set_title('%s' % class_enc.classes_[np.argmax(c_y)])

#     c_ax.axis('off')
#create generator

class DataGenerator(keras.utils.Sequence):

    def __init__(self, df, id_col, label_col, image_dir, batch_size=32,

                 img_h=256, img_w=512, phase='train', shuffle=True):

        

        self.list_ids = df[id_col].values

        self.list_ids = [id1 + ".dcm" for id1 in self.list_ids]

        self.labels = {(row[1][id_col] + ".dcm"):row[1][label_col] for row in df[[id_col, label_col]].iterrows()}

        self.image_dir = image_dir

        self.batch_size = batch_size

        self.img_h = img_h

        self.img_w = img_w

        self.phase = phase

        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):

        'denotes the number of batches per epoch'

        return int(np.floor(len(self.list_ids)) / self.batch_size)

    

    def __getitem__(self, index):

        'generate one batch of data'

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # get list of IDs

        list_ids_temp = [self.list_ids[k] for k in indexes]

        # generate data

        X, y = self.__data_generation(list_ids_temp)

        # return data 

        return X, y

    

    def on_epoch_end(self):

        'update ended after each epoch'

        self.indexes = np.arange(len(self.list_ids))

        if self.shuffle:

            np.random.shuffle(self.indexes)

            

    def __data_generation(self, list_ids_temp):

        'generate data containing batch_size samples'

        X = np.empty((self.batch_size, self.img_h, self.img_w, 3))

        y = np.empty((self.batch_size, 1))

        

        for idx, id in enumerate(list_ids_temp):

            file_path =  os.path.join(self.image_dir, id)

            image = pydicom.read_file(file_path)

            image = image.pixel_array



            image_resized = cv2.resize(image, (self.img_w, self.img_h))

            

            

            image_resized = np.array(image_resized, dtype=np.float64)

#             image_resized /= 255.0

            

            # standardization of the image

            image_resized -= image_resized.mean()

            image_resized /= image_resized.std()#标准化和归一化看图像还是有区别的，这个要尝试两种方式

            

            X[idx,] = np.expand_dims(image_resized, axis=2)

            y[idx,] = self.labels.get(id)

        

        #image augmentation

        if self.phase == "train":

            aug_seq = iaa.Sequential([

                iaa.Fliplr(0.5),

                iaa.Flipud(0.5),

#                 iaa.Affine(

#                     scale={"x": (0.9, 1.0), "y": (0.1, 0.1)}               

#                 )

            ]

            )

            X = aug_seq(images=X)

        

            

        return X, y
img_h = 256

img_w = 256

batch_size = 16



train_params = {'img_h': img_h,

          'img_w': img_w,

          'image_dir': train_image_dir,

          'batch_size': batch_size,

          'phase':"train",

          'shuffle': True}

val_params = {'img_h': img_h,

          'img_w': img_w,

          'image_dir': train_image_dir,

          'batch_size': batch_size,

          'phase':"validation",

          'shuffle': False}



# Get Generators

training_generator = DataGenerator(train_df, "patientId", "hasMask", **train_params)

valid_generator = DataGenerator(valid_df, "patientId", "hasMask", **val_params)
#check data generator

# x_test, y_test = training_generator.__getitem__(0)

# print(x_test.shape, y_test.shape)

# pylab.imshow(x_test[8])
# try augmentation

# pid2 = train_df["patientId"].values

# def load_batch(ids):

#     ids = ids + ".dcm"

#     img2 = pydicom.read_file(os.path.join(train_image_dir, ids))

#     img2 = img2.pixel_array

#     img2 = np.stack([img2] * 3, -1)

#     return img2



# img2 = np.array([load_batch(id1) for id1 in pid2[:5]])

# print(img2.shape)



# seq2 = iaa.Sequential([

#     iaa.Crop(px=(50, 16), keep_size=False),

#     iaa.Fliplr(0.5),

#     iaa.GaussianBlur(sigma=(0, 20.0)),

#     iaa.CropAndPad(

#             percent=(-0.05, 0.1),

#             pad_cval=(0, 255)

#         ),

#     iaa.AverageBlur(k=(2, 7)),

# ])

# imgseq2 = seq2(images=img2)

# pylab.imshow(imgseq2[0])

# pylab.imshow(img2[0])

# del img2, pid2, seq2, imgseq2
#build model

def build_model(clf_model, drop_rate, lr):

    if clf_model == "Densenet121":

        model_name = DenseNet121

    base_model = model_name(

        include_top=False,

        input_shape=(256, 256, 3),

        weights="imagenet"

    )

    

    model = Sequential()

    model.add(base_model)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(drop_rate))

    model.add(layers.Dense(512, activation="relu"))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(drop_rate))

    model.add(layers.Dense(1, activation="sigmoid"))

    

    #freeze pretrained model

#     for ly in base_model.layers:

#         ly.trainable = False

    

    model.compile(

        loss="binary_crossentropy",

        optimizer=Adam(lr),

        metrics=['accuracy']

    )

    return model
#hyperparameters

lr = 1e-3

drop_rate = 0.5

coarse_gs = 2

clf_model = "Densenet121"
class myHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.losses = []

        self.acc = []



    def on_batch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))

        self.acc.append(logs.get('accuracy'))



myhistory = myHistory()
#define callback

total_steps = train_df.shape[0]/batch_size

checkpoint = ModelCheckpoint(

    model_dir+clf_model+".h5",

    monitor="val_acc",

    verbose=1,

    save_best_only=True,

    save_weights_only=False,

    mode="auto"

)



# model = build_model(clf_model, drop_rate, 0.0001)

# history = model.fit_generator(

#     training_generator,

#     validation_data=valid_generator,

#     steps_per_epoch=total_steps*0.85*0.2,

#     epochs=1,

#     callbacks=[checkpoint, myhistory],

#     use_multiprocessing=True,

#     workers=4

# )



history_list_gs = pd.DataFrame(columns=["round", "learning_rate", "drop_rate", "history", "myhistory"])

for ci in tqdm(range(coarse_gs)):

    lr1 = 10**(np.random.uniform(-4,-2))

    dr1 = np.random.uniform(0.01, 0.99)

    

    model = build_model(clf_model, dr1, lr1)

    history = model.fit_generator(

        training_generator,

        validation_data=valid_generator,

        steps_per_epoch=total_steps*0.85*0.05,

        validation_steps=total_steps*0.15*0.05,

        use_multiprocessing=True,

        epochs=2,

        callbacks=[checkpoint,myhistory]

    )

    print("round:{}, learning_rate:{}, drop_rate:{}".format(ci+1, lr1, dr1))

    print("history:{}".format(history.history))

    history_list_gs.loc[ci, "round"] = ci + 1

    history_list_gs.loc[ci, "learning_rate"] = lr1

    history_list_gs.loc[ci, "drop_rate"] = dr1

    history_list_gs.loc[ci, "history"] = history

    history_list_gs.loc[ci, "myhistory"] = myhistory
# history_df = pd.DataFrame(history.history)

# history_df[['loss', 'val_loss']].plot()

# history_df[['accuracy', 'val_accuracy']].plot()
fig, ax = plt.subplots(3,2, figsize=(16,12))

for row in history_list_gs.iterrows():

    row1 = row[1]

    lr = row1["learning_rate"]

    dr = row1["drop_rate"]

    history = row1["history"].history

    myhistory = row1["myhistory"]

    color = np.random.rand(3,)

    label = "lr:{:.4f}-dr:{:.2f}".format(lr,dr)

    

    

    for (metric, ax1) in zip(history.keys(), ax.flatten()[:4]):

        ax1.set_title(metric)

        ax1.plot(history[metric], color=color, label=label)

        ax1.legend()

    

    ax.flatten()[4].set_title("losses_iteration")

    ax.flatten()[4].plot(myhistory.losses, color=color, label=label)

    ax.flatten()[4].legend()

    ax.flatten()[5].set_title("acc_iteration")

    ax.flatten()[5].plot(myhistory.acc, color=color, label=label)

    ax.flatten()[5].legend()

    