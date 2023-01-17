import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import seaborn as sns
import cv2
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from math import ceil, floor, log
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
!ls ../input
df_train = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv')
df_train.head(10)
df_train.shape
df_train.Label.isnull().sum()
df_train.ID.isnull().sum()
# Images Example
train_images_dir = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/'
train_images = [f for f in listdir(train_images_dir) if isfile(join(train_images_dir, f))]
test_images_dir = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_test/'
test_images = [f for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]
print('5 Training images', train_images[:5]) # Print the first 5
print('Total File sizes')
for f in os.listdir('../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/' + f) / 1000000, 2)) + 'MB')
fig=plt.figure(figsize=(15, 10))
columns = 5; rows = 4
for i in range(1, columns*rows +1):
    ds = pydicom.dcmread(train_images_dir + train_images[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot
print(ds) # this is file type of image
im = ds.pixel_array
print(type(im))
print(im.dtype)
print(im.shape)
sns.countplot(df_train.Label)
print('Number of train images:', len(train_images))
print('Number of test images:', len(test_images))
df_train.Label.value_counts()
df_train['Sub_type'] = df_train['ID'].str.split("_", n = 3, expand = True)[2]
df_train['PatientID'] = df_train['ID'].str.split("_", n = 3, expand = True)[1]
df_train.head()
gbSub = df_train.groupby('Sub_type').sum()
gbSub
sns.barplot(y=gbSub.index, x=gbSub.Label, palette="deep")
fig=plt.figure(figsize=(10, 8))

sns.countplot(x="Sub_type", hue="Label", data=df_train)

plt.title("Total Images by Subtype")
def window_image(img, window_center,window_width, intercept, slope):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img 
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
train_images_dir

def view_images(images, title = '', aug = None):
    width = 5
    height = 2
    fig, axs = plt.subplots(height, width, figsize=(15,5))
    
    for im in range(0, height * width):
        
        data = pydicom.read_file(os.path.join(train_images_dir,'ID_'+images[im]+ '.dcm'))
        image = data.pixel_array
        window_center , window_width, intercept, slope = get_windowing(data)
        image_windowed = window_image(image, window_center, window_width, intercept, slope)


        i = im // width
        j = im % width
        axs[i,j].imshow(image_windowed, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
        
    plt.suptitle(title)
    plt.show()
view_images(df_train[(df_train['Sub_type'] == 'epidural') & (df_train['Label'] == 1)][:10].PatientID.values, title = 'Images of hemorrhage epidural')
view_images(df_train[(df_train['Sub_type'] == 'subdural') & (df_train['Label'] == 1)][:20].PatientID.values, title = 'Images of hemorrhage subdural')
view_images(df_train[(df_train['Sub_type'] == 'intraventricular') & (df_train['Label'] == 1)][:20].PatientID.values, title = 'Images of hemorrhage intraventricular')
view_images(df_train[(df_train['Sub_type'] == 'subarachnoid') & (df_train['Label'] == 1)][:20].PatientID.values, title = 'Images of hemorrhage subarachnoid')
view_images(df_train[(df_train['Sub_type'] == 'intraparenchymal') & (df_train['Label'] == 1)][:20].PatientID.values, title = 'Images of hemorrhage intraparenchymal')
def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def window_image_dcm(dcm, window_center, window_width):
    
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img

def bsb_window(dcm):
    brain_img = window_image_dcm(dcm, 40, 80)
    subdural_img = window_image_dcm(dcm, 80, 200)
    soft_img = window_image_dcm(dcm, 40, 380)
    
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    return bsb_img

dicom = pydicom.dcmread(train_images_dir + 'ID_5c8b5d701' + '.dcm')
plt.imshow(bsb_window(dicom), cmap=plt.cm.bone);
def _read(path, desired_size):
    
    dcm = pydicom.dcmread(path)
    
    try:
        img = bsb_window(dcm)
    except:
        img = np.zeros(desired_size)
    
    
    img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)
    
    return img

# Another sanity check 
plt.imshow(
    _read(train_images_dir+'ID_5c8b5d701'+'.dcm', (128,128)), cmap=plt.cm.bone
);
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, list_IDs, labels=None, batch_size=1, img_size=(512, 512, 1), 
                 img_dir=train_images_dir, *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        
        if self.labels is not None:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X
        
    def on_epoch_end(self):
        
        
        if self.labels is not None: # for training phase we undersample and shuffle
            # keep probability of any=0 and any=1
            keep_prob = self.labels.iloc[:, 0].map({0: 0.35, 1: 0.5})
            keep = (keep_prob > np.random.rand(len(keep_prob)))
            self.indices = np.arange(len(self.list_IDs))[keep]
            np.random.shuffle(self.indices)
        else:
            self.indices = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size))
        
        if self.labels is not None: # training phase
            Y = np.empty((self.batch_size, 6), dtype=np.float32)
        
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir+ID+".dcm", self.img_size)
                Y[i,] = self.labels.loc[ID].values
        
            return X, Y
        
        else: # test phase
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir+ID+".dcm", self.img_size)
            
            return X
class PredictionCheckpoint(tf.keras.callbacks.Callback):
    
    def __init__(self, test_df, valid_df, 
                 test_images_dir=test_images_dir, 
                 valid_images_dir=train_images_dir, 
                 batch_size=32, input_size=(224, 224, 3)):
        
        self.test_df = test_df
        self.valid_df = valid_df
        self.test_images_dir = test_images_dir
        self.valid_images_dir = valid_images_dir
        self.batch_size = batch_size
        self.input_size = input_size
        
    def on_train_begin(self, logs={}):
        self.test_predictions = []
        self.valid_predictions = []
        
    def on_epoch_end(self,batch, logs={}):
        self.test_predictions.append(
            self.model.predict_generator(
                DataGenerator(self.test_df.index, None, self.batch_size, self.input_size, self.test_images_dir), verbose=2)[:len(self.test_df)])

        # Commented out to save time
#         self.valid_predictions.append(
#             self.model.predict_generator(
#                 DataGenerator(self.valid_df.index, None, self.batch_size, self.input_size, self.valid_images_dir), verbose=2)[:len(self.valid_df)])
        
#         print("validation loss: %.4f" %
#               weighted_log_loss_metric(self.valid_df.values, 
#                                    np.average(self.valid_predictions, axis=0, 
#                                               weights=[2**i for i in range(len(self.valid_predictions))])))
        
        # here you could also save the predictions with np.save()
        
class MyDeepModel:
    
    def __init__(self, engine, input_dims, batch_size=5, num_epochs=5, learning_rate=1e-3, 
                 decay_rate=1.0, decay_steps=1, weights="imagenet", verbose=1):
        
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.weights = weights
        self.verbose = verbose
        self._build()

    def _build(self):
        
        engine = self.engine(include_top=False, weights=self.weights, input_shape=self.input_dims,
                             backend = tf.keras.backend, layers = tf.keras.layers,
                             models = tf.keras.models, utils = tf.keras.utils)
        
        x = GlobalAveragePooling2D(name='avg_pool')(engine.output)
        out = Dense(6, activation="sigmoid", name='dense_output')(x)

        self.model = Model(inputs=engine.input, outputs=out)

        self.model.compile(loss=binary_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    

    def fit_and_predict(self, train_df, valid_df, test_df):
        
        # callbacks
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)
        scheduler = LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))
        
        self.model.fit_generator(
            DataGenerator(
                train_df.index, 
                train_df, 
                self.batch_size, 
                self.input_dims, 
                train_images_dir
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
            use_multiprocessing=True,
            workers=4,
            callbacks=[pred_history, scheduler]
        )
        
        return pred_history
    
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)
def read_testset(filename="../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    
    return df

def read_trainset(filename="../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    
    duplicates_to_remove = [ 56346, 56347, 56348, 56349, 56350, 56351, 1171830, 1171831, 1171832, 
                            1171833, 1171834, 1171835, 3705312, 3705313, 3705314, 3705315, 3705316, 
                            3705317, 3842478, 3842479, 3842480, 3842481, 3842482, 3842483]
    
    df = df.drop(index=duplicates_to_remove)
    df = df.reset_index(drop=True)
    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    
    return df

    
test_df = read_testset()
train_df = read_trainset()
train_df.head()
test_df.head()
# train set (90%) and validation set (10%)
ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42).split(train_df.index)

# lets go for the first fold only
train_idx, valid_idx = next(ss)

# obtain model
model = MyDeepModel(engine=InceptionV3, input_dims=(256, 256, 3), batch_size=32, learning_rate=5e-3,
                    num_epochs=5, decay_rate=0.8, decay_steps=1, weights="imagenet", verbose=1)

# obtain test + validation predictions (history.test_predictions, history.valid_predictions)
# history = model.fit_and_predict(train_df.iloc[train_idx], train_df.iloc[valid_idx], test_df)
# test_df.iloc[:, :] = np.average(history.test_predictions, axis=0, weights=[0, 1, 2, 4, 6]) # let's do a weighted average for epochs (>1)

# test_df = test_df.stack().reset_index()

# test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])

# test_df = test_df.drop(["Image", "Diagnosis"], axis=1)

# test_df.to_csv('submission.csv', index=False)