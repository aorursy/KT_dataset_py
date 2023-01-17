# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydicom

import seaborn as sns

import imageio

from IPython import display

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/rsna-str-pulmonary-embolism-detection/train.csv')

df_test = pd.read_csv('/kaggle/input/rsna-str-pulmonary-embolism-detection/test.csv')

df_train.head()
df_train.shape, df_test.shape
df_train.head().T
df_train.describe()
df_train.info()
df_test.head()
df_test.info()
print("Number of unique Study instances are", df_train['StudyInstanceUID'].nunique())

print("Number of unique Series instances are", df_train['SeriesInstanceUID'].nunique())
print('Null values in train data:',df_train.isnull().sum().sum())

print('Null values in test data:',df_test.isnull().sum().sum())
import matplotlib.pyplot as plt

import pydicom

from pydicom.data import get_testdata_files
dataset = pydicom.dcmread('/kaggle/input/rsna-str-pulmonary-embolism-detection/train/b4548bee81e8/ac1aea5d7662/cc96a7a2e72c.dcm')
dataset
dataset.pixel_array
dataset.PixelData[0:40]
# Normal mode:

print("Modality.........:", dataset.Modality)



if 'PixelData' in dataset:

    rows = int(dataset.Rows)

    cols = int(dataset.Columns)

    print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

        rows=rows, cols=cols, size=len(dataset.PixelData)))

    if 'PixelSpacing' in dataset:

        print("Pixel spacing....:", dataset.PixelSpacing)



# plot the image using matplotlib

plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

plt.show()
from os import listdir, mkdir
basepath = "../input/rsna-str-pulmonary-embolism-detection/"

listdir(basepath)


dcmfile_path = basepath + "train/" + df_train.StudyInstanceUID.values[0] +'/'+ df_train.SeriesInstanceUID.values[0]

files = listdir(dcmfile_path)

scans = [pydicom.dcmread(dcmfile_path + "/" + str(file)) for file in files]
len(scans)
fig=plt.figure(figsize=(8, 8))

columns = 4

rows = 4

for i in range(1, columns*rows +1):

    img = scans[i].pixel_array 

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

plt.show()
plt.figure(figsize=(12,6))

for n in range(10):

    image = scans[n].pixel_array.flatten()

    sns.distplot(image);

plt.title("HU unit distributions for 10 examples");
def load_slice(path):

    slices = [pydicom.read_file(path + '/' + s) for s in listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def transform_to_hu(slices):

    images = np.stack([file.pixel_array for file in slices])

    images = images.astype(np.int16)



    # convert ouside pixel-values to air:

    # I'm using <= -1000 to be sure that other defaults are captured as well

    images[images <= -1000] = 0

    

    # convert to HU

    for n in range(len(slices)):

        

        intercept = slices[n].RescaleIntercept

        slope = slices[n].RescaleSlope

        

        if slope != 1:

            images[n] = slope * images[n].astype(np.float64)

            images[n] = images[n].astype(np.int16)

            

        images[n] += np.int16(intercept)

    

    return np.array(images, dtype=np.int16)



def resample(image, scan, new_spacing=[1,1,1]):

    spacing = np.array([float(scans_0[0].SliceThickness), 

                        float(scans_0[0].PixelSpacing[0]), 

                        float(scans_0[0].PixelSpacing[0])])





    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    

    return image, new_spacing



def make_mesh(image, threshold=-300, step_size=1):

    p = image.transpose(2,1,0)

    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)

    return verts, faces





def plt_3d(verts, faces):

    print("Drawing")

    x,y,z = zip(*verts) 

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')



    # Fancy indexing: `verts[faces]` to generate a collection of triangles

    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)

    face_color = [1, 1, 0.9]

    mesh.set_facecolor(face_color)

    ax.add_collection3d(mesh)



    ax.set_xlim(0, max(x))

    ax.set_ylim(0, max(y))

    ax.set_zlim(0, max(z))

#     ax.set_axis_bgcolor((0.7, 0.7, 0.7))

    ax.set_facecolor((0.7,0.7,0.7))

    plt.show()
first_patient = load_slice('../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf')

first_patient_pixels = transform_to_hu(first_patient)

imageio.mimsave("/tmp/gif.gif", first_patient_pixels, duration=0.1)

display.Image(filename="/tmp/gif.gif", format='png')
df_train['filename'] = df_train[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']].apply(

    lambda x: '/'.join(x.astype(str)),

    axis=1

)
df_train['filename'].head()
from keras.utils import Sequence

from skimage.transform import resize

import math

class generator(Sequence):

    

    def __init__(self,df,images_path,batch_size=32, image_size=256, shuffle=True):

        self.df=df

        self.images_path = images_path

        self.batch_size = batch_size

        self.image_size = image_size

        self.shuffle = shuffle

        self.nb_iteration = math.ceil((self.df.shape[0])/self.batch_size)

        self.on_epoch_end()

        

    def load_img(self, filename):

        # load dicom file as numpy array

        img = pydicom.dcmread(filename).pixel_array

        img= resize(img,(self.image_size, self.image_size))

        img = img.reshape((self.image_size, self.image_size, 1))

        np.stack([img, img, img], axis=2).reshape((self.image_size, self.image_size, 3))

        return img

        

    def __getitem__(self, index):

        # select batch

        indicies = list(range(index*self.batch_size, min((index*self.batch_size)+self.batch_size ,(self.df.shape[0]))))

        

        images = []

        for img_path in self.df['filename'].iloc[indicies].tolist():

            img_path = img_path+".dcm"

            img = self.load_img(os.path.join(self.images_path,img_path))

            images.append(img)

        y = self.df[['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',

                     'leftsided_pe', 'chronic_pe', 'rightsided_pe',

                     'acute_and_chronic_pe', 'central_pe', 'indeterminate']].iloc[indicies].values

        return np.array(images), np.array(y)

         

    def on_epoch_end(self):

        if self.shuffle:

            self.df=self.df.sample(frac=1)

        

    def __len__(self):

        return self.nb_iteration
images_path="../input/rsna-str-pulmonary-embolism-detection/train/"

df_train= df_train.iloc[:20000]

df_val= df_train.iloc[20000:25000]

train_dataloader =  generator(df_train,images_path)

val_dataloader =  generator(df_val,images_path)
x,y = next(enumerate(train_dataloader))[1]
x.shape
inputs = Input((256, 256, 1))

Densenet_model = tf.keras.applications.DenseNet121(

            include_top=False,

            weights=None,

            input_shape=(256,256,1))



outputs = Densenet_model(inputs)

outputs = GlobalAveragePooling2D()(outputs)

outputs = Dropout(0.25)(outputs)

outputs = Dense(1024, activation='relu')(outputs)

outputs = Dropout(0.25)(outputs)

outputs = Dense(256, activation='relu')(outputs)

outputs = Dropout(0.25)(outputs)

outputs = Dense(64, activation='relu')(outputs)

nepe = Dense(1, activation='sigmoid', name='negative_exam_for_pe')(outputs)

rlrg1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_gte_1')(outputs)

rlrl1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_lt_1')(outputs) 

lspe = Dense(1, activation='sigmoid', name='leftsided_pe')(outputs)

cpe = Dense(1, activation='sigmoid', name='chronic_pe')(outputs)

rspe = Dense(1, activation='sigmoid', name='rightsided_pe')(outputs)

aacpe = Dense(1, activation='sigmoid', name='acute_and_chronic_pe')(outputs)

cnpe = Dense(1, activation='sigmoid', name='central_pe')(outputs)

indt = Dense(1, activation='sigmoid', name='indeterminate')(outputs)



model = Model(inputs=inputs, outputs={'negative_exam_for_pe':nepe,

                                      'rv_lv_ratio_gte_1':rlrg1,

                                      'rv_lv_ratio_lt_1':rlrl1,

                                      'leftsided_pe':lspe,

                                      'chronic_pe':cpe,

                                      'rightsided_pe':rspe,

                                      'acute_and_chronic_pe':aacpe,

                                      'central_pe':cnpe,

                                      'indeterminate':indt})





model.compile(optimizer=Adam(lr=1e-3),

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.summary()
from tensorflow.keras.utils import plot_model

plot_model(model)
hist = model.fit_generator( train_dataloader,validation_data = val_dataloader,epochs = 5)