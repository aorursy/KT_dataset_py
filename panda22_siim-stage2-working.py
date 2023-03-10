# Basic imports for the entire Kernel

import numpy as np

import pandas as pd

# import mask function

import sys

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')

from mask_functions import rle2mask, mask2rle

# imports for loading data

import pydicom

from glob import glob

from tqdm import tqdm
# load rles

rles_df = pd.read_csv('../input/siim-dicom-images/train-rle.csv')

# the second column has a space at the start, so manually giving column name

rles_df.columns = ['ImageId', 'EncodedPixels']

rles_df.head(5)
def dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=True):

    """

    Args:

        dicom_data (dicom): chest x-ray data in dicom format.

        file_path (str): file path of the dicom data.

        rles_df (pandas.core.frame.DataFrame): Pandas dataframe of the RLE.

        encoded_pixels (bool): if True we will search for annotation.

        

    Returns:

        dict: contains metadata of relevant fields.

    """

    

    data = {}

        

    # Parse fields with meaningful information

    data['patient_name'] = dicom_data.PatientName

    data['patient_id'] = dicom_data.PatientID

    data['patient_age'] = int(dicom_data.PatientAge)

    data['patient_sex'] = dicom_data.PatientSex

    data['pixel_spacing'] = dicom_data.PixelSpacing

    data['file_path'] = file_path

    data['id'] = dicom_data.SOPInstanceUID

    

    # look for annotation if enabled (train set)

    if encoded_pixels:

        encoded_pixels_list = rles_df[rles_df['ImageId']==dicom_data.SOPInstanceUID]['EncodedPixels'].values

       

        pneumothorax = False

        for encoded_pixels in encoded_pixels_list:

            if encoded_pixels != ' -1':

                pneumothorax = True

        

        # get meaningful information (for train set)

        data['encoded_pixels_list'] = encoded_pixels_list

        data['has_pneumothorax'] = pneumothorax

        data['encoded_pixels_count'] = len(encoded_pixels_list)

        

    return data
# create a list of all the files

train_fns = sorted(glob('../input/siim-dicom-images/siim-original/dicom-images-train/*/*/*.dcm'))

# parse train DICOM dataset

train_metadata_df = pd.DataFrame()

train_metadata_list = []

for file_path in tqdm(train_fns):

    dicom_data = pydicom.dcmread(file_path)

    train_metadata = dicom_to_dict(dicom_data, file_path, rles_df)

    train_metadata_list.append(train_metadata)

train_metadata_df = pd.DataFrame(train_metadata_list)
# create a list of all the files

test_fns = sorted(glob('../input/siim-dicom-images/siim-original/dicom-images-test/*/*/*.dcm'))

# parse test DICOM dataset

test_metadata_df = pd.DataFrame()

test_metadata_list = []

for file_path in tqdm(test_fns):

    dicom_data = pydicom.dcmread(file_path)

    test_metadata = dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=False)

    test_metadata_list.append(test_metadata)

test_metadata_df = pd.DataFrame(test_metadata_list)
import matplotlib.pyplot as plt

from matplotlib import patches as patches
num_img = 4

subplot_count = 0

fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for index, row in train_metadata_df.sample(n=num_img).iterrows():

    dataset = pydicom.dcmread(row['file_path'])

    ax[subplot_count].imshow(dataset.pixel_array, cmap=plt.cm.bone)

    # label the x-ray with information about the patient

    ax[subplot_count].text(0,0,'Age:{}, Sex: {}, Pneumothorax: {}'.format(row['patient_age'],row['patient_sex'],row['has_pneumothorax']),

                           size=26,color='white', backgroundcolor='black')

    subplot_count += 1
def bounding_box(img):

    # return max and min of a mask to draw bounding box

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]



    return rmin, rmax, cmin, cmax



def plot_with_mask_and_bbox(file_path, mask_encoded_list, figsize=(20,10)):

    

    import cv2

    

    """

    Args:

        file_path (str): file path of the dicom data.

        mask_encoded (numpy.ndarray): Pandas dataframe of the RLE.

        

    Returns:

        plots the image with and without mask.

    """

    

    pixel_array = pydicom.dcmread(file_path).pixel_array

    

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    clahe_pixel_array = clahe.apply(pixel_array)

    

    # use the masking function to decode RLE

    mask_decoded_list = [rle2mask(mask_encoded, 1024, 1024).T for mask_encoded in mask_encoded_list]

    

    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(20,10))

    

    # print out the xray

    ax[0].imshow(pixel_array, cmap=plt.cm.bone)

    # print the bounding box

    for mask_decoded in mask_decoded_list:

        # print out the annotated area

        ax[0].imshow(mask_decoded, alpha=0.3, cmap="Reds")

        rmin, rmax, cmin, cmax = bounding_box(mask_decoded)

        bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')

        ax[0].add_patch(bbox)

    ax[0].set_title('With Mask')

    

    # plot image with clahe processing with just bounding box and no mask

    ax[1].imshow(clahe_pixel_array, cmap=plt.cm.bone)

    for mask_decoded in mask_decoded_list:

        rmin, rmax, cmin, cmax = bounding_box(mask_decoded)

        bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')

        ax[1].add_patch(bbox)

    ax[1].set_title('Without Mask - Clahe')

    

    # plot plain xray with just bounding box and no mask

    ax[2].imshow(pixel_array, cmap=plt.cm.bone)

    for mask_decoded in mask_decoded_list:

        rmin, rmax, cmin, cmax = bounding_box(mask_decoded)

        bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')

        ax[2].add_patch(bbox)

    ax[2].set_title('Without Mask')

    plt.show()
# lets take 10 random samples of x-rays with 

train_metadata_sample = train_metadata_df[train_metadata_df['has_pneumothorax']==1].sample(n=10)

# plot ten xrays with and without mask

for index, row in train_metadata_sample.iterrows():

    file_path = row['file_path']

    mask_encoded_list = row['encoded_pixels_list']

    print('image id: ' + row['id'])

    plot_with_mask_and_bbox(file_path, mask_encoded_list)
# plotly offline imports

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

from plotly import tools

from plotly.graph_objs import *

from plotly.graph_objs.layout import Margin, YAxis, XAxis

init_notebook_mode()
# print missing annotation

missing_vals = train_metadata_df[train_metadata_df['encoded_pixels_count']==0]['encoded_pixels_count'].count()

print("Number of x-rays with missing labels: {}".format(missing_vals))
nok_count = train_metadata_df['has_pneumothorax'].sum()

ok_count = len(train_metadata_df) - nok_count

x = ['No Pneumothorax','Pneumothorax']

y = [ok_count, nok_count]

trace0 = Bar(x=x, y=y, name = 'Ok vs Not OK')

nok_encoded_pixels_count = train_metadata_df[train_metadata_df['has_pneumothorax']==1]['encoded_pixels_count'].values

trace1 = Histogram(x=nok_encoded_pixels_count, name='# of annotations')

fig = tools.make_subplots(rows=1, cols=2)

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=900, title='Pneumothorax Instances')

iplot(fig)
pneumo_pat_age = train_metadata_df[train_metadata_df['has_pneumothorax']==1]['patient_age'].values

no_pneumo_pat_age = train_metadata_df[train_metadata_df['has_pneumothorax']==0]['patient_age'].values
pneumothorax = Histogram(x=pneumo_pat_age, name='has pneumothorax')

no_pneumothorax = Histogram(x=no_pneumo_pat_age, name='no pneumothorax')

fig = tools.make_subplots(rows=1, cols=2)

fig.append_trace(pneumothorax, 1, 1)

fig.append_trace(no_pneumothorax, 1, 2)

fig['layout'].update(height=400, width=900, title='Patient Age Histogram')

iplot(fig)
trace1 = Box(x=pneumo_pat_age, name='has pneumothorax')

trace2 = Box(x=no_pneumo_pat_age[no_pneumo_pat_age <= 120], name='no pneumothorax')

data = [trace1, trace2]

iplot(data)
train_male_df = train_metadata_df[train_metadata_df['patient_sex']=='M']

train_female_df = train_metadata_df[train_metadata_df['patient_sex']=='F']
male_ok_count = len(train_male_df[train_male_df['has_pneumothorax']==0])

female_ok_count = len(train_female_df[train_female_df['has_pneumothorax']==0])

male_nok_count = len(train_male_df[train_male_df['has_pneumothorax']==1])

female_nok_count = len(train_female_df[train_female_df['has_pneumothorax']==1])
ok = Bar(x=['male', 'female'], y=[male_ok_count, female_ok_count], name='no pneumothorax')

nok = Bar(x=['male', 'female'], y=[male_nok_count, female_nok_count], name='has pneumothorax')



data = [ok, nok]

layout = Layout(barmode='stack', height=400)



fig = Figure(data=data, layout=layout)

iplot(fig, filename='stacked-bar')
m_pneumo_labels = ['no pneumothorax','has pneumothorax']

f_pneumo_labels = ['no pneumothorax','has pneumothorax']

m_pneumo_values = [male_ok_count, male_nok_count]

f_pneumo_values = [female_ok_count, female_nok_count]

colors = ['#FEBFB3', '#E1396C']
def get_affected_area(encoded_pixels_list, pixel_spacing):

    

    # take the encoded mask, decode, and get the sum of nonzero elements

    pixel_sum = 0

    

    for encoded_mask in encoded_pixels_list:

        mask_decoded = rle2mask(encoded_mask, 1024, 1024).T

        pixel_sum += np.count_nonzero(mask_decoded)

        

    area_per_pixel = pixel_spacing[0] * pixel_spacing[1]

    

    return pixel_sum * area_per_pixel
# create a subset of dataframe for pneumothorax patients

pneumothorax_df = train_metadata_df[train_metadata_df['has_pneumothorax']==1].copy()

# get sum of non zero elements in mask

pneumothorax_df['pneumothorax_area'] = pneumothorax_df.apply(lambda row: get_affected_area(row['encoded_pixels_list'], row['pixel_spacing']),axis=1)
pneumothorax_df_m = pneumothorax_df[pneumothorax_df['patient_sex']=='M']

pneumothorax_df_f = pneumothorax_df[pneumothorax_df['patient_sex']=='F']

pneumo_size_m = pneumothorax_df_m['pneumothorax_area'].values

pneumo_size_f = pneumothorax_df_f['pneumothorax_area'].values
pneumo_size_m_trace = Box(x = pneumo_size_m, name='M')

pneumo_size_f_trace = Box(x = pneumo_size_f, name='F')

layout = Layout(title='Pneumothorax Affected Area for Male and Female Population', 

               xaxis = XAxis(title='Area (in sq mm)'))



data = [pneumo_size_m_trace, pneumo_size_f_trace]

fig = Figure(data=data, layout=layout)

iplot(fig)
pneumo_size_m_trace = Scatter(x=pneumothorax_df_m['patient_age'].values, 

                              y=pneumothorax_df_m['pneumothorax_area'].values, 

                              mode='markers', name='Male')



pneumo_size_f_trace = Scatter(x=pneumothorax_df_f['patient_age'].values, 

                              y=pneumothorax_df_f['pneumothorax_area'].values, 

                              mode='markers', name='Female')



layout = Layout(title='Pneumothorax Affected Area vs Age for Male and Female Population', 

                yaxis=YAxis(title='Area (in sq mm)'), xaxis=XAxis(title='Age'))



data = [pneumo_size_m_trace, pneumo_size_f_trace]

fig = Figure(data=data, layout=layout)

iplot(fig)
size_m = pneumothorax_df_m['pneumothorax_area'].values

size_ref_m = 2.*max(size_m)/(40.**2)

size_f = pneumothorax_df_f['pneumothorax_area'].values

size_ref_f = 2.*max(size_f)/(40.**2)



pneumo_size_m_trace = Scatter(x=pneumothorax_df_m['patient_age'].values, 

                              y=pneumothorax_df_m['encoded_pixels_count'].values,

                              marker=dict(size= size_m, sizemode='area', sizeref=size_ref_m, sizemin=4), 

                              mode='markers', name='Male')



pneumo_size_f_trace = Scatter(x=pneumothorax_df_f['patient_age'].values, 

                              y=pneumothorax_df_f['encoded_pixels_count'].values,

                              marker=dict(size=size_f, sizemode='area', sizeref=size_ref_f, sizemin=4), 

                              mode='markers', name='Female')



layout = Layout(title='Pneumothorax Affected Area vs Age for Male and Female Population', yaxis=YAxis(title='Area (in sq mm)'), xaxis=XAxis(title='Age'))



data = [pneumo_size_m_trace, pneumo_size_f_trace]

fig = Figure(data=data, layout=layout)

iplot(fig)
def age_categories(age):

    # take age as input and return age category

    if age <= 14:

        return 'Child'

    if age >=15 and age <= 24:

        return 'Youth'

    if age >=25 and age <=64:

        return 'Adult'

    if age >= 65:

        return 'Senior'



# get age categories

pneumothorax_df['age_category'] = pneumothorax_df['patient_age'].apply(age_categories)
# here we loop over the different age categories and M and F genders to create a subplot

data = []

fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Child','Youth','Adult','Senior'))

subplot_positions = [(1,1),(1,2),(2,1),(2,2)]



# loop over each age category

for i, cat in enumerate(['Child','Youth','Adult','Senior']):

    # and gender

    for gender in ['M','F']:

        # get affected area for given age group and gender

        values = pneumothorax_df[(pneumothorax_df['patient_sex']==gender) 

                        & (pneumothorax_df['age_category']==cat)]['pneumothorax_area'].values

        # add to the respective trace

        trace = Box(x=values, name=gender)

        # add to figure

        fig.append_trace(trace, subplot_positions[i][0], subplot_positions[i][1])
fig['layout'].update(height=600, width=900, title='Pneumothorax Size in Different Age Categories', showlegend=False)

iplot(fig)
# defining configuration parameters

img_size = 256 # image resize size

batch_size = 32 # batch size for training unet

k_size = 3 # kernel size 3x3
import tensorflow as tf

from sklearn.model_selection import train_test_split

import cv2
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, file_path_list, labels, batch_size=32, 

                 img_size=256, channels=1, shuffle=True):

        self.file_path_list = file_path_list

        self.labels = labels

        self.batch_size = batch_size

        self.img_size = img_size

        self.channels = channels

        self.shuffle = shuffle

        self.on_epoch_end()

    

    def __len__(self):

        'denotes the number of batches per epoch'

        return int(np.floor(len(self.file_path_list)) / self.batch_size)

    

    def __getitem__(self, index):

        'generate one batch of data'

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # get list of IDs

        file_path_list_temp = [self.file_path_list[k] for k in indexes]

        # generate data

        X, y = self.__data_generation(file_path_list_temp)

        # return data 

        return X, y

    

    def on_epoch_end(self):

        'update ended after each epoch'

        self.indexes = np.arange(len(self.file_path_list))

        if self.shuffle:

            np.random.shuffle(self.indexes)

            

    def __data_generation(self, file_path_list_temp):

        'generate data containing batch_size samples'

        X = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))

        y = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))

        

        for idx, file_path in enumerate(file_path_list_temp):

            

            id = file_path.split('/')[-1][:-4]

            rle = self.labels.get(id)

            image = pydicom.read_file(file_path).pixel_array

            image_resized = cv2.resize(image, (self.img_size, self.img_size))

            image_resized = np.array(image_resized, dtype=np.float64)

            

            X[idx,] = np.expand_dims(image_resized, axis=2)

            

            # if there is no mask create empty mask

            # notice we are starting of with 1024 because we need to use the rle2mask function

            if rle is None:

                mask = np.zeros((1024, 1024))

            else:

                if len(rle) == 1:

                    #y[i,] = np.expand_dims(rle2mask(rle[0], 1024, 1024).T, axis=2)

                    mask = rle2mask(rle[0], 1024, 1024).T

                else: 

                    mask = np.zeros((1024, 1024))

                    for x in rle:

                        mask =  mask + rle2mask(rle[0], 1024, 1024).T

                        

            mask_resized = cv2.resize(mask, (self.img_size, self.img_size))

            y[idx,] = np.expand_dims(mask_resized, axis=2)

            

        # normalize 

        X = X / 255

        y = y / 255

            

        return X, y
masks = {}

for index, row in train_metadata_df[train_metadata_df['has_pneumothorax']==1].iterrows():

    masks[row['id']] = list(row['encoded_pixels_list'])
# split the training data into train and validation set (stratified)

X_train, X_val, y_train, y_val = train_test_split(train_metadata_df.index, train_metadata_df['has_pneumothorax'].values, test_size=0.2, random_state=42)

X_train, X_val = train_metadata_df.loc[X_train]['file_path'].values, train_metadata_df.loc[X_val]['file_path'].values
params = {'img_size': img_size,

          'batch_size': batch_size,

          'channels': 1,

          'shuffle': True}



# Generators

training_generator = DataGenerator(X_train, masks, **params)

validation_generator = DataGenerator(X_val, masks, **params)
x, y = training_generator.__getitem__(0)

print(x.shape, y.shape)
fig = plt.figure()

fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)

ax.imshow(x[0].reshape(256,256), cmap=plt.cm.bone)

ax = fig.add_subplot(1, 2, 2)

ax.imshow(np.reshape(y[0], (img_size, img_size)), cmap="gray")
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, concatenate
merge_axis = -1

data = Input((img_size, img_size, 1))

conv1 = Conv2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(data)

conv1 = Conv2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(conv1)

pool1 = MaxPool2D(pool_size=(2, 2))(conv1)



conv2 = Conv2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool1)

conv2 = Conv2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv2)

pool2 = MaxPool2D(pool_size=(2, 2))(conv2)



conv3 = Conv2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool2)

conv3 = Conv2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv3)

pool3 = MaxPool2D(pool_size=(2, 2))(conv3)



conv4 = Conv2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(pool3)

conv4 = Conv2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv4)

pool4 = MaxPool2D(pool_size=(2, 2))(conv4)



conv5 = Conv2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(pool4)



up1 = UpSampling2D(size=(2, 2))(conv5)

conv6 = Conv2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up1)

conv6 = Conv2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv6)

merged1 = concatenate([conv4, conv6], axis=merge_axis)

conv6 = Conv2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged1)



up2 = UpSampling2D(size=(2, 2))(conv6)

conv7 = Conv2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up2)

conv7 = Conv2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv7)

merged2 = concatenate([conv3, conv7], axis=merge_axis)

conv7 = Conv2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged2)



up3 = UpSampling2D(size=(2, 2))(conv7)

conv8 = Conv2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(up3)

conv8 = Conv2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv8)

merged3 = concatenate([conv2, conv8], axis=merge_axis)

conv8 = Conv2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(merged3)



up4 = UpSampling2D(size=(2, 2))(conv8)

conv9 = Conv2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(up4)

conv9 = Conv2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv9)

merged4 = concatenate([conv1, conv9], axis=merge_axis)

conv9 = Conv2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(merged4)



conv10 = Conv2D(filters=1, kernel_size=k_size, padding='same', activation='sigmoid')(conv9)



output = conv10
smooth = 1.



def dice_coef(y_true, y_pred):

    y_true_f = tf.keras.layers.Flatten()(y_true)

    y_pred_f = tf.keras.layers.Flatten()(y_pred)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)





def dice_coef_loss(y_true, y_pred):

    return 1.0 - dice_coef(y_true, y_pred)
model = Model(data, output)

adam = tf.keras.optimizers.Adam(lr = 0.01, epsilon = 0.1)

model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])

model.summary()
history = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=1, verbose=1)
model.save("model.h5")
def plot_train(img, mask, pred):

    

    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,5))

    

    ax[0].imshow(img, cmap=plt.cm.bone)

    ax[0].set_title('Chest X-Ray')

    

    ax[1].imshow(mask, cmap=plt.cm.bone)

    ax[1].set_title('Mask')

    

    ax[2].imshow(pred, cmap=plt.cm.bone)

    ax[2].set_title('Pred Mask')

    

    plt.show()
for i in range(0,3):

    x, y = validation_generator.__getitem__(i)

    predictions = model.predict(x)

    for idx, val in enumerate(x):

        if y[idx].sum() > 0: 

            print(y[idx].sum(), predictions[idx].sum(), y[idx].sum() - predictions[idx].sum())

            img = np.reshape(x[idx]* 255, (img_size, img_size))

            mask = np.reshape(y[idx]* 255, (img_size, img_size))

            pred = np.reshape(predictions[idx], (img_size, img_size))

            pred = pred > 0.5

            pred = pred * 255



            plot_train(img, mask, pred)
def get_test_tensor(file_path, batch_size, img_size, channels):

    

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.empty((batch_size, img_size, img_size, channels))



        # Store sample

        pixel_array = pydicom.read_file(file_path).pixel_array

        image_resized = cv2.resize(pixel_array, (img_size, img_size))

        image_resized = np.array(image_resized, dtype=np.float64)

        image_resized -= image_resized.mean()

        image_resized /= image_resized.std()

        X[0,] = np.expand_dims(image_resized, axis=2)



        return X
output = []



for i, row in test_metadata_df.iterrows():



    test_img = get_test_tensor(test_metadata_df['file_path'][i],1,256,1)

    

    pred_mask = model.predict(test_img).reshape((img_size,img_size))

    prediction = {}

    prediction['ImageId'] = str(test_metadata_df['id'][i])

    pred_mask = (pred_mask > .5).astype(int)

    

    

    if pred_mask.sum() < 1:

        prediction['EncodedPixels'] =  -1

    else:

        prediction['EncodedPixels'] = mask2rle(pred_mask * 255, img_size, img_size)

    output.append(prediction)
output_df = pd.DataFrame(output)

output_df = output_df[['ImageId','EncodedPixels']]

output_df.head()
output_df.to_csv('./output.csv', index=False)


    

    