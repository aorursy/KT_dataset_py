# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob, pylab, pandas as pd
import pydicom, numpy as np
import matplotlib.pyplot as plt
!ls /kaggle/input/rsna-pneumonia-detection-challenge
import sys
print(sys.version)
pr_root='/kaggle/input/rsna-pneumonia-detection-challenge'

detail_info_df = pd.read_csv(pr_root+'/stage_2_detailed_class_info.csv')

detail_info_df.info()
detail_info_df.info()
detail_info_df.head(3)


detail_info_df['class'].unique()
detail_info_df.isnull().apply(pd.value_counts)
# Read stage_2_train_labels.csv: Sample file is in proper format.
train_df = pd.read_csv(pr_root+'/stage_2_train_labels.csv')
train_df.info()
print("\nShape of train labels dataset:", train_df.shape)
train_df.head(3)
train_df.nunique(dropna=True)
train_df.isnull().apply(pd.value_counts)
# Analyse patientIds and their corresponding bounding boxes. 
bounding_box_by_pat_grp = train_df.groupby(['patientId']).size().to_frame('total_boxes').reset_index()
bounding_box_by_pat_grp.groupby(['total_boxes']).size().to_frame('NumberOfPatients').reset_index()
# Concate the train_lbl_df and cls_info_df and create a single dataframe for processing, as each patientId is associated with unique class label
#detailed_trained_lbl_df = pd.merge(train_lbl_df, detailed_cls_info_df, how='inner')
combined_df = pd.concat([train_df, detail_info_df['class']],  join='inner', verify_integrity=True, axis = 1)
unique_values, count = np.unique(combined_df['Target'], return_counts=True)
print("unique_values: ",unique_values)
lbls = {1: 'Pneumonia symptoms present', 0: 'Normal'}

# Visualize the propertion of Pneumonia vs normal patients
plt.pie(count, labels = ['Pneumonia symptoms present', 'Normal'], autopct='%1.1f%%', startangle=90)
plt.tight_layout()
unique_values, count = np.unique(combined_df['Target'], return_counts=True)
print("unique_values: ",unique_values)
lbls = {1: 'Pneumonia symptoms present', 0: 'Normal'}

# Visualize the propertion of Pneumonia vs normal patients
plt.pie(count, labels = ['Pneumonia symptoms present', 'Normal'], autopct='%1.1f%%', startangle=90)
plt.tight_layout()
combined_df.head(3)
#add dicom file column to dataframe
combined_df['dicom'] = combined_df.apply(lambda x: (pr_root+'/stage_2_train_images/%s.dcm' % x['patientId']),axis=1)
import glob, pandas as pd
import matplotlib.pyplot as plt
import pydicom, numpy as np

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
                'dicom': pr_root+'/stage_2_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'class': row['class'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed


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

    plt.imshow(im, cmap=plt.cm.gist_gray)
    plt.axis('off')

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
combined_df.head()

parsed = parse_data(combined_df)

patientId = combined_df['patientId'][4]
print('Just a checking that everything is working fine...')
print(parsed[patientId])



draw(parsed[patientId])
unique_values, count = np.unique(combined_df['class'], return_counts=True)

# Visualize the distribution of class info
plt.pie(count, labels = ['Lung Opacity', 'No Lung Opacity / Not Normal', 'Normal'], autopct='%1.1f%%', startangle=90)
plt.tight_layout()
import seaborn as sns
sns.countplot(combined_df['class'],  hue=combined_df['Target'], palette='Greens')
# Analyse patientIds and their corresponding bounding boxes. 
bounding_box_by_pat_grp = combined_df.groupby(['patientId']).size().to_frame('total_boxes').reset_index()
bounding_box_by_pat_grp.groupby(['total_boxes']).size().to_frame('NumberOfPatients').reset_index()
import skimage
print (skimage.__version__)
import numpy as np # linear algebra
import tensorflow as tf # for tensorflow based registration
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.util import montage
import os
from cv2 import imread, createCLAHE # read and equalize images
import cv2
from glob import glob
%matplotlib inline
import matplotlib.pyplot as plt
xray_paths = glob(os.path.join('..', 'input', 'pulmonary-chest-xray-abnormalities',
                              'Montgomery', 'MontgomerySet', '*', '*.png'))
xray_images = [(c_path, 
               [os.path.join('/'.join(c_path.split('/')[:-2]),'ManualMask','leftMask', os.path.basename(c_path)),
               os.path.join('/'.join(c_path.split('/')[:-2]),'ManualMask','rightMask', os.path.basename(c_path))]
              ) for c_path in xray_paths]
print('xray Images', len(xray_paths))
print(xray_images[0])
!ls -l ../input/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png/*
!ls -l /kaggle/input/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/ManualMask/leftMask |wc -l
from skimage.io import imread as imread_raw
from skimage.transform import resize
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore', category=UserWarning, module='skimage') # skimage is really annoying
OUT_DIM = (512, 512)
def imread(in_path, apply_clahe = False):
    img_data = imread_raw(in_path)
    n_img = (255*resize(img_data, OUT_DIM, mode = 'constant')).clip(0,255).astype(np.uint8)
    if apply_clahe:
        clahe_tool = createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        n_img = clahe_tool.apply(n_img)
    return np.expand_dims(n_img, -1)
#Create Numpy Array iof Imaages and Masks
img_vol, seg_vol = [], []
for img_path, s_paths in tqdm(xray_images):
    img_vol += [imread(img_path)]    
    seg_vol += [np.max(np.stack([imread(s_path, apply_clahe = False) for s_path in s_paths],0),0)]
img_vol = np.stack(img_vol,0)
seg_vol = np.stack(seg_vol,0)
print('Images', img_vol.shape, 'Segmentations', seg_vol.shape)
np.random.seed(2018)
t_img, m_img = img_vol[0], seg_vol[0]

fig, (ax_img, ax_mask) = plt.subplots(1,2, figsize = (12, 6))
ax_img.imshow(np.clip(255*t_img, 0, 255).astype(np.uint8) if t_img.shape[2]==3 else t_img[:,:,0],
              interpolation = 'none', cmap = 'bone')
ax_mask.imshow(m_img[:,:,0], cmap = 'bone')
from keras.layers import Conv2D, Activation, Input, UpSampling2D, concatenate, BatchNormalization
from keras.layers import LeakyReLU
from keras.initializers import RandomNormal
def c2(x_in, nf, strides=1):
    x_out = Conv2D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out
def unet_enc(vol_size, enc_nf, pre_filter = 8):
    src = Input(shape=vol_size + (1,), name = 'EncoderInput')
    # down-sample path.
    x_in = BatchNormalization(name = 'NormalizeInput')(src)
    x_in = c2(x_in, pre_filter, 1)
    x0 = c2(x_in, enc_nf[0], 2)  
    x1 = c2(x0, enc_nf[1], 2)  
    x2 = c2(x1, enc_nf[2], 2)  
    x3 = c2(x2, enc_nf[3], 2) 
    return Model(inputs = [src], 
                outputs = [x_in, x0, x1, x2, x3],
                name = 'UnetEncoder')
from keras.models import Model
from keras import layers
def unet(vol_size, enc_nf, dec_nf, full_size=True, edge_crop=48):
    """
    unet network for voxelmorph 
    Args:
        vol_size: volume size. e.g. (256, 256, 256)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]
            TODO: make this flexible.
        full_size
    """

    # inputs
    raw_src = Input(shape=vol_size + (1,), name = 'ImageInput')
    src = layers.GaussianNoise(0.25)(raw_src)
    enc_model = unet_enc(vol_size, enc_nf)
    # run the same encoder on the source and the target and concatenate the output at each level
    x_in, x0, x1, x2, x3 = [s_enc for s_enc in enc_model(src)]

    x = c2(x3, dec_nf[0])
    x = UpSampling2D()(x)
    x = concatenate([x, x2])
    x = c2(x, dec_nf[1])
    x = UpSampling2D()(x)
    x = concatenate([x, x1])
    x = c2(x, dec_nf[2])
    x = UpSampling2D()(x)
    x = concatenate([x, x0])
    x = c2(x, dec_nf[3])
    x = c2(x, dec_nf[4])
    x = UpSampling2D()(x)
    x = concatenate([x, x_in])
    x = c2(x, dec_nf[5])

    # transform the results into a flow.
    y_seg = Conv2D(1, kernel_size=3, padding='same', name='lungs', activation='sigmoid')(x)
    y_seg = layers.Cropping2D((edge_crop, edge_crop))(y_seg)
    y_seg = layers.ZeroPadding2D((edge_crop, edge_crop))(y_seg)
    # prepare model
    model = Model(inputs=[raw_src], outputs=[y_seg])
    return model
# use the predefined depths
nf_enc=[16,32,32,32]
nf_dec=[32,32,32,32,32,16,16,2]
net = unet(OUT_DIM, nf_enc, nf_dec)
# ensure the model roughly works
a= net.predict([np.zeros((1,)+OUT_DIM+(1,))])
print(a.shape)
net.summary()
from keras.optimizers import Adam
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

reg_param = 1.0
lr = 2e-4
dice_bce_param = 0.0
use_dice = True

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return dice_bce_param*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

net.compile(optimizer=Adam(lr=lr), 
              loss=[dice_p_bce], 
           metrics = [true_positive_rate, 'binary_accuracy'])
img_arr = np.array(img_vol)
seg_arr = np.array(seg_vol)
from sklearn.model_selection import train_test_split
train_vol, test_vol, train_seg, test_seg = train_test_split((img_arr-127.0)/127.0, 
                                                            (seg_arr>127).astype(np.float32), 
                                                            test_size = 0.2, 
                                                            random_state = 2018)
print('Train', train_vol.shape, 'Test', test_vol.shape, test_vol.mean(), test_vol.max())
print('Seg', train_seg.shape, train_seg.max(), np.unique(train_seg.ravel()))
fig, (ax1, ax1hist, ax2, ax2hist) = plt.subplots(1, 4, figsize = (20, 4))
ax1.imshow(test_vol[0, :, :, 0])
ax1hist.hist(test_vol.ravel())
ax2.imshow(test_seg[0, :, :, 0]>0.5)
ax2hist.hist(train_seg.ravel());
from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 5, 
                  width_shift_range = 0.05, 
                  height_shift_range = 0.05, 
                  shear_range = 0.01,
                  zoom_range = [0.8, 1.2],  
               # anatomically it doesnt make sense, but many images are flipped
                  horizontal_flip = True,  
                  vertical_flip = False,
                  fill_mode = 'nearest',
               data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)

def gen_augmented_pairs(in_vol, in_seg, batch_size = 16):
    while True:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_vol = image_gen.flow(in_vol, batch_size = batch_size, seed = seed)
        g_seg = image_gen.flow(in_seg, batch_size = batch_size, seed = seed)
        for i_vol, i_seg in zip(g_vol, g_seg):
            yield i_vol, i_seg
train_gen = gen_augmented_pairs(train_vol, train_seg, batch_size = 16)
test_gen = gen_augmented_pairs(test_vol, test_seg, batch_size = 16)
train_X, train_Y = next(train_gen)
test_X, test_Y = next(test_gen)
print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)
test_X.mean()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage(train_X[:, :, :, 0]), cmap = 'bone')
ax1.set_title('CXR Image')
ax2.imshow(montage(train_Y[:, :, :, 0]), cmap = 'bone')
ax2.set_title('Seg Image')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage(test_X[:, :, :, 0]), cmap = 'bone')
ax1.set_title('CXR Image')
ax2.imshow(montage(test_Y[:, :, :, 0]), cmap = 'bone')
ax2.set_title('Seg Image')
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('cxr_reg')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
from IPython.display import clear_output
loss_history = net.fit_generator(train_gen, 
                  steps_per_epoch=len(train_vol)//train_X.shape[0],
                  epochs = 25,
                  validation_data = (test_vol, test_seg),
                  callbacks=callbacks_list
                 )
#clear_output()
net.load_weights(weight_path)
net.save('full_model.h5')
import numpy as np
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
ax1.plot(loss_history.history['loss'], '-', label = 'Loss')
ax1.plot(loss_history.history['val_loss'], '-', label = 'Validation Loss')
ax1.legend()

ax2.plot(100*np.array(loss_history.history['binary_accuracy']), '-', 
         label = 'Accuracy')
ax2.plot(100*np.array(loss_history.history['val_binary_accuracy']), '-',
         label = 'Validation Accuracy')
ax2.legend()
import pydicom
from glob import glob
base_rsna_dir = os.path.join('..', 'input', 'rsna-pneumonia-detection-challenge')
test_mean, test_std = test_X.mean(), test_X.std()
print("{} {} ".format(test_mean, test_std))
def read_dicom_as_float(in_path):
    out_mat = pydicom.read_file(in_path).pixel_array
    norm_mat = (out_mat-1.0*np.mean(out_mat))/np.std(out_mat)
    # make the RSNA distribution look like the training distribution
    norm_mat = norm_mat*test_std+test_mean
    return np.expand_dims(norm_mat, -1).astype(np.float32)
all_rsna_df = pd.DataFrame({'path': glob(os.path.join(base_rsna_dir, 
                                                      'stage_*_images', '*.dcm'))})
all_rsna_df.sample(3)
n_shape = read_dicom_as_float(combined_df['dicom'].iloc[0]).shape
n_shape
pneumonia_locations = {}
# load table
with open(os.path.join('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'), mode='r') as infile:
    # open reader
    reader = csv.reader(infile)
    # skip header
    next(reader, None)
    # loop through rows
    for rows in reader:
        # retrieve information
        filename = rows[0]
        location = rows[1:5]
        pneumonia = rows[5]
        # if row contains pneumonia add label to dictionary
        # which contains a list of pneumonia locations per filename
        if pneumonia == '1':
            # convert string to float to int
            location = [int(float(i)) for i in location]
            # save pneumonia location in dictionary
            if filename in pneumonia_locations:
                pneumonia_locations[filename].append(location)
            else:
                pneumonia_locations[filename] = [location]
class generator_single_channel(keras.utils.Sequence):
    
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=100,
                 image_size=256, shuffle=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains pneumonia
        is_pneumonia = int(0)
        if filename in self.pneumonia_locations:
            # loop through pneumonia
            is_pneumonia = int(1)
            for location in self.pneumonia_locations[filename]:
                # add 1's at the location of the pneumonia
                x, y, w, h = location
                msk[y:y+h, x:x+w] = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect')
        # if augment then horizontal flip half the time
        # if self.augment and random.random() > 0.5:
        #     img = np.fliplr(img)
        #     msk = np.fliplr(msk)

        # add trailing channel dimension
        img = np.expand_dims(img, axis=-1)
        msk = np.expand_dims(msk, axis=-1)
        is_pneumonia = np.array(is_pneumonia)

        return img, msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, axis=-1)
        
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            
            return imgs,filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)

            return imgs,msks
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)
from keras import layers
in_shape = (1024,1024,1 )
in_img = layers.Input(in_shape, name='DICOMInput')
scale_factor = (2,2)
ds_dicom = layers.AvgPool2D(scale_factor)(in_img)
unet_out = net(ds_dicom)
us_out = layers.UpSampling2D(scale_factor)(unet_out)
unet_big = Model(inputs=[in_img], outputs=[us_out])
unet_big.save('big_model.h5')
unet_big.summary()
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
from skimage.util import montage
def add_boundary(in_img, in_seg, cmap = 'bone', norm = True, add_labels = True):
    if norm:
        n_img = (1.0*in_img-in_img.min())/(1.1*(in_img.max()-in_img.min()))
    else:
        n_img = in_img
    rgb_img = plt.cm.get_cmap(cmap)(n_img)[:, :, :3]
    if add_labels:
        return label2rgb(image = rgb_img, label = in_seg.astype(int), bg_label = 0)
    else:
        return mark_boundaries(image = rgb_img, label_img = in_seg.astype(int), color = (0, 1, 0), mode = 'thick')
def show_full_st(in_img, in_seg, gt_seg):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 10))
    out_mtg = add_boundary(montage(in_img[:, :, :, 0]), 
                           montage(gt_seg[:, :, :, 0]>0.5))
    ax1.imshow(out_mtg)
    ax1.set_title('Ground Truth')
    out_mtg = add_boundary(montage(in_img[:, :, :, 0]), 
                           montage(in_seg[:, :, :, 0]>0.5))
    ax2.imshow(out_mtg)
    ax2.set_title('Prediction')
    out_mtg = montage(in_seg[:, :, :, 0]-gt_seg[:, :, :, 0])
    ax3.imshow(out_mtg, cmap='RdBu', vmin=-1, vmax=1)
    ax3.set_title('Difference')
def show_examples(n=1, with_roi = True):
    roi_func = lambda x: x[:, 
                               OUT_DIM[0]//2-32:OUT_DIM[0]//2+32,
                               OUT_DIM[1]//2-64:OUT_DIM[1]//2,
                               :
                              ]
    for (test_X, test_Y), _ in zip(test_gen, range(n)):
        seg_Y = net.predict(test_X)
        show_full_st(test_X, seg_Y, test_Y)
        show_full_st(roi_func(test_X), roi_func(seg_Y), roi_func(test_Y))
opacity_sample_df = combined_df[combined_df.Target==1].sample(8)
opacity_sample_df.head()
parsed_opacity = parse_data(opacity_sample_df)
import numpy as np
fig, ax = plt.subplots(4, 3, figsize = (20, 15))
col=0
row=0

for index,c_row in opacity_sample_df.sample(8).iterrows():
    if row==4:
        break
    col=0
    c_img = read_dicom_as_float(c_row['dicom'])
    c_seg = unet_big.predict(np.expand_dims(c_img, 0))[0]
    ax[row,col].imshow(c_img[:, :, 0],cmap='bone')
    col = col +1 
    ax[row,col].imshow(add_boundary(c_img[:, :, 0], c_seg[:, :, 0]>0.5,add_labels = True ))
    pid=c_row['patientId']
    col =col+1
    fig.add_subplot(ax[row,col]);
    draw(parsed_opacity[pid])
    #print('looped')
    if col==2:
        row=row+1
    #print(row)    


combined_df
"""
import zipfile as zf
from io import BytesIO
from PIL import Image
batch_size = 12
with zf.ZipFile('masks.zip', 'w') as f:
    for i, c_rows in tqdm(all_rsna_df.groupby(lambda x: x//batch_size)):
        cur_x = np.stack(c_rows['path'].map(read_dicom_as_float), 0)
        cur_pred = unet_big.predict(cur_x)>0.5
        for out_img, (_, c_row) in zip(cur_pred[:, :, :, 0], c_rows.iterrows()):
            arc_name = os.path.relpath(c_row['path'], base_rsna_dir)
            arc_name, _ = os.path.splitext(arc_name)
            out_pil_obj = Image.fromarray((255*out_img).astype(np.uint8))
            out_obj = BytesIO()
            out_pil_obj.save(out_obj, format='png')
            out_obj.seek(0)
            f.writestr('{}.png'.format(arc_name), out_obj.read(), zf.ZIP_STORED)

"""
parsed_opacity

import keras
import tensorflow as tf
from keras.models import Model
from skimage.transform import resize
import os
import random
import csv
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
checkpoint = keras.callbacks.ModelCheckpoint("pnuemonia-detection-unet_{val_loss:.4f}.h5",monitor='val_loss',
                             verbose=1, save_best_only=False,save_weights_only=True, mode="auto")

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
folder = '../input/rsna-pneumonia-detection-challenge/stage_2_train_images'
filenames = os.listdir(folder)
random.shuffle(filenames)
# split into train and validation filenames
n_valid_samples = 6000
train_filenames = filenames[n_valid_samples:]
valid_filenames = filenames[:n_valid_samples]
print('n train samples', len(train_filenames))
print('n valid samples', len(valid_filenames))
n_train_samples = len(filenames) - n_valid_samples
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x
def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
input_img = Input((224, 224, 1), name='img')
model_unet = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model_unet.summary()
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

# def bce_loss(y_true,y_pred):
#   return keras.losses.binary_crossentropy(y_true,y_pred)

# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))
model_unet.compile(optimizer='adam',
                     loss=iou_loss,
                     metrics=[mean_iou,'accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint("pnuemonia-detection-unet_{val_loss:.4f}.h5",monitor='val_loss',
                             verbose=1, save_best_only=False,save_weights_only=True, mode="auto")

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
train_gen_simple = generator_single_channel(folder, train_filenames, pneumonia_locations, batch_size=64, image_size=224, shuffle=False, predict=False)
valid_gen_simple = generator_single_channel(folder, valid_filenames, pneumonia_locations, batch_size=64, image_size=224, shuffle=False, predict=False)
history = model_unet.fit_generator(train_gen_simple, validation_data=valid_gen_simple, callbacks=[checkpoint,es], epochs=5)
#confusion matrix, f1 score
#10% 
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(history.epoch, history.history["loss"], label="Train loss")
plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(132)
plt.plot(history.epoch, history.history["acc"], label="Train accuracy")
plt.plot(history.epoch, history.history["val_acc"], label="Valid accuracy")
plt.legend()
plt.subplot(133)
plt.plot(history.epoch, history.history["mean_iou"], label="Train iou")
plt.plot(history.epoch, history.history["val_mean_iou"], label="Valid iou")
plt.legend()
plt.show()
model_unet.load_weights('../input/capstone-rsna-pneumonia-detection-using-u-net/pnuemonia-detection-unet_0.6716.h5')
valid_gen_pred = generator_single_channel(folder, valid_filenames, pneumonia_locations,
                                            batch_size=64, image_size=224, shuffle=False, predict=True)