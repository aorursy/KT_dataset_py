%matplotlib inline
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from pydicom import read_file as read_dicom
import SimpleITK as sitk
base_dir = os.path.join('..', 'input')
all_dicom_paths = glob(os.path.join(base_dir, '*', '*', '*', '*', '*'))
print(len(all_dicom_paths), 'dicom files')
dicom_df = pd.DataFrame(dict(path = all_dicom_paths))
dicom_df['SliceNumber'] = dicom_df['path'].map(lambda x: int(os.path.splitext(x.split('/')[-1])[0][2:]))
dicom_df['SeriesName'] = dicom_df['path'].map(lambda x: x.split('/')[-2])
dicom_df['StudyID'] = dicom_df['path'].map(lambda x: x.split('/')[-3])
dicom_df['PatientID'] = dicom_df['path'].map(lambda x: x.split('/')[-4].split(' ')[0])
dicom_df.sample(3)
dicom_df.describe(include = 'all')
fig, m_axs = plt.subplots(3, 3, figsize = (20, 20))
for c_ax, (_, c_row) in zip(m_axs.flatten(), dicom_df.sample(9).iterrows()):
    try:
        c_slice = read_dicom(c_row['path'])
        c_ax.imshow(c_slice.pixel_array, cmap = 'bone')
        c_ax.set_title('{PatientID}\n{SeriesName}'.format(**c_row))
    except Exception as e:
        c_ax.set_title('{}'.format(str(e)[:40]))
        print(e)
    c_ax.axis('off')
fig, m_axs = plt.subplots(3, 3, figsize = (20, 20))
for c_ax, (_, c_row) in zip(m_axs.flatten(), dicom_df.sample(9).iterrows()):
    try:
        c_img = sitk.ReadImage(c_row['path'])
        c_slice = sitk.GetArrayFromImage(c_img)[0]
        c_ax.imshow(c_slice, cmap = 'bone')
        c_ax.set_title('{PatientID}\n{SeriesName}'.format(**c_row))
    except Exception as e:
        c_ax.set_title('{}'.format(str(e)[:40]))
        print(e)
    c_ax.axis('off')
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
patid_series_df = dicom_df[['PatientID', 'SeriesName']].drop_duplicates()
# keep only classes with more than two scans
valid_series = patid_series_df.groupby('SeriesName').count().reset_index().query('PatientID>2')['SeriesName']
series_name_encoder = LabelEncoder()
series_name_encoder.fit(valid_series.values)
patid_series_df = patid_series_df[patid_series_df['SeriesName'].isin(valid_series)]
valid_dicom_df = dicom_df[dicom_df['SeriesName'].isin(valid_series)].copy()
valid_dicom_df['cat_vec'] = valid_dicom_df['SeriesName'].map(lambda x: to_categorical(series_name_encoder.transform([x]), num_classes = len(series_name_encoder.classes_)))
print(patid_series_df.shape[0], 'unique groups', valid_dicom_df.shape[0], 'rows', len(series_name_encoder.classes_), 'classes')
from sklearn.model_selection import train_test_split
train_ids, test_ids = train_test_split(patid_series_df[['PatientID', 'SeriesName']], 
                                       test_size = 0.25, 
                                       stratify = patid_series_df['SeriesName'])

train_unbalanced_df = valid_dicom_df[valid_dicom_df['PatientID'].isin(train_ids['PatientID'])]
test_df = valid_dicom_df[valid_dicom_df['PatientID'].isin(test_ids['PatientID'])]
print(train_unbalanced_df.shape[0], 'training images', test_df.shape[0], 'testing images')
train_unbalanced_df['SeriesName'].hist(figsize = (10, 5))
train_df = train_unbalanced_df.groupby(['SeriesName']).apply(lambda x: x.sample(1500, replace = True)
                                                      ).reset_index(drop = True)
print('New Data Size:', train_df.shape[0], 'Old Size:', train_unbalanced_df.shape[0])
train_df['SeriesName'].hist(figsize = (20, 5))
from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (128, 128) # many of the ojbects are small so 512x512 lets us see them
img_gen_args = dict(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.05, 
                              width_shift_range = 0.02, 
                              rotation_range = 3, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range = 0.05)
img_gen = ImageDataGenerator(**img_gen_args)
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, seed = None, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways: seed: {}'.format(seed))
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                              seed = seed,
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.concatenate(in_df[y_col].values,0)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen
import keras.preprocessing.image as KPImage
from PIL import Image
def apply_window(data, center, width):
    low = center - width/2.
    high = center + width/2
    data = np.clip(data, low, high)
    data += -1 * low
    data /= width
    return data
def read_dicom_image(in_path):
    c_img = sitk.ReadImage(in_path)
    c_slice = sitk.GetArrayFromImage(c_img)[0]
    return c_slice
    
class pil_image_awesome():
    @staticmethod
    def open(in_path):
        if '.dcm' in in_path:
            # we only want to keep the positive labels not the background
            c_slice = read_dicom_image(in_path)
            wind_slice = apply_window(c_slice, 40, 400)
            int_slice =  (255*wind_slice).clip(0, 255).astype(np.uint8) # 8bit images are more friendly
            return Image.fromarray(int_slice)
        else:
            return Image.open(in_path)
    fromarray = Image.fromarray
KPImage.pil_image = pil_image_awesome
batch_size = 16
train_gen = flow_from_dataframe(img_gen, train_df, 
                             path_col = 'path',
                            y_col = 'cat_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = batch_size)
test_gen = flow_from_dataframe(img_gen, test_df, 
                             path_col = 'path',
                            y_col = 'cat_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = batch_size)
t_x, t_y = next(train_gen)
print(t_x.shape, '->', t_y.shape)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    c_ax.set_title('{}'.format(series_name_encoder.classes_[np.argmax(c_y, -1)]))
    c_ax.axis('off')
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
ct_model = Sequential()
ct_model.add(BatchNormalization(input_shape = t_x.shape[1:]))
ct_model.add(MobileNet(input_shape = (None, None, 1), include_top = False, weights = None))
ct_model.add(GlobalAveragePooling2D())
ct_model.add(Dropout(0.5))
ct_model.add(Dense(128))
ct_model.add(Dropout(0.5))
ct_model.add(Dense(t_y.shape[1], activation = 'softmax'))
from keras.metrics import top_k_categorical_accuracy
def top_5_accuracy(in_gt, in_pred):
    return top_k_categorical_accuracy(in_gt, in_pred, k=5)

ct_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy', top_5_accuracy])
ct_model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('cthead')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=6) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
ct_model.fit_generator(train_gen, 
                       steps_per_epoch = 50,
                        validation_data = test_gen, 
                       validation_steps = 50,
                              epochs = 5, 
                              callbacks = callbacks_list,
                             workers = 4,
                             use_multiprocessing=False, 
                             max_queue_size = 10
                            )
_, acc, top5_acc = ct_model.evaluate_generator(test_gen, steps = 50, workers=4)
print('Overall Accuracy: %2.1f%%\nTop 5 Accuracy %2.1f%%' % (acc*100, top5_acc*100))
