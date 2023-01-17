# params we will probably want to do some hyperparameter optimization later
IMG_SIZE = (384, 384) # [(224, 224), (384, 384), (512, 512), (640, 640)]
BATCH_SIZE = 24 # [1, 8, 16, 24]
TRAIN_SAMPLES = 8000 # [3000, 6000, 15000]
TEST_SAMPLES = 800
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Add input path to file path

image_bbox_df = pd.read_csv('../input/lung-opacity-overview/image_bbox_full.csv')
image_bbox_df['path'] = image_bbox_df['path'].map(lambda x: x.replace('input', 
                                                            'input/rsna-pneumonia-detection-challenge'))
print(image_bbox_df.shape[0], 'images')
image_bbox_df.sample(3)
# get the labels in the right format
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Turn classes into numerical encodings
class_enc = LabelEncoder()
image_bbox_df['class_idx'] = class_enc.fit_transform(image_bbox_df['class'])
# One hot encode class
oh_enc = OneHotEncoder(sparse=False)
image_bbox_df['class_vec'] = oh_enc.fit_transform(
    image_bbox_df['class_idx'].values.reshape(-1, 1)).tolist() 
image_bbox_df.sample(3)
from sklearn.model_selection import train_test_split
image_df = image_bbox_df.groupby('patientId').apply(lambda x: x.sample(1))
raw_train_df, valid_df = train_test_split(image_df, test_size=0.25, random_state=2018,
                                    stratify=image_df['class'])
print(raw_train_df.shape, 'training data')
print(valid_df.shape, 'validation data')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
raw_train_df.groupby('class').size().plot.bar(ax=ax1)
train_df = raw_train_df.groupby('class').apply(lambda x: x.sample(TRAIN_SAMPLES//3)).reset_index(drop=True)
train_df.groupby('class').size().plot.bar(ax=ax2) 
print(train_df.shape[0], 'new training size')
import keras_preprocessing.image as KPImage
from PIL import Image
import pydicom

def read_dicom_image(in_path):
    img_arr = pydicom.read_file(in_path).pixel_array
    return img_arr/img_arr.max()
class medical_pil():
    @staticmethod
    def open(in_path):
        if '.dcm' in in_path:
            c_slice = read_dicom_image(in_path)
            int_slice =  (255*c_slice).clip(0, 255).astype(np.uint8) # 8bit images are more friendly
            return Image.fromarray(int_slice)
        else:
            return Image.open(in_path)
    fromarray = Image.fromarray
# Overwrite image loading
KPImage.pil_image = medical_pil
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
# MODIFY THIS TO ADD IMAGE AUGMENTATION
img_gen_args = dict(preprocessing_function=preprocess_input)
img_gen = ImageDataGenerator(**img_gen_args)
'''
Clever hacking of the standard generator to load it from a dataframe. 
Replaces the file paths keras would from with the paths from the dataframe.

'''
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, seed = None, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways: seed: {}'.format(seed))
    df_gen = img_data_gen.flow_from_directory(base_dir,class_mode = 'sparse',seed = seed,**dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values,0)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen
train_gen = flow_from_dataframe(img_gen, train_df,path_col = 'path',y_col = 'class_vec',
                                target_size = IMG_SIZE,color_mode = 'rgb',batch_size = BATCH_SIZE)

valid_gen = flow_from_dataframe(img_gen, valid_df,path_col = 'path',y_col = 'class_vec',
                                target_size = IMG_SIZE,color_mode = 'rgb',batch_size = 256) 
# used a fixed dataset for evaluating the algorithm
valid_X, valid_Y = next(flow_from_dataframe(img_gen,valid_df,path_col = 'path',y_col = 'class_vec',
                                            target_size = IMG_SIZE,color_mode = 'rgb',
                                            batch_size = TEST_SAMPLES))
t_x, t_y = next(train_gen)
print(t_x.shape, t_y.shape)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    c_ax.set_title('%s' % class_enc.classes_[np.argmax(c_y)])
    c_ax.axis('off')
base_pretrained_model = VGG16(input_shape =  t_x.shape[1:],include_top = False, weights = 'imagenet')
base_pretrained_model.trainable = True
# Make the last n conv layer trainable
n = 2
for layer in base_pretrained_model.layers[:-n]:
    layer.trainable = False
base_pretrained_model.summary()
# MODEL HYPER PARAMS
# MODIFY THESE
DENSE_COUNT = 128 # [32, 64, 128, 256]
DROPOUT = 0.25 # [0, 0.25, 0.5]
LEARN_RATE = 1e-4 # [1e-4, 1e-3, 4e-3]
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, LeakyReLU
from keras.layers import Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.layers import AvgPool2D,BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
pneu_model = Sequential(name = 'combined_model')
#base_pretrained_model.trainable = False
pneu_model.add(base_pretrained_model)
pneu_model.add(BatchNormalization())
pneu_model.add(GlobalAveragePooling2D())
pneu_model.add(Dense(DENSE_COUNT, activation = 'linear', use_bias=False))
pneu_model.add(Dropout(DROPOUT))
pneu_model.add(BatchNormalization())
pneu_model.add(LeakyReLU(0.1))
pneu_model.add(Dense(t_y.shape[1], activation = 'softmax'))

pneu_model.compile(optimizer = Adam(lr = LEARN_RATE), loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy'])
pneu_model.summary()
train_gen.batch_size = BATCH_SIZE
pneu_model.fit_generator(train_gen, 
                         steps_per_epoch=train_gen.n//BATCH_SIZE,
                         validation_data=(valid_X, valid_Y), 
                         epochs=20,
                         workers=2)
pneu_model.save('full_model.h5')
pred_Y = pneu_model.predict(valid_X, 
                          batch_size = BATCH_SIZE, 
                          verbose = True)
from sklearn.metrics import classification_report, confusion_matrix
plt.matshow(confusion_matrix(np.argmax(valid_Y, -1), np.argmax(pred_Y,-1)))
print(classification_report(np.argmax(valid_Y, -1), 
                            np.argmax(pred_Y,-1), target_names = class_enc.classes_))
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(np.argmax(valid_Y,-1)==0, pred_Y[:,0])
fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 250)
ax1.plot(fpr, tpr, 'b.-', label = 'VGG-Model (AUC:%2.2f)' % roc_auc_score(np.argmax(valid_Y,-1)==0, pred_Y[:,0]))
ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')
ax1.legend(loc = 4)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate');
ax1.set_title('Lung Opacity ROC Curve')
fig.savefig('roc_valid.pdf')
from glob import glob
sub_img_df = pd.DataFrame({'path': 
              glob('../input/rsna-pneumonia-detection-challenge/stage_2_test_images/*.dcm')})
sub_img_df['patientId'] = sub_img_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
sub_img_df.sample(3)
submission_gen = flow_from_dataframe(img_gen, 
                                     sub_img_df, 
                             path_col = 'path',
                            y_col = 'patientId', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = BATCH_SIZE,
                                    shuffle=False)
from tqdm import tqdm
sub_steps = 2*sub_img_df.shape[0]//BATCH_SIZE
out_ids, out_vec = [], []
for _, (t_x, t_y) in zip(tqdm(range(sub_steps)), submission_gen):
    out_vec += [pneu_model.predict(t_x)]
    out_ids += [t_y]
out_vec = np.concatenate(out_vec, 0)
out_ids = np.concatenate(out_ids, 0)
pred_df = pd.DataFrame(out_vec, columns=class_enc.classes_)
pred_df['patientId'] = out_ids
pred_avg_df = pred_df.groupby('patientId').agg('mean').reset_index()
pred_avg_df['Lung Opacity'].hist()
pred_avg_df.to_csv('image_level_class_probs.csv', index=False) # not hte submission file
pred_avg_df.sample(2)
pred_avg_df['PredictionString'] = pred_avg_df['Lung Opacity'].map(lambda x: ('%2.2f 0 0 1024 1024' % x) if x>0.5 else '')
pred_avg_df[['patientId', 'PredictionString']].to_csv('submission.csv', index=False)
