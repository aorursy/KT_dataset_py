!mkdir ~/.keras

!mkdir ~/.keras/models

!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/

!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/

!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob 

import matplotlib.pyplot as plt

import os

import pandas as pd

import seaborn as sns

from skimage.util import montage

from skimage.io import imread

base_dir = os.path.join('..', 'input', 'pulmonary-chest-xray-abnormalities')

mont_paths = glob(os.path.join(base_dir, 'Montgomery', 'MontgomerySet', '*', '*.*'))

shen_paths = glob(os.path.join(base_dir, 'ChinaSet_AllFiles', 'ChinaSet_AllFiles', '*', '*.*'))

print('Montgomery Files', len(mont_paths))

print('Shenzhen Files', len(shen_paths))

all_paths_df = pd.DataFrame(dict(path = mont_paths + shen_paths))

all_paths_df['source'] = all_paths_df['path'].map(lambda x: x.split('/')[3])

all_paths_df['file_id'] = all_paths_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])

all_paths_df['patient_group']  = all_paths_df['file_id'].map(lambda x: x.split('_')[0])



all_paths_df['file_ext'] = all_paths_df['path'].map(lambda x: os.path.splitext(x)[1][1:])

all_paths_df = all_paths_df[all_paths_df.file_ext.isin(['png', 'txt'])]

all_paths_df['pulm_state']  = all_paths_df['file_id'].map(lambda x: int(x.split('_')[-1]))

all_paths_df.sample(5)
clean_patients_df = all_paths_df.pivot_table(index = ['patient_group', 'pulm_state', 'file_id'], 

                                             columns=['file_ext'], 

                                             values = 'path', aggfunc='first').reset_index()

clean_patients_df.sample(5)

from warnings import warn

def report_to_dict(in_path):

    with open(in_path, 'r') as f:

        all_lines = [x.strip() for x in f.read().split('\n')]

    info_dict = {}

    try:

        if "Patient's Sex" in all_lines[0]:

            info_dict['age'] = all_lines[1].split(':')[-1].strip().replace('Y', '')

            info_dict['sex'] = all_lines[0].split(':')[-1].strip()

            info_dict['report'] = ' '.join(all_lines[2:]).strip()

        else:

            info_dict['age'] = all_lines[0].split(' ')[-1].replace('yrs', '').replace('yr', '')

            info_dict['sex'] = all_lines[0].split(' ')[0].strip()

            info_dict['report'] = ' '.join(all_lines[1:]).strip()

        

        info_dict['sex'] = info_dict['sex'].upper().replace('FEMALE', 'F').replace('MALE', 'M').replace('FEMAL', 'F')[0:1]

        if 'month' in info_dict.get('age', ''):

            info_dict.pop('age') # invalid

        if 'day' in info_dict.get('age', ''):

            info_dict.pop('age') # invalid

        elif len(info_dict.get('age',''))>0:

            info_dict['age'] = float(info_dict['age'])

        else:

            info_dict.pop('age')

        return info_dict

    except Exception as e:

        print(all_lines)

        warn(str(e), RuntimeWarning)

        return {}

report_df = pd.DataFrame([dict(**report_to_dict(c_row.pop('txt')), **c_row) 

              for  _, c_row in clean_patients_df.iterrows()])

report_df.sample(5)
report_df[['age', 'patient_group', 'pulm_state', 'sex']].hist(figsize = (10, 5))
from sklearn.model_selection import train_test_split

raw_train_df, valid_df = train_test_split(report_df, 

                                   test_size = 0.25, 

                                   random_state = 2018,

                                   stratify = report_df[['pulm_state', 'patient_group']])

print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])

raw_train_df.sample(1)
train_df = raw_train_df.groupby(['pulm_state', 'patient_group']).apply(lambda x: x.sample(400, replace = True)

                                                      ).reset_index(drop = True)

print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])

train_df[['pulm_state', 'patient_group']].hist(figsize = (10, 5))
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.imagenet_utils import preprocess_input

from PIL import Image

ppi = lambda x: Image.fromarray(preprocess_input(np.array(x).astype(np.float32)))

IMG_SIZE = (224, 224) # slightly smaller than vgg16 normally expects

core_idg = ImageDataGenerator(samplewise_center=False, 

                              samplewise_std_normalization=False, 

                              horizontal_flip = True, 

                              vertical_flip = False, 

                              height_shift_range = 0.15, 

                              width_shift_range = 0.15, 

                              rotation_range = 5, 

                              shear_range = 0.01,

                              fill_mode = 'nearest',

                              zoom_range=0.10)
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):

    base_dir = os.path.dirname(in_df[path_col].values[0])

    print('## Ignore next message from keras, values are replaced anyways')

    df_gen = img_data_gen.flow_from_directory(base_dir, 

                                     class_mode = 'sparse',

                                    **dflow_args)

    df_gen.filenames = in_df[path_col].values

    df_gen.classes = np.stack(in_df[y_col].values)

    df_gen.samples = in_df.shape[0]

    df_gen.n = in_df.shape[0]

    df_gen._set_index_array()

    df_gen.directory = '' # since we have the full path

    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))

    return df_gen
from tensorflow.keras.utils import to_categorical

train_gen = flow_from_dataframe(core_idg, train_df, 

                             path_col = 'png',

                            y_col = 'pulm_state', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 8)



valid_gen = flow_from_dataframe(core_idg, valid_df, 

                             path_col = 'png',

                            y_col = 'pulm_state', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 32) # we can use much larger batches for evaluation

# used a fixed dataset for evaluating the algorithm

test_X, test_Y = next(flow_from_dataframe(core_idg, 

                               valid_df, 

                             path_col = 'png',

                            y_col = 'pulm_state', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',batch_size=32)) # one big batch

t_x, t_y = next(train_gen)

fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))

for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):

    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = 0, vmax = 255)

    c_ax.set_title('%s' % ('Pulmonary Abnormality' if c_y>0.5 else 'Healthy'))

    c_ax.axis('off')
from tensorflow.keras.applications import ResNet50,VGG16

from tensorflow.keras import regularizers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D,Dropout,Input



#image_input=Input(shape=(224,224,3))

weights='../input/keras_pretrained_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

HEIGHT=224

WIDTH=224

DEPTH=3



model=Sequential()

model.add(ResNet50(include_top=False,pooling='avg',weights='imagenet'))

model.add(Dense(1024,activation='relu'))

model.add(Dense(1024,activation='relu'))

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.7))

model.add(Dense(2,activation='softmax',activity_regularizer=regularizers.l2(0.01)))

model.layers[0].trainable=False

from tensorflow.python.keras.optimizers import SGD,RMSprop,Adam

sgd=SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)

model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

!rm -rf ~/.keras # clean up the model / make space for other things
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('tb_detector')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', 

                             save_best_only=True, save_weights_only = True)





reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto')

early = EarlyStopping(monitor="val_loss",  patience=5) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early]
history=model.fit_generator(train_gen,

                       validation_data = (test_X, test_Y),

                       steps_per_epoch=5,

                       epochs = 40,

                   verbose=1)



import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.figure()



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()
# load the best version of the model

#model.load_weights(weight_path)

model.save('full_tb_model.h5',include_optimizer=False)
pred_Y = model.predict(test_X, batch_size = 8 , verbose = True)
scores=model.evaluate(test_X,test_Y)