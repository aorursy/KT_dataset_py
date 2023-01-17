# Importing required libraries 

import numpy as np 

import pandas as pd 

import os

from glob import glob

%matplotlib inline

import matplotlib.pyplot as plt
all_xray_df = pd.read_csv('../input/Data_Entry_2017.csv')

all_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('..', 'input', 'images*', '*', '*.png'))}

print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])

all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

all_xray_df.sample(3)
all_xray_df['Finding Labels']=all_xray_df['Finding Labels'].map(lambda x: 'Unhealthy' if x != "No Finding" else "Healthy" )

all_xray_df['Finding Labels'].value_counts()
all_xray_df['Output'] = all_xray_df['Finding Labels'].map({'Unhealthy': 1 , 'Healthy': 0 })

all_xray_df.tail()
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img



liist=['Image Index','path', 'Finding Labels','Output']

all_xray_df=pd.DataFrame(data=all_xray_df,columns=all_xray_df[liist].columns)

a=(all_xray_df[all_xray_df['Finding Labels'].isin(['Healthy'])][0:9000]).index.values

all_xray_df=all_xray_df.drop(a)

all_xray_df['Finding Labels'].value_counts()
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(all_xray_df, 

                                   test_size = 0.25, 

                                   random_state = 2018)

print('train', train_df.shape[0], 'Validation', valid_df.shape[0])

training_samples=77340

validation_samples = 25780
## Below code was not used as simple np.stack function did the trick

## basically it was for providing model all labels of images in one array



from sklearn.preprocessing import LabelEncoder

train_labels=train_df['Finding Labels'].values

validation_labels = valid_df['Finding Labels'].values

le = LabelEncoder()

le.fit(train_labels)

# encode wine type labels

train_labels_enc = le.transform(train_labels)

validation_labels_enc = le.transform(validation_labels)



print(train_labels[0:5], train_labels_enc[0:5])



## encodings
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (150,150)



core_idg = ImageDataGenerator(rescale=1./255

                              ,samplewise_center=True, 

                              samplewise_std_normalization=True, 

                              horizontal_flip = True, 

                              vertical_flip = False, 

                              height_shift_range= 0.05, 

                              width_shift_range=0.1, 

                              rotation_range=5, 

                              shear_range = 0.1,

                              fill_mode = 'reflect',

                              zoom_range=0.15)
def flow_from_dataframe(img_data_gen, in_df, path_col,y_col, **dflow_args):

    base_dir = os.path.dirname(in_df[path_col].values[0])

    df_gen = img_data_gen.flow_from_directory(base_dir, 

                                     class_mode = 'binary',

                                    **dflow_args)

    df_gen.filenames = in_df[path_col].values

    df_gen.classes = np.stack(in_df[y_col].values)

    df_gen.samples = in_df.shape[0]

    df_gen.n = in_df.shape[0]

    df_gen._set_index_array()

    df_gen.directory = '' # since we have the full path

    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))

    return df_gen
train_gen = flow_from_dataframe(core_idg, train_df, 

                             path_col = 'path',

                                color_mode='rgb',

                                y_col='Output',

                                batch_size = 100,

                            target_size = IMG_SIZE)



valid_gen = flow_from_dataframe(core_idg, valid_df, 

                             path_col = 'path',

                                color_mode='rgb', 

                                 y_col='Output',

                                batch_size = 150,

                            target_size = IMG_SIZE)



test_gen= flow_from_dataframe(core_idg, 

                               valid_df, 

                             path_col = 'path',

                             color_mode='rgb',

                                    y_col='Output',

                            target_size = IMG_SIZE,

                            batch_size = 1)

# train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,

#                                    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 

#                                    horizontal_flip=True, fill_mode='nearest')

# valid_datagen = ImageDataGenerator(rescale=1./255)



# T_base_dir = os.path.dirname(train_df['path'].values[0])

# V_base_dir = os.path.dirname(valid_df['path'].values[0])

# batch_size =30

# train_generator = train_datagen.flow_from_directory(

#     T_base_dir,

#     target_size = (299,299),

#     color_mode = 'rgb',

#     batch_size = batch_size,

#     class_mode = 'binary')

# train_generator.classes = train_labels_enc



# validation_generator = valid_datagen.flow_from_directory(

#     V_base_dir,

#     target_size = (299,299),

#     color_mode = 'rgb',

#     batch_size = batch_size,

#     class_mode = 'binary')

# validation_generator.classes = validation_labels_enc



# test_generator = valid_datagen.flow_from_directory(

#     V_base_dir,

#     target_size = (299,299),

#     color_mode = 'rgb',

#     batch_size = 1,

#     class_mode = 'binary',

#     shuffle = False)
# t_x, t_y = next(train_gen)

# test_X, test_Y = next(test_gen)

batch_size = 30

num_classes = 2

epochs = 100

input_shape = (150,150, 3)

print(training_samples , validation_samples)



steps_per_epoch= (training_samples )/100

val_steps = validation_samples/150
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

from keras.models import Sequential

from keras import optimizers

from keras.applications import VGG16



vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)



for layer in vgg_conv.layers[:-9]:

    layer.trainable = False



for layer in vgg_conv.layers:

    print(layer, layer.trainable)

    

model = Sequential()

model.add(vgg_conv)

model.add(Dense(512, activation='relu', input_dim=input_shape ))

model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-5),

              metrics=['accuracy'])



model.summary()
history = model.fit_generator(    train_gen, 

                                  validation_data = valid_gen, 

                                  epochs = 10,

                                  steps_per_epoch= steps_per_epoch,

                              validation_steps= val_steps,

                                  verbose=1)





print(history.history.keys())

#  "Accuracy"

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# "Loss"

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
for c_label, s_count in zip(all_xray_df['Finding Labels'].unique(), 100*np.mean(test_Y,0)):

    print('%s: %2.2f%%' % (c_label, s_count))
pred_Y = model.predict(test_X, batch_size = 32, verbose = True)
from sklearn.metrics import roc_curve, auc

fig, c_ax = plt.subplots(1,1, figsize = (9, 9))

for (idx, c_label) in enumerate(all_xray_df['Finding Labels'].unique()):

    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])

    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

c_ax.legend()

c_ax.set_xlabel('False Positive Rate')

c_ax.set_ylabel('True Positive Rate')

fig.savefig('barely_trained_net.png')
model.save('modelVgg.h5')
# look at how often the algorithm predicts certain diagnoses 

for c_label, p_count, t_count in zip(all_xray_df['Finding Labels'].values, 

                                     100*np.mean(pred_Y,0), 

                                     100*np.mean(test_Y,0)):

    print('%s: Dx: %2.2f%%, PDx: %2.2f%%' % (c_label, t_count, p_count))
all_xray_df['Finding Labels'].unique()
from sklearn.metrics import roc_curve, auc

fig, c_ax = plt.subplots(1,1, figsize = (9, 9))

for (idx, c_label) in enumerate(all_xray_df['Finding Labels'].unique()):

    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])

    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

c_ax.legend()

c_ax.set_xlabel('False Positive Rate')

c_ax.set_ylabel('True Positive Rate')

fig.savefig('trained_net.png')
sickest_idx = np.argsort(np.sum(test_Y, 1)<1)

fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))

for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):

    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')

    stat_str = [n_class[:6] for n_class, n_score in zip(all_xray_df['Finding Labels'].unique(), 

                                                                  test_Y[idx]) 

                             if n_score>0.5]

    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(all_xray_df['Finding Labels'].unique(), 

                                                                  test_Y[idx], pred_Y[idx]) 

                             if (n_score>0.5) or (p_score>0.5)]

    c_ax.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))

    c_ax.axis('off')

fig.savefig('trained_img_predictions.png')