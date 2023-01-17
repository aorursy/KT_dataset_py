import os
import pandas as pd
import seaborn as sns
import numpy as np

from os import path
from matplotlib import pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 

xray_data = pd.read_csv('../input/data/Data_Entry_2017.csv')

num_obs = len(xray_data)
print(f'Number of observations: {num_obs}')

xray_data.head(10)
images_glob = glob('../input/data/images_*/images/*.png')
print(f'Number of images {len(images_glob)}')
image_paths = { os.path.basename(path): path for path in images_glob }
xray_data['full_path'] = xray_data['Image Index'].map(image_paths.get)

mun_unique_labels = xray_data['Finding Labels'].nunique()
print(f'Number of unique labels {mun_unique_labels}')

count_per_unique_label = xray_data['Finding Labels'].value_counts()
df_count_per_unique_label = count_per_unique_label.to_frame()
print(df_count_per_unique_label)

sns.barplot(x=df_count_per_unique_label.index[:20], y='Finding Labels', data=df_count_per_unique_label[:20], color='green')
plt.xticks(rotation=90)
dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

for label in dummy_labels:
    xray_data[label] = xray_data['Finding Labels'].map(lambda res: 1. if label in res else 0.)
    
xray_data.head(20)
clean_labels = xray_data[dummy_labels].sum().sort_values(ascending=False)
df_clean_labels = clean_labels.to_frame()
sns.barplot(x=df_clean_labels.index[::], y=0, data=df_clean_labels[::], color='green')
plt.xticks(rotation=90)
xray_data['target_vector'] = xray_data.apply(lambda target: [target[dummy_labels].values], 1).map(lambda target: target[0])
xray_data.head(10)
train_set, test_set = train_test_split(xray_data, test_size=0.2, random_state=0)
print(f'Number of train set {len(train_set)}')
print(f'Number of test set {len(test_set)}')
data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
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
image_size = (224, 224)
train_gen = flow_from_dataframe(data_gen, train_set, path_col='full_path', y_col='target_vector', target_size=image_size, color_mode='grayscale', batch_size=32)
valid_gen = flow_from_dataframe(data_gen, test_set, path_col='full_path', y_col='target_vector', target_size=image_size, color_mode='grayscale', batch_size=128)
X_test, y_test = next(flow_from_dataframe(data_gen, test_set, path_col='full_path', y_col='target_vector', target_size=image_size, color_mode='grayscale', batch_size=2048))
def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(dummy_labels), activation='softmax'))
    
    return model
model = build_model(X_test.shape[1:])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
check_point = ModelCheckpoint(filepath='weights.best.{epoch:02d}-{val_loss:2f}.hdf5', verbose=1, save_best_only=True)
model.fit_generator(generator=train_gen,
                    steps_per_epoch=20,
                    validation_data=(X_test, y_test),
                    epochs=10,
                    callbacks=[check_point])
quick_model_predictions = model.predict(X_test, batch_size = 64, verbose = 1)

# import libraries
from sklearn.metrics import roc_curve, auc

# create plot
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (i, label) in enumerate(dummy_labels):
    fpr, tpr, thresholds = roc_curve(y_test[:,i].astype(int), quick_model_predictions[:,i])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

# Set labels for plot
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('quick_trained_model.png')