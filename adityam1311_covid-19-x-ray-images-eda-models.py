# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# v1

import matplotlib.pyplot as plt

import seaborn as sns



from PIL import Image



# v3

import warnings

warnings.filterwarnings('ignore')



from sklearn.utils import shuffle



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

from keras.optimizers import Adam

from keras.losses import binary_crossentropy

from keras.callbacks import LearningRateScheduler

from keras.metrics import *

# v4



ACCURACY_LIST = []

from keras.applications.resnet50 import ResNet50

from keras.layers import GlobalMaxPooling2D

from keras.models import Model



# v5

!pip install efficientnet

from efficientnet.keras import EfficientNetB4

from keras import backend as K



# v6

# Get reproducible results

from numpy.random import seed

seed(1)

import tensorflow as tf

tf.random.set_seed(1)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
metadata = pd.read_csv('/kaggle/input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')

summary = pd.read_csv('/kaggle/input/coronahack-chest-xraydataset/Chest_xray_Corona_dataset_Summary.csv')



metadata.sample(10)
train_data = metadata[metadata['Dataset_type'] == 'TRAIN']

test_data = metadata[metadata['Dataset_type'] == 'TEST']

assert train_data.shape[0] + test_data.shape[0] == metadata.shape[0]

print(f"Shape of train data : {train_data.shape}")

print(f"Shape of test data : {test_data.shape}")

test_data.sample(10)
# Null value calculation

print(f"Count of null values in train :\n{train_data.isnull().sum()}")

print(f"Count of null values in test :\n{test_data.isnull().sum()}")
# Substitute null values with string unknown

train_fill = train_data.fillna('unknown')

test_fill = test_data.fillna('unknown')



train_fill.sample(10)
# Count plot for 3 attributes with unknown variable addition

targets = ['Label', 'Label_2_Virus_category', 'Label_1_Virus_category']

fig, ax = plt.subplots(2, 2, figsize=(20, 10))

sns.countplot(x=targets[0], data=train_fill, ax=ax[0, 0])

sns.countplot(x=targets[1], data=train_fill, ax=ax[0, 1])

sns.countplot(x=targets[2], data=train_fill, ax=ax[1, 0])

plt.show()
# Pie chart representation of Label_2_Virus_category values



colors = ['#ff5733', '#33ff57']

explode = [0.02, 0.02]



values = ['unknown', 'other']

percentages = [100 * (train_fill[train_fill[targets[1]] == 'unknown'].shape[0]) / train_fill.shape[0],

              100 * (train_fill[train_fill[targets[1]] != 'unknown'].shape[0]) / train_fill.shape[0]]



fig1, ax1 = plt.subplots(figsize=(7, 7))



plt.pie(percentages, colors=colors, labels=values,

        autopct='%1.1f%%', startangle=0, explode=explode)

fig = plt.gcf()

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig.gca().add_artist(centre_circle)



ax1.axis('equal')

plt.tight_layout()

plt.title('Percentage of "unknown" values present in Label_2_Virus_category')

plt.show()
# Count plot for 3 target variables without filling unknown variable

fig, ax = plt.subplots(2, 2, figsize=(20, 10))

sns.countplot(x=targets[0], data=train_data, ax=ax[0, 0])

sns.countplot(x=targets[1], data=train_data, ax=ax[0, 1])

sns.countplot(x=targets[2], data=train_data, ax=ax[1, 0])

plt.show()
print(f"Label = Normal Cases : {train_data[train_data['Label'] == 'Normal'].shape[0]}")

print(f"""Label = Pnemonia + Label_2_Virus_category = COVID-19 cases : {train_data[(train_data['Label'] == 'Pnemonia')

      & (train_data['Label_2_Virus_category'] == 'COVID-19')].shape[0]}""")

print(f"""Label = Normal + Label_2_Virus_category = COVID-19 cases : {train_data[(train_data['Label'] == 'Normal')

      & (train_data['Label_2_Virus_category'] == 'COVID-19')].shape[0]}""")
TEST_FOLDER = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'

TRAIN_FOLDER = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'



assert os.path.isdir(TEST_FOLDER) == True

assert os.path.isdir(TRAIN_FOLDER) == True
sample_train_images = list(os.walk(TRAIN_FOLDER))[0][2][:8]

sample_train_images = list(map(lambda x: os.path.join(TRAIN_FOLDER, x), sample_train_images))



sample_test_images = list(os.walk(TEST_FOLDER))[0][2][:8]

sample_test_images = list(map(lambda x: os.path.join(TEST_FOLDER, x), sample_test_images))
# Plot sample training images

plt.figure(figsize=(20, 20))



for iterator, filename in enumerate(sample_train_images):

    image = Image.open(filename)

    plt.subplot(4, 2, iterator+1)

    plt.axis('off')

    plt.imshow(image)





plt.tight_layout()
# Plot sample testing images

plt.figure(figsize=(20, 20))



for iterator, filename in enumerate(sample_test_images):

    image = Image.open(filename)

    plt.subplot(4, 2, iterator+1)

    plt.axis('off')

    plt.imshow(image)





plt.tight_layout()
# Plot b/w image histograms of Label_2_Virus_category type "COVID-19" patients 

fig, ax = plt.subplots(4, 2, figsize=(20, 20))



covid19_type_file_paths = train_data[train_data['Label_2_Virus_category'] == 'COVID-19']['X_ray_image_name'].values

sample_covid19_file_paths = covid19_type_file_paths[:4]

sample_covid19_file_paths = list(map(lambda x: os.path.join(TRAIN_FOLDER, x), sample_covid19_file_paths))



for row, file_path in enumerate(sample_covid19_file_paths):

    image = plt.imread(file_path)

    ax[row, 0].imshow(image)

    ax[row, 1].hist(image.ravel(), 256, [0,256])

    ax[row, 0].axis('off')

    if row == 0:

        ax[row, 0].set_title('Images')

        ax[row, 1].set_title('Histograms')

fig.suptitle('Label 2 Virus Category = COVID-19', size=16)

plt.show()
# Plot b/w image histograms of Label type "Normal" patients 

fig, ax = plt.subplots(4, 2, figsize=(20, 20))



other_type_file_paths = train_data[train_data['Label'] == 'Normal']['X_ray_image_name'].values

sample_other_file_paths = other_type_file_paths[:4]

sample_other_file_paths = list(map(lambda x: os.path.join(TRAIN_FOLDER, x), sample_other_file_paths))



for row, file_path in enumerate(sample_other_file_paths):

    image = plt.imread(file_path)

    ax[row, 0].imshow(image)

    ax[row, 1].hist(image.ravel(), 256, [0,256])

    ax[row, 0].axis('off')

    if row == 0:

        ax[row, 0].set_title('Images')

        ax[row, 1].set_title('Histograms')

fig.suptitle('Label = Normal', size=16)

plt.show()
# Generate the final train data from original train data with conditions refered from EDA inference

final_train_data = train_data[(train_data['Label'] == 'Normal') | 

                              ((train_data['Label'] == 'Pnemonia') & (train_data['Label_2_Virus_category'] == 'COVID-19'))]





# Create a target attribute where value = positive if 'Pnemonia + COVID-19' or value = negative if 'Normal'

final_train_data['target'] = ['negative' if holder == 'Normal' else 'positive' for holder in final_train_data['Label']]



final_train_data = shuffle(final_train_data, random_state=1)



final_validation_data = final_train_data.iloc[1000:, :]

final_train_data = final_train_data.iloc[:1000, :]



print(f"Final train data shape : {final_train_data.shape}")

final_train_data.sample(10)
train_image_generator = ImageDataGenerator(

    rescale=1./255,

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=90,

    width_shift_range=0.15,

    height_shift_range=0.15,

    horizontal_flip=True,

    zoom_range=[0.9, 1.25],

    brightness_range=[0.5, 1.5]

)



test_image_generator = ImageDataGenerator(

    rescale=1./255

)



train_generator = train_image_generator.flow_from_dataframe(

    dataframe=final_train_data,

    directory=TRAIN_FOLDER,

    x_col='X_ray_image_name',

    y_col='target',

    target_size=(224, 224),

    batch_size=8,

    seed=2020,

    shuffle=True,

    class_mode='binary'

)



validation_generator = train_image_generator.flow_from_dataframe(

    dataframe=final_validation_data,

    directory=TRAIN_FOLDER,

    x_col='X_ray_image_name',

    y_col='target',

    target_size=(224, 224),

    batch_size=8,

    seed=2020,

    shuffle=True,

    class_mode='binary'

)



test_generator = test_image_generator.flow_from_dataframe(

    dataframe=test_data,

    directory=TEST_FOLDER,

    x_col='X_ray_image_name',

    target_size=(224, 224),

    shuffle=False,

    batch_size=16,

    class_mode=None

)
def scheduler(epoch):

    if epoch < 5:

        return 0.0001

    else:

        print(f"Learning rate reduced to {0.0001 * np.exp(0.5 * (5 - epoch))}")

        return 0.0001 * np.exp(0.5 * (5 - epoch))

    

custom_callback = LearningRateScheduler(scheduler)



METRICS = [

      TruePositives(name='tp'),

      FalsePositives(name='fp'),

      TrueNegatives(name='tn'),

      FalseNegatives(name='fn'), 

      BinaryAccuracy(name='accuracy'),

      Precision(name='precision'),

      Recall(name='recall'),

      AUC(name='auc'),

]
model = Sequential([

    Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu'),

    MaxPooling2D((3, 3)),

    Conv2D(32, (3, 3), activation='relu'),

    MaxPooling2D((3, 3)),

    Conv2D(32, (3, 3), activation='relu'),

    MaxPooling2D((3, 3)),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.4),

    Dense(32, activation='relu'),

    Dropout(0.4),

    Dense(1, activation='sigmoid')

])



model.compile(optimizer=Adam(), loss=binary_crossentropy,

             metrics=METRICS)
history = model.fit_generator(train_generator,

                   validation_data=validation_generator,

                   epochs=20,

                   callbacks=[custom_callback])
model.save('covid19_xray_base_cnn_model.h5')

ACCURACY_LIST.append(['Base CNN Model', history])
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

sns.lineplot(x=np.arange(1, 21), y=history.history.get('loss'), ax=ax[0, 0])

sns.lineplot(x=np.arange(1, 21), y=history.history.get('auc'), ax=ax[0, 1])

sns.lineplot(x=np.arange(1, 21), y=history.history.get('val_loss'), ax=ax[1, 0])

sns.lineplot(x=np.arange(1, 21), y=history.history.get('val_auc'), ax=ax[1, 1])

ax[0, 0].set_title('Training Loss vs Epochs')

ax[0, 1].set_title('Training AUC vs Epochs')

ax[1, 0].set_title('Validation Loss vs Epochs')

ax[1, 1].set_title('Validation AUC vs Epochs')

fig.suptitle('Base CNN model', size=16)

plt.show()
balanced_data = train_data[(train_data['Label'] == 'Normal') | 

                              ((train_data['Label'] == 'Pnemonia') & (train_data['Label_2_Virus_category'] == 'COVID-19'))]



balanced_data['target'] = ['negative' if holder == 'Normal' else 'positive' for holder in balanced_data['Label']]



balanced_data_subset_normal = balanced_data[balanced_data['target'] == 'negative']

balanced_data_subset_covid = balanced_data[balanced_data['target'] == 'positive']

balanced_data_frac_normal = balanced_data_subset_normal.sample(frac=(1/5))



balanced_data_concat = pd.concat([balanced_data_frac_normal, balanced_data_subset_covid], axis=0)

balanced_data_concat = shuffle(balanced_data_concat, random_state=0)

balanced_data_train = balanced_data_concat[:240]

balanced_data_validation = balanced_data_concat[240:]



print(f"Balanced train data shape {balanced_data_train.shape}")

print(f"Balanced validation data shape {balanced_data_validation.shape}")
balanced_train_generator = train_image_generator.flow_from_dataframe(

    dataframe=balanced_data_train,

    directory=TRAIN_FOLDER,

    x_col='X_ray_image_name',

    y_col='target',

    target_size=(224, 224),

    batch_size=64,

    class_mode='binary'

)



balanced_validation_generator = train_image_generator.flow_from_dataframe(

    dataframe=balanced_data_validation,

    directory=TRAIN_FOLDER,

    x_col='X_ray_image_name',

    y_col='target',

    target_size=(224, 224),

    batch_size=64,

    class_mode='binary'

)
METRICS = [

      TruePositives(name='tp'),

      FalsePositives(name='fp'),

      TrueNegatives(name='tn'),

      FalseNegatives(name='fn'), 

      BinaryAccuracy(name='accuracy'),

      Precision(name='precision'),

      Recall(name='recall'),

      AUC(name='auc'),

]



balanced_model = Sequential([

    Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu'),

    MaxPooling2D((3, 3)),

    Conv2D(32, (3, 3), activation='relu'),

    MaxPooling2D((3, 3)),

    Conv2D(32, (3, 3), activation='relu'),

    MaxPooling2D((3, 3)),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.4),

    Dense(32, activation='relu'),

    Dropout(0.4),

    Dense(1, activation='sigmoid')

])



balanced_model.compile(optimizer=Adam(), loss=binary_crossentropy,

             metrics=METRICS)
balanced_model.summary()
balanced_history = balanced_model.fit_generator(balanced_train_generator,

                                               epochs=30,

                                               validation_data=balanced_validation_generator,

                                               callbacks=[custom_callback])
balanced_model.save('covid19_xray_base_cnn_model_balanced.h5')

ACCURACY_LIST.append(['Balanced Base Model', balanced_history])
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

sns.lineplot(x=np.arange(1, 31), y=balanced_history.history.get('loss'), ax=ax[0, 0])

sns.lineplot(x=np.arange(1, 31), y=balanced_history.history.get('auc'), ax=ax[0, 1])

sns.lineplot(x=np.arange(1, 31), y=balanced_history.history.get('val_loss'), ax=ax[1, 0])

sns.lineplot(x=np.arange(1, 31), y=balanced_history.history.get('val_auc'), ax=ax[1, 1])

ax[0, 0].set_title('Training Loss vs Epochs')

ax[0, 1].set_title('Training AUC vs Epochs')

ax[1, 0].set_title('Validation Loss vs Epochs')

ax[1, 1].set_title('Validation AUC vs Epochs')

fig.suptitle('Balanced base CNN model', size=16)

plt.show()
METRICS = [

      TruePositives(name='tp'),

      FalsePositives(name='fp'),

      TrueNegatives(name='tn'),

      FalseNegatives(name='fn'), 

      BinaryAccuracy(name='accuracy'),

      Precision(name='precision'),

      Recall(name='recall'),

      AUC(name='auc'),

]



def output_custom_model(prebuilt_model):

    print(f"Processing {prebuilt_model}")

    prebuilt = prebuilt_model(include_top=False,

                            input_shape=(224, 224, 3),

                            weights='imagenet')

    output = prebuilt.output

    output = GlobalMaxPooling2D()(output)

    output = Dense(128, activation='relu')(output)

    output = Dropout(0.2)(output)

    output = Dense(1, activation='sigmoid')(output)



    model = Model(inputs=prebuilt.input, outputs=output)

    model.compile(optimizer='sgd', loss=binary_crossentropy,

              metrics=METRICS)

    return model
resnet_custom_model = output_custom_model(ResNet50)

resnet_history = resnet_custom_model.fit_generator(train_generator,

                                 epochs=20,

                                 validation_data=validation_generator,

                                 callbacks=[custom_callback])
resnet_custom_model.save('covid19_xray_resnet_50.h5')

ACCURACY_LIST.append(['ResNet 50', resnet_history])
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

sns.lineplot(x=np.arange(1, 21), y=resnet_history.history.get('loss'), ax=ax[0, 0])

sns.lineplot(x=np.arange(1, 21), y=resnet_history.history.get('auc'), ax=ax[0, 1])

sns.lineplot(x=np.arange(1, 21), y=resnet_history.history.get('val_loss'), ax=ax[1, 0])

sns.lineplot(x=np.arange(1, 21), y=resnet_history.history.get('val_auc'), ax=ax[1, 1])

ax[0, 0].set_title('Training Loss vs Epochs')

ax[0, 1].set_title('Training AUC vs Epochs')

ax[1, 0].set_title('Validation Loss vs Epochs')

ax[1, 1].set_title('Validation AUC vs Epochs')

fig.suptitle('ResNet 50 model', size=16)

plt.show()
METRICS = [

      TruePositives(name='tp'),

      FalsePositives(name='fp'),

      TrueNegatives(name='tn'),

      FalseNegatives(name='fn'), 

      BinaryAccuracy(name='accuracy'),

      Precision(name='precision'),

      Recall(name='recall'),

      AUC(name='auc'),

]



efficient_net_custom_model = output_custom_model(EfficientNetB4)

efficient_net_history = efficient_net_custom_model.fit_generator(train_generator,

                                 epochs=20,

                                 validation_data=validation_generator,

                                 callbacks=[custom_callback])
efficient_net_custom_model.save('covid19_xray_efficient_net_B4.h5')

ACCURACY_LIST.append(['EfficientNet B4', efficient_net_history])
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

sns.lineplot(x=np.arange(1, 21), y=efficient_net_history.history.get('loss'), ax=ax[0, 0])

sns.lineplot(x=np.arange(1, 21), y=efficient_net_history.history.get('auc'), ax=ax[0, 1])

sns.lineplot(x=np.arange(1, 21), y=efficient_net_history.history.get('val_loss'), ax=ax[1, 0])

sns.lineplot(x=np.arange(1, 21), y=efficient_net_history.history.get('val_auc'), ax=ax[1, 1])

ax[0, 0].set_title('Training Loss vs Epochs')

ax[0, 1].set_title('Training AUC vs Epochs')

ax[1, 0].set_title('Validation Loss vs Epochs')

ax[1, 1].set_title('Validation AUC vs Epochs')

fig.suptitle('EfficientNet B4 model', size=16)

plt.show()
ACCURACY_LIST = np.array(ACCURACY_LIST)

model_names = ACCURACY_LIST[:, 0]

histories = ACCURACY_LIST[:, 1]



fig, ax = plt.subplots(2, 2, figsize=(20, 20))

sns.barplot(x=model_names, y=list(map(lambda x: x.history.get('auc')[-1], histories)), ax=ax[0, 0], palette='Spectral')

sns.barplot(x=model_names, y=list(map(lambda x: x.history.get('val_auc')[-1], histories)), ax=ax[0, 1], palette='gist_yarg')

sns.barplot(x=model_names, y=list(map(lambda x: x.history.get('accuracy')[-1], histories)), ax=ax[1, 0], palette='rocket')

sns.barplot(x=model_names, y=list(map(lambda x: x.history.get('val_accuracy')[-1], histories)), ax=ax[1, 1], palette='ocean_r')

ax[0, 0].set_title('Model Training AUC scores')

ax[0, 1].set_title('Model Validation AUC scores')

ax[1, 0].set_title('Model Training Accuracies')

ax[1, 1].set_title('Model Validation Accuracies')

fig.suptitle('Model Comparisions')

plt.show()
metric_dataframe = pd.DataFrame({

    'Model Names': model_names,

    'True Positives': list(map(lambda x: x.history.get('tp')[-1], histories)),

    'False Positives': list(map(lambda x: x.history.get('fp')[-1], histories)),

    'True Negatives': list(map(lambda x: x.history.get('tn')[-1], histories)),

    'False Negatives': list(map(lambda x: x.history.get('fn')[-1], histories))

})

fig, ax = plt.subplots(2, 2, figsize=(20, 20))

sns.barplot(x='Model Names', y='True Positives', data=metric_dataframe, ax=ax[0, 0], palette='BrBG')

sns.barplot(x='Model Names', y='False Positives', data=metric_dataframe, ax=ax[0, 1], palette='icefire_r')

sns.barplot(x='Model Names', y='True Negatives', data=metric_dataframe, ax=ax[1, 0], palette='PuBu_r')

sns.barplot(x='Model Names', y='False Negatives', data=metric_dataframe, ax=ax[1, 1], palette='YlOrBr')

ax[0, 0].set_title('True Positives of Models')

ax[0, 1].set_title('False Positives of Models')

ax[1, 0].set_title('True Negatives of Models')

ax[1, 1].set_title('False Negatives of Models')

fig.suptitle('Confusion Matrix comparision of Models', size=16)

plt.show()