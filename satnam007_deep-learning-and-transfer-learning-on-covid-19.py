import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import pandas as pd

import numpy as np



cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 





full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')

full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])





day_wise = pd.read_csv('../input/corona-virus-report/day_wise.csv')

day_wise['Date'] = pd.to_datetime(day_wise['Date'])





country_wise = pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')

country_wise = country_wise.replace('', np.nan).fillna(0)

# country_wise.head()
worldometer_data = pd.read_csv('../input/worldometer-stats/wworldometer.csv')

worldometer_data = worldometer_data.replace('', np.nan).fillna(0)
temp = day_wise[['Date','Deaths', 'Recovered', 'Active']].tail(1)

temp = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])

fig = px.treemap(temp, path=["variable"], values="value", height=225, 

                 color_discrete_sequence=[act, rec, dth])

fig.data[0].textinfo = 'label+text+value'
def plot_map(df, col, pal):

    df = df[df[col]>0]

    fig = px.choropleth(df, locations="Country/Region", locationmode='country names', 

                  color=col, hover_name="Country/Region", 

                  title=col, hover_data=[col], color_continuous_scale=pal)

    fig.show()

plot_map(country_wise, 'Confirmed', 'matter')
from IPython.display import YouTubeVideo

YouTubeVideo('i0ZabxXmH4Y', width=700, height=450)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()

import numpy as np # linear algebra

 # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import  Input, Conv2D, MaxPooling2D,GlobalMaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Activation, MaxPool2D, AvgPool2D, Dropout, Conv1D, MaxPooling1D

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.keras.applications import DenseNet121, VGG19, ResNet50

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from IPython.display import display, Image

import matplotlib.pyplot as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

import warnings

warnings.filterwarnings("ignore")



from sklearn.utils import shuffle

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')

valid_df = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_dataset_Summary.csv')



print('The training dataset has rows : ', format(train_df.shape[0]))

print('The training dataset has cols : ', format(train_df.shape[1]))
train_df.head(5)
missing_vals = train_df.isnull().sum()

missing_vals.plot(kind = 'bar')
train_df.dropna(how = 'all')

train_df.isnull().sum()
train_data = train_df[train_df['Dataset_type'] == 'TRAIN']

test_data = train_df[train_df['Dataset_type'] == 'TEST']

assert train_data.shape[0] + test_data.shape[0] == train_df.shape[0]

print(f"Shape of train data : {train_data.shape}")

print(f"Shape of test data : {test_data.shape}")

test_data.sample(10)
train_fill = train_data.fillna('unknown')

test_fill = test_data.fillna('unknown')

display(train_fill.head(5))
# Count plot for 3 attributes with unknown variable addition

targets = ['Label', 'Label_2_Virus_category', 'Label_1_Virus_category']

fig, ax = plt.subplots(2, 2, figsize=(20, 10))

sns.countplot(x=targets[0], data=train_fill, ax=ax[0, 0])

sns.countplot(x=targets[1], data=train_fill, ax=ax[0, 1])

sns.countplot(x=targets[2], data=train_fill, ax=ax[1, 0])

plt.show()
test_img_dir = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'

train_img_dir = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'



assert os.path.isdir(test_img_dir) == True

assert os.path.isdir(train_img_dir) == True



sample_train_images = list(os.walk(train_img_dir))[0][2][:8]

sample_train_images = list(map(lambda x: os.path.join(train_img_dir, x), sample_train_images))



sample_test_images = list(os.walk(test_img_dir))[0][2][:8]

sample_test_images = list(map(lambda x: os.path.join(test_img_dir, x), sample_test_images))
from PIL import Image

plt.figure(figsize = (17,17))

for iterator, filename in enumerate(sample_train_images):

    image = Image.open(filename)

    plt.subplot(4,2,iterator+1)

    plt.imshow(image)



plt.tight_layout()
plt.figure(figsize = (17,17))

for iterator, filename in enumerate(sample_test_images):

    image = Image.open(filename)

    plt.subplot(4,2,iterator+1)

    plt.imshow(image)



plt.tight_layout()
fig, ax = plt.subplots(4, 2, figsize=(17, 17))





covid_path = train_data[train_data['Label_2_Virus_category']=='COVID-19']['X_ray_image_name'].values



sample_covid_path = covid_path[:4]

sample_covid_path = list(map(lambda x: os.path.join(train_img_dir, x), sample_covid_path))



for row, file in enumerate(sample_covid_path):

    image = plt.imread(file)

    ax[row, 0].imshow(image)

    ax[row, 1].hist(image.ravel(), 256, [0,256])

    ax[row, 0].axis('off')

    if row == 0:

        ax[row, 0].set_title('Images')

        ax[row, 1].set_title('Histograms')

fig.suptitle('Label 2 Virus Category = COVID-19', size=16)

plt.show()
fig, ax = plt.subplots(4, 2, figsize=(17, 17))





normal_path = train_data[train_data['Label']=='Normal']['X_ray_image_name'].values



sample_normal_path = normal_path[:4]

sample_normal_path = list(map(lambda x: os.path.join(train_img_dir, x), sample_normal_path))



for row, file in enumerate(sample_normal_path):

    image = plt.imread(file)

    ax[row, 0].imshow(image)

    ax[row, 1].hist(image.ravel(), 256, [0,256])

    ax[row, 0].axis('off')

    if row == 0:

        ax[row, 0].set_title('Images')

        ax[row, 1].set_title('Histograms')

fig.suptitle('Label = NORMAL', size=16)

plt.show()
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

    rotation_range=90,

    width_shift_range=0.15,

    height_shift_range=0.15,

    horizontal_flip=True,

    zoom_range=0.5,

)



test_image_generator = ImageDataGenerator(

    rescale=1./255

)



train_generator = train_image_generator.flow_from_dataframe(

    dataframe=final_train_data,

    directory=train_img_dir,

    x_col='X_ray_image_name',

    y_col='target',

    target_size=(224, 224),

    batch_size=16,

    seed=2020,

    shuffle=True,

    class_mode='binary'

)



validation_generator = train_image_generator.flow_from_dataframe(

    dataframe=final_validation_data,

    directory=train_img_dir,

    x_col='X_ray_image_name',

    y_col='target',

    target_size=(224, 224),

    batch_size=16,

    seed=2020,

    shuffle=True,

    class_mode='binary'

)



test_generator = test_image_generator.flow_from_dataframe(

    dataframe=test_data,

    directory=test_img_dir,

    x_col='X_ray_image_name',

    target_size=(224, 224),

    shuffle=False,

    batch_size=16,

    class_mode=None)
IMG_W = 224

IMG_H = 224

CHANNELS = 3



INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)

NB_CLASSES = 2

EPOCHS = 30

BATCH_SIZE = 6
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Conv2D(64,(3,3)))

model.add(Activation("relu"))

model.add(Conv2D(250,(3,3)))

model.add(Activation("relu"))

  

model.add(Conv2D(128,(3,3)))

model.add(Activation("relu"))

model.add(AvgPool2D(2,2))

model.add(Conv2D(64,(3,3)))

model.add(Activation("relu"))

model.add(AvgPool2D(2,2))



model.add(Conv2D(256,(2,2)))

model.add(Activation("relu"))

model.add(MaxPool2D(2,2))

    

model.add(Flatten())

model.add(Dense(32))

model.add(Dropout(0.25))

model.add(Dense(1))

model.add(Activation("sigmoid"))
model.summary()
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.9325):

            print("\nReached 94.25% accuracy so cancelling training!")

            self.model.stop_training = True

callbacks = myCallback()
#callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(train_generator,

                              steps_per_epoch = len(train_generator),

                              validation_data=validation_generator,

                              epochs=20,

                              validation_steps=len(validation_generator),

                              callbacks = [callbacks]

                                     )
plt.figure(figsize=(17,17))

plt.subplot(2, 2, 1)

plt.plot(history.history['loss'], label='Loss')

plt.legend()

plt.title('Training - Loss Function')



plt.subplot(2, 2, 2)

plt.plot(history.history['accuracy'], label='Accuracy')

plt.legend()

plt.title('Train - Accuracy')



plt.legend()

plt.title('Metrics estimations')

dense_model = Sequential()

dense_model.add(DenseNet121(include_top=False, pooling = 'avg', weights='imagenet',input_shape=(224, 224, 3), classes=2))

dense_model.add(Dense(512, activation='relu'))

dense_model.add(Dense(128, activation='relu'))

dense_model.add(Dense(64, activation='relu'))

dense_model.add(Dense(1, activation='sigmoid'))

dense_model.layers[0].trainable = False

dense_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
dense_history = dense_model.fit_generator(train_generator,

                                          steps_per_epoch = len(train_generator),

                                          validation_data=validation_generator,

                                          epochs=20,

                                          validation_steps=len(validation_generator),

                                          callbacks = [callbacks]

                                             )
plt.figure(figsize=(17,17))



plt.subplot(2, 2, 1)

plt.plot(dense_history.history['loss'], label='Loss')

plt.plot(dense_history.history['val_loss'], label='Validation Loss')

plt.legend()

plt.title('Training - Loss Function')



plt.subplot(2, 2, 2)

plt.plot(dense_history.history['accuracy'], label='Accuracy')

plt.plot(dense_history.history['val_accuracy'], label='Validation Accuracy')



plt.legend()

plt.title('Train - Accuracy')
mob_model = Sequential()

mob_model.add(tf.keras.applications.MobileNetV2(include_top=False, pooling = 'avg', weights='imagenet',input_shape=(224, 224, 3), classes=2))

mob_model.add(Dense(32, activation='relu'))

mob_model.add(Dense(1, activation='sigmoid'))

mob_model.layers[0].trainable = False

mob_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mob_history = mob_model.fit_generator(train_generator,

                              steps_per_epoch = len(train_generator),

                              validation_data=validation_generator,

                              epochs=20,

                              validation_steps=len(validation_generator),

                              callbacks = [callbacks]

                                     )
plt.figure(figsize=(17,17))



plt.subplot(2, 2, 1)

plt.plot(mob_history.history['loss'], label='Loss')

plt.plot(mob_history.history['loss'], label='Validation Loss')



plt.legend()

plt.title('Training - Loss Function')



plt.subplot(2, 2, 2)

plt.plot(mob_history.history['accuracy'], label='Accuracy')

plt.plot(mob_history.history['val_accuracy'], label='Validation Accuracy')





plt.legend()

plt.title('Train - Accuracy')
model.predict(validation_generator)
dense_model.predict(validation_generator)
mob_model.predict(validation_generator)
label = validation_generator.classes

print('Cases summary of the models : \n{}'.format(label))
pred= model.predict(validation_generator)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (validation_generator.class_indices)

labels2 = dict((v,k) for k,v in labels.items())

predictions = [labels2[k] for k in predicted_class_indices]
print('CNN Model Predictions : \n{}'.format(predictions))
from sklearn.metrics import confusion_matrix, classification_report



cf_matrix = confusion_matrix(predicted_class_indices,label)

cf_matrix
exp_series = pd.Series(label)

pred_series = pd.Series(predicted_class_indices)

pd.crosstab(exp_series, pred_series, rownames=['Actual'], colnames=['Predicted'],margins=True)
import seaborn as sns



matrix_index = ["Normal", "Covid"]



preds = model.predict(validation_generator)

classpreds = np.argmax(preds, axis=1) # predicted classes 

#y_testclass = np.argmax(valida, axis=1) # true classes



cm = confusion_matrix(predicted_class_indices,label)

print(classification_report(predicted_class_indices,label, target_names=matrix_index))



# Get percentage value for each element of the matrix

cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)





# Display confusion matrix 

df_cm = pd.DataFrame(cm, index = matrix_index, columns = matrix_index)

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

fig, ax = plt.subplots(figsize=(10,7))

sns.heatmap(df_cm, annot=annot, fmt='')
pred= dense_model.predict(validation_generator)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (validation_generator.class_indices)

labels2 = dict((v,k) for k,v in labels.items())

predictions = [labels2[k] for k in predicted_class_indices]
print('DenseNet Model Predictions : \n{}'.format(predictions))
cf_matrix = confusion_matrix(predicted_class_indices,label)

cf_matrix
exp_series = pd.Series(label)

pred_series = pd.Series(predicted_class_indices)

pd.crosstab(exp_series, pred_series, rownames=['Actual'], colnames=['Predicted'],margins=True)
matrix_index = ["Normal", "Covid"]



preds = dense_model.predict(validation_generator)

classpreds = np.argmax(preds, axis=1) # predicted classes 

#y_testclass = np.argmax(valida, axis=1) # true classes



cm = confusion_matrix(predicted_class_indices,label)

print(classification_report(predicted_class_indices,label, target_names=matrix_index))



# Get percentage value for each element of the matrix

cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)





# Display confusion matrix 

df_cm = pd.DataFrame(cm, index = matrix_index, columns = matrix_index)

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

fig, ax = plt.subplots(figsize=(10,7))

sns.heatmap(df_cm, annot=annot, fmt='')
pred= mob_model.predict(validation_generator)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (validation_generator.class_indices)

labels2 = dict((v,k) for k,v in labels.items())

predictions = [labels2[k] for k in predicted_class_indices]
print('MobileNet Model Predictions : \n{}'.format(predictions))
cf_matrix = confusion_matrix(predicted_class_indices,label)

cf_matrix
exp_series = pd.Series(label)

pred_series = pd.Series(predicted_class_indices)

pd.crosstab(exp_series, pred_series, rownames=['Actual'], colnames=['Predicted'],margins=True)
matrix_index = ["Normal", "Covid" ]



preds = mob_model.predict(validation_generator)

classpreds = np.argmax(preds, axis=1) # predicted classes 

#y_testclass = np.argmax(valida, axis=1) # true classes



cm = confusion_matrix(predicted_class_indices,label)

print(classification_report(predicted_class_indices,label, target_names=matrix_index))



# Get percentage value for each element of the matrix

cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)





# Display confusion matrix 

df_cm = pd.DataFrame(cm, index = matrix_index, columns = matrix_index)

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

fig, ax = plt.subplots(figsize=(10,7))

sns.heatmap(df_cm, annot=annot, fmt='')