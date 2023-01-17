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
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('/kaggle/input/age-gender-and-ethnicity-face-data-csv/age_gender.csv')
data.head()
data.shape
data.describe()
data.isnull().describe()
# Actually images are in string format, let's transform it in more useful type of data.

data['pixels'] = data['pixels'].map(lambda x: np.array(x.split(' '), dtype=np.float32).reshape(48, 48))
## normalizing pixels data
data['pixels'] = data['pixels'].apply(lambda x: x/255)

## calculating distributions
age_dist = data['age'].value_counts()
ethnicity_dist = data['ethnicity'].value_counts()
gender_dist = data['gender'].value_counts().rename(index={0:'Male',1:'Female'})

def ditribution_plot(x,y,name):
    fig = go.Figure([
        go.Bar(x=x, y=y)
    ])

    fig.update_layout(title_text=name)
    fig.show()
import plotly.graph_objects as go
import plotly.express as px
ditribution_plot(x=age_dist.index, y=age_dist.values, name='Age Distribution')
ditribution_plot(x=ethnicity_dist.index, y=ethnicity_dist.values, name='Ethnicity Distribution')
ditribution_plot(x=gender_dist.index, y=gender_dist.values, name='Gender Distribution')
# Plot some pictures
fig, axes = plt.subplots(1, 5, figsize=(20, 10))

for i in range(5):
    random_face = np.random.choice(len(data))
    
    age = data['age'][random_face]
    ethnicity = data['ethnicity'][random_face]
    gender = data['gender'][random_face]
    
    axes[i].set_title('Age: {0}, Ethnicity: {1}, Sex: {2}'.format(age, ethnicity, gender))
    axes[i].imshow(data['pixels'][random_face])
    axes[i].imshow(data['pixels'][random_face])
    axes[i].axis('off')
X = np.array(data['pixels'].tolist())

## Converting pixels from 1D to 3D
X = X.reshape(X.shape[0],48,48,1)
# Normalise images
if np.max(X) > 1: 
    X = X/255
# Set some useful variables
input_shape = X.shape[1:] 

epochs = 20
batch_size = 64
random_seeds = 42
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,InputLayer, Dropout, BatchNormalization, Flatten, Dense, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# split the data into train ad test
y = data['age'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=random_seeds)
AgeModel = Sequential()

AgeModel.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
AgeModel.add(MaxPooling2D(pool_size=(2,2)))
AgeModel.add(BatchNormalization())

AgeModel.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
AgeModel.add(MaxPooling2D(pool_size=(2,2)))
AgeModel.add(Dropout(0.2))
AgeModel.add(BatchNormalization())

AgeModel.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
AgeModel.add(MaxPooling2D(pool_size=(2,2)))
AgeModel.add(Dropout(0.5))
AgeModel.add(BatchNormalization())

AgeModel.add(Flatten())
AgeModel.add(Dense(128, activation='relu'))
AgeModel.add(Dropout(0.4))
AgeModel.add(Dense(1))

AgeModel.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse'])


# Callbacks for age model
callbacks = [tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss', mode='min'), 
             tf.keras.callbacks.ReduceLROnPlateau(patience=2, verbose=1)]

AgeModel.summary()
history = AgeModel.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size, callbacks=callbacks)
pd.DataFrame(history.history).plot()
valid_score = AgeModel.evaluate(X_test, y_test, verbose=1)
# Make predictions 
y_pred = AgeModel.predict(X_test)

# split the data into train ad test
y = data['gender'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=random_seeds)

gender_model = Sequential([
    InputLayer(input_shape=(48,48,1)),
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(rate=0.5),
    Dense(1, activation='sigmoid')
])

gender_model.compile(optimizer='sgd',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


## Stop training when validation loss reach 0.2700
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss')<0.2000):
            print("\nReached 0.2000 val_loss so cancelling training!")
            self.gender_model.stop_training = True
        
callback = myCallback()

gender_model.summary()
history = gender_model.fit(
    X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size, callbacks=[callback])

pd.DataFrame(history.history).plot();
valid_score = gender_model.evaluate(X_test, y_test, verbose=1)
# Make predictions 
y_pred = gender_model.predict(X_test)
