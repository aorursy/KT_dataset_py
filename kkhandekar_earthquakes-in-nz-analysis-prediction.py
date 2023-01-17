# Generic Libraries

import numpy as np

import pandas as pd



# Visualisation Libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

import warnings

from matplotlib import cm

from mpl_toolkits.basemap import Basemap

import folium

from folium.plugins import CirclePattern, FastMarkerCluster



#Data Formatting

pd.plotting.register_matplotlib_converters()

%matplotlib inline

plt.style.use('seaborn-whitegrid')

pd.set_option('display.max_columns', 500)

warnings.filterwarnings("ignore")

#pd.options.display.float_format = '{:.2f}'.format



#Garbage Collector

import gc



#Date-Time Libraries

import datetime

import time



#SK Learn Libraries

from sklearn import model_selection

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn import metrics



#Tabulate Library

from prettytable import PrettyTable

#Loading Data

url = '../input/earthquakes-data-nz/earthquakes_NZ.csv'

data = pd.read_csv(url, header='infer',parse_dates=True)

data.shape
#Check for missing values

data.isna().sum()
#Checking the data-types for each column

data.info()
#Renaming Columns

data = data.rename(columns={"origintime": "Time", "longitude": "Long", " latitude": "Lat", " depth": "Depth", " magnitude": "Magnitude"})
#Formating Column Data

data[['Depth']] = data[['Depth']].applymap("{0:.2f}".format)

data[['Magnitude']] = data[['Magnitude']].applymap("{0:.1f}".format)

#Converting Depth & Magnitude columns to Float

for col in ['Depth', 'Magnitude']:

    data[col] = data[col].astype('float')



#Converting Time column to datetime

data['Time'] =  pd.to_datetime(data['Time'], format='%Y-%m-%d%H:%M:%S.%f')
#function to categorize Magnitude

def desc(mag):

    if 0 <= mag <= 2.0:

        return 'Micro'

    elif 2.0 <= mag <= 3.9:

        return 'Minor'

    elif 4.0 <= mag <= 4.9:

        return 'Light'

    elif 5.0 <= mag <= 5.9:

        return 'Moderate'

    elif 6.0 <= mag <= 6.9:

        return 'Strong'

    elif 7.0 <= mag <= 7.9:

        return 'Major'

    elif 8.0 <= mag <= 9.9:

        return 'Great'

    else:

        return 'Epic'

    

#Applying the function to the Magnitude Column

data['Desc'] = data['Magnitude'].apply(lambda mag: desc(mag))
#function to convert UTC time to Unix Time

def ConvertTime(UTCtime):

    dt = datetime.datetime.strptime(UTCtime, '%Y-%m-%d %H:%M:%S.%f')

    ut = time.mktime(dt.timetuple())

    return ut



#Converting to string type as the 'time' only accepts str

data['Time'] = data['Time'].astype('str')  



#Applying the function to the Magnitude Column

data['Time'] = data['Time'].apply(ConvertTime)

    
data.head()
data_backup = data.copy()
#freeing Memory

gc.collect()
#Creating a Count Plot

sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(8,8))

ax = sns.countplot(x="Desc", data=data, palette="Blues_d")



plt.title('Earthquake Count')

plt.ylabel('Count')

plt.xlabel('Earthquake Magnitude')



totals = []



for i in ax.patches:

    totals.append(i.get_height())



total = sum(totals)



for i in ax.patches:

   # ax.text(i.get_x()+0.25, i.get_height()+100,str(round((i.get_height()/total)*100, 1))+'%', fontsize=9,color='black')

   ax.text(i.get_x()+0.25, i.get_height()+100,str(i.get_height()), fontsize=9,color='black')





#Creating a plot of Magnitude vs Depth

plt.figure(figsize=(10,8))

sns.boxenplot(x='Desc', y='Depth', data=data, scale="linear")

plt.show();
#Creating a seperate dataset for observing correlation

df = pd.DataFrame(data, columns=['Time','Long','Lat', 'Depth', 'Magnitude'])
#Visualizing the Correlation

corr = df.corr(method='pearson')

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df.columns)

ax.set_yticklabels(df.columns)

plt.show()
#Defining the feature & target

feature_col = ['Time','Long','Lat']

target_col = ['Depth','Magnitude']



#Applying to the current dataset

X = df[feature_col]

y = df[target_col]

size = 0.1

state = 0



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=size, random_state=state)
#Tabulate the Dataset & its size

t = PrettyTable(['Dataset','Size'])

t.add_row(['X_train', X_train.size])

t.add_row(['X_test', X_test.size])

t.add_row(['y_train', y_train.size])

t.add_row(['y_test', y_test.size])

print(t)
#Build the model

rfr = RandomForestRegressor(n_estimators=500, criterion='mse', min_samples_split = 4, verbose=0, random_state=0)



#Train the model

rfr.fit(X_train, y_train)
#Making a Prediction using the testing-data

y_pred = rfr.predict(X_test)



#Finding the accuracy & precision of the model

print("Simple Model Accuracy : ",'{:.1%}'.format(rfr.score(X_test, y_test)))



#garbage collection

gc.collect()
#TensorFlow & Keras Libraries

import tensorflow as tf    

import keras

from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.metrics import categorical_accuracy

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.wrappers.scikit_learn import KerasClassifier



from sklearn.model_selection import GridSearchCV

#Building the neural model

def build_model():

    model = Sequential()

    model.add(Dense(16, activation='relu', input_shape=(3,)))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    

    model.compile(loss='squared_hinge', metrics=['accuracy'],optimizer='adam')

    return model
#Instantiate the Model

model = build_model()



#Model Architecture Summary

model.summary()
#Train the Model

model.fit(X_train, y_train, batch_size=10, epochs=20, verbose=1, validation_data=(X_test, y_test))
#Evaluating the Model

[test_loss, test_acc] = model.evaluate(X_test, y_test)

print("Model Evaluation Results on Test Data : Loss = {:.1%}, Accuracy = {:.1%}".format(test_loss, test_acc))