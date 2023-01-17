import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Ridge

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

import keras
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

import keras
train_df = pd.read_csv('../input/cs-challenge/training_set.csv')

train_df.columns
#train_df.info()

#test_df.info()

train_df.describe()

#train_df.tail()
def fill(train):

    train['MAC_CODE']=train['MAC_CODE'].map({'WT1':1,'WT2':2,'WT3':3,'WT4':4,})

    train1=train[train.MAC_CODE.eq(1)]

    train2=train[train.MAC_CODE.eq(2)]

    train3=train[train.MAC_CODE.eq(3)]

    train4=train[train.MAC_CODE.eq(4)]

    train1=train1.sort_values('Date_time')

    train2=train2.sort_values('Date_time')

    train3=train3.sort_values('Date_time')

    train4=train4.sort_values('Date_time')

    train1=train1.fillna(method='bfill')

    train2=train2.fillna(method='bfill')

    train3=train3.fillna(method='bfill')

    train4=train4.fillna(method='bfill')

    train1=train1.fillna(method='ffill')

    train2=train2.fillna(method='ffill')

    train3=train3.fillna(method='ffill')

    train4=train4.fillna(method='ffill')

    trainconcat=pd.concat([train1,train2,train3,train4])

    return trainconcat

train_df=fill(train_df)
def dropmaxminstd(train):

    traind=train.drop(columns=['Pitch_angle_min','Pitch_angle_max','Pitch_angle_std'])

    traind=traind.drop(columns=['Hub_temperature_min','Hub_temperature_max','Hub_temperature_std'])

    traind=traind.drop(columns=['Generator_converter_speed_min','Generator_converter_speed_max','Generator_converter_speed_std'])

    traind=traind.drop(columns=['Generator_speed_min','Generator_speed_max','Generator_speed_std'])

    traind=traind.drop(columns=['Generator_bearing_1_temperature_min','Generator_bearing_1_temperature_max','Generator_bearing_1_temperature_std'])

    traind=traind.drop(columns=['Generator_bearing_2_temperature_min','Generator_bearing_2_temperature_max','Generator_bearing_2_temperature_std'])

    traind=traind.drop(columns=['Generator_stator_temperature_min','Generator_stator_temperature_max','Generator_stator_temperature_std'])

    traind=traind.drop(columns=['Gearbox_bearing_1_temperature_min','Gearbox_bearing_1_temperature_max','Gearbox_bearing_1_temperature_std'])

    traind=traind.drop(columns=['Gearbox_bearing_2_temperature_min','Gearbox_bearing_2_temperature_max','Gearbox_bearing_2_temperature_std'])

    traind=traind.drop(columns=['Gearbox_inlet_temperature_min','Gearbox_inlet_temperature_max','Gearbox_inlet_temperature_std'])

    traind=traind.drop(columns=['Gearbox_oil_sump_temperature_min','Gearbox_oil_sump_temperature_max','Gearbox_oil_sump_temperature_std'])

    traind=traind.drop(columns=['Nacelle_angle_min','Nacelle_angle_max','Nacelle_angle_std'])

    traind=traind.drop(columns=['Nacelle_temperature_min','Nacelle_temperature_max','Nacelle_temperature_std'])

    traind=traind.drop(columns=['Outdoor_temperature_min','Outdoor_temperature_max','Outdoor_temperature_std'])

    traind=traind.drop(columns=['Grid_frequency_min','Grid_frequency_max','Grid_frequency_std'])

    traind=traind.drop(columns=['Grid_voltage_min','Grid_voltage_max','Grid_voltage_std'])

    traind=traind.drop(columns=['Rotor_speed_min','Rotor_speed_max','Rotor_speed_std'])

    traind=traind.drop(columns=['Rotor_bearing_temperature_min','Rotor_bearing_temperature_max','Rotor_bearing_temperature_std'])

    return traind

train_df=dropmaxminstd(train_df)
plt.figure(figsize=(12,10))

cor=abs(train_df.corr( method = "pearson"))

sns.heatmap(cor)

plt.show()

cor_target=cor['TARGET']

relevant_features = cor_target[cor_target>0.2]

relevant_features
relevant_features.index
train_IDs=train_df['ID']

train_clean = train_df[relevant_features.index]

#MAC_CODEs=train['MAC_CODE']

#train_clean['MAC_CODE']=MAC_CODEs

#train_clean['ID']=IDs

Y_clean = train_clean['TARGET']

X_clean = train_clean.drop('TARGET', axis=1)

X_clean = StandardScaler().fit_transform(X_clean)

X_train_clean, X_test_clean, Y_train_clean, Y_test_clean = train_test_split(X_clean, Y_clean, test_size=0.20, random_state=404)
Y = train_df['TARGET']

X = train_df.drop('TARGET', axis=1)

X = StandardScaler().fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=404)
test_df = pd.read_csv('../input/cs-challenge/test_set.csv')

test_df=fill(test_df)

test_df=dropmaxminstd(test_df)

test_clean=test_df[['Pitch_angle','Generator_converter_speed','Generator_speed','Generator_stator_temperature','Gearbox_bearing_1_temperature','Gearbox_bearing_2_temperature','Gearbox_oil_sump_temperature','Nacelle_temperature','Rotor_speed','Rotor_bearing_temperature']] 

# à compléter avec  l'out des relevant features

test_IDs=test_df['ID']

test_df.drop(columns='ID')

X_real_world = StandardScaler().fit_transform(test_df)

X_real_world_clean = StandardScaler().fit_transform(test_clean)
pca=PCA(n_components = 15).fit(X)
X_pca=pca.transform(X_train)

X_test_pca=pca.transform(X_test)

X_real_world_pca=pca.transform(X_real_world)
regpca = LinearRegression().fit(X_pca, Y_train)

prediction = regpca.predict(X_real_world_pca)

for i in range(len(prediction)) :

    if prediction[i]<0:

        prediction[i]=0

results = pd.DataFrame()

results['ID'] = test_IDs

results['TARGET'] = prediction

results.to_csv('resultsregpca.csv', index=False)



results
Y_pred = regpca.predict(X_test_pca)

mean_absolute_error(Y_test, Y_pred)
reg = LinearRegression().fit(X_train, Y_train)

reg_clean = LinearRegression().fit(X_train_clean, Y_train_clean)
prediction = reg.predict(X_real_world)

for i in range(len(prediction)) :

    if prediction[i]<0:

        prediction[i]=0

results = pd.DataFrame()

results['ID'] = test_IDs

results['TARGET'] = prediction

results.to_csv('resultsreg.csv', index=False)



prediction_clean = reg_clean.predict(X_real_world_clean)

for i in range(len(prediction_clean)) :

    if prediction_clean[i]<0:

        prediction_clean[i]=0

results_clean = pd.DataFrame()

results_clean['ID'] = test_IDs

results_clean['TARGET'] = prediction_clean

results_clean.to_csv('results_clean.csv', index=False)

results
results_clean
model = keras.models.Sequential()

model.add(keras.layers.BatchNormalization(input_shape=(15,)))

model.add(keras.layers.Dense(5, input_dim=15, activation='softmax'))

model.add(keras.layers.Dense(5, input_dim=5, activation='sigmoid'))

model.add(keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X_pca, Y_train, batch_size=32, epochs=15, verbose=1, validation_data=(X_test_pca,Y_test)) 
Y_pred = model.predict(X_test_pca)

Y_pred

mean_absolute_error(Y_test, Y_pred)
Y_pred
prediction1 = model.predict(X_real_world_pca)

for i in range(len(prediction)) :

    if prediction[i]<0:

        prediction[i]=0

results1 = pd.DataFrame()

results['ID'] = test_IDs

results['TARGET'] = prediction

results.to_csv('results1.csv', index=False)

results