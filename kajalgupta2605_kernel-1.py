# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

import os

from keras import regularizers

import tensorflow as tf

from sklearn import preprocessing

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.callbacks import ModelCheckpoint

from keras.callbacks import Callback, EarlyStopping

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

# roc curve and auc score

from sklearn import preprocessing

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



trainData = pd.read_csv("../input/widsdatathon2020/training_v2.csv")

testData = pd.read_csv("../input/widsdatathon2020/unlabeled.csv")



# Any results you write to the current directory are saved as output.
categorical_cols = [c for c in trainData.columns if (trainData[c].dtype != np.number)& (trainData[c].dtype != int) ]

Categorical_df= trainData[categorical_cols]

# for col in categorical_cols:

#     print(col, "*****************")

#     print(Categorical_df[col].value_counts(), Categorical_df[col].unique())
# frequency encoder

for col in ['ethnicity', 'hospital_admit_source', 'icu_admit_source', 'apache_3j_bodysystem', 'apache_2_bodysystem', 'icu_type']:

    trainData[col] = trainData[col].astype('str')

    freq = trainData.groupby(col).size()/len(trainData[col])

    trainData[col] = trainData[col].map(freq)

    

    testData[col] = testData[col].astype('str')

    freq = testData.groupby(col).size()/len(testData[col])

    testData[col] = testData[col].map(freq)



trainData.head()
# ordinal encoder

icu_st_dict={'admit':0,'readmit':1,'transfer':2}



trainData['icu_stay_type'] = trainData['icu_stay_type'].astype('str')

trainData['icu_stay_type'] = trainData['icu_stay_type'].map(icu_st_dict)



testData['icu_stay_type'] = testData['icu_stay_type'].astype('str')

testData['icu_stay_type'] = testData['icu_stay_type'].map(icu_st_dict)
trainData['icu_stay_type'].tail(10)

trainData['icu_type'].tail(10)
# label encoding the data gender

le = LabelEncoder()

for col in ['gender']:

    trainData[col] = trainData[col].astype('str')



    #Fit LabelEncoder

    le.fit(np.unique(trainData[col].unique()))



    #At the end 0 will be used for null values so we start at 1 

    trainData[col] = le.transform(trainData[col])+1

    trainData[col] = trainData[col].replace(np.nan, 0).astype('int')

    

    testData[col] = testData[col].astype('str')



    #Fit LabelEncoder

    le.fit(np.unique(testData[col].unique()))



    #At the end 0 will be used for null values so we start at 1 

    testData[col] = le.transform(testData[col])+1

    testData[col] = testData[col].replace(np.nan, 0).astype('int')



testData.head()
# replace na with mean for following categories



for col in ['age', 'bmi', 'weight', 'height']:

    mean = trainData[col].mean()

    trainData[col] = trainData[col].replace(np.nan, mean).astype('int')

    

    mean = testData[col].mean()

    testData[col] = testData[col].replace(np.nan, mean).astype('int')
x = testData.isnull().sum(axis=0)

x
trainData = trainData.replace(np.nan, 0).astype('int')

testData = testData.replace(np.nan, 0).astype('int')

testData.head(5)
to_drop = ['gender','ethnicity' ,'encounter_id', 'patient_id',  'hospital_death', 'hospital_id']

testDataOld = testData

trainLabel = trainData['hospital_death']

for col in to_drop:

    trainData = trainData.drop(col, axis = 1)

    testData = testData.drop(col, axis = 1)



trainData.head()
cols_with_missing = (col for col in y_test.columns if y_test[col].isnull().any())

for col in cols_with_missing:

    y_test[col + '_was_missing'] = y_test[col].isnull()

    y_test[col + '_was_missing'] = y_test[col].isnull()
x_train, y_test ,x_label, y_label = train_test_split(trainData, trainLabel, test_size=0.3, random_state=1)

std_scale = preprocessing.StandardScaler().fit(x_train)

x_train = std_scale.transform(x_train)

y_test  = std_scale.transform(y_test)

testData = std_scale.transform(testData)

x_train.shape
# from sklearn.linear_model import LinearRegression

# from sklearn.feature_selection import RFE

# from sklearn.feature_selection import SelectKBest, SelectFpr, f_classif



# reg = LinearRegression()

# x_train_new = SelectFpr(f_classif, alpha=0.01).fit_transform(x_train, x_label)

# fit = reg.fit(x_train_new, x_label)

# pred = fit.predict(y_test)

# print(testData.shape, pred.shape)



# auc = roc_auc_score(y_label, pred)

# fpr, tpr, thresholds = roc_curve(y_label, pred)

# plot_roc_curve(fpr, tpr)

# print(auc)

# pred = fit.predict(testData)

# testDataOld["hospital_death"] = pred

# testDataOld[["encounter_id","hospital_death"]].to_csv("submission.csv",index=False)

# testDataOld[["encounter_id","hospital_death"]].head()
def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
checkpoint_callback = ModelCheckpoint("model.h5", monitor='accuracy', save_best_only=True, save_freq=2)

y_test.shape
model = Sequential()

model.add(Dense(64, input_shape=(180,),kernel_initializer='normal', activation='sigmoid', name='fc1'))

model.add(Dense(32, activation='sigmoid',kernel_initializer='normal', name='fc3'))

model.add(Dense(1, name='output'))

optimizer = tf.keras.optimizers.RMSprop(0.0001)
model.compile(optimizer, loss='mse', metrics=['accuracy', 'mse'])
model.fit(x_train, x_label, batch_size=60, epochs=50, callbacks=[checkpoint_callback])
probs = model.predict_proba(y_test).flatten()

auc = roc_auc_score(y_label, probs)

fpr, tpr, thresholds = roc_curve(y_label, probs)

plot_roc_curve(fpr, tpr)

print("AUC-ROC :",auc)

probs
probstest = model.predict_proba(testData)

probstest = probstest[:]

print(probstest)

testDataOld["hospital_death"] = probstest

testDataOld[["encounter_id","hospital_death"]].to_csv("submission3.csv",index=False)

testDataOld[["encounter_id","hospital_death"]].head()

testDataOld[["encounter_id","hospital_death"]].head()
from IPython.display import FileLink

import os

os.chdir(r'/kaggle/working')

FileLink(r'submission3.csv')