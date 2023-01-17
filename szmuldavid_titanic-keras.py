import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.image import imread

from zipfile import ZipFile
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train_test_data = [train, test]

#df_test['Survived']=df_submit[df_submit['PassengerId']==df_test['PassengerId']]['Survived']
#Get Missing Values + Types

train.info()



print('\nRatio Missing Values')

def ratio_missing_values(df):

  ### Indicates  Missing Values

  return df.isnull().sum()/len(df)*100

ratio_missing_values(train)



# Age might have chances to be an important parameter despite being often missing values.

# Needs to be proved with correlation but because there are not a lot of database,

# It is advised to concatenate train + test for the 'Filling Values' Part
sns.countplot(x='Survived',data=train);
plt.figure(figsize=(12,7))

sns.heatmap(train.corr(),annot=True)

#plt.ylim(10, 0);
plt.figure(num=0)

sns.countplot(x='Embarked',data=train);

# Majority embarked at Southampton + Only 0.22 of null values
'''

Majority of passengers embarked at Southampton so fill the values on this town by default

'''

def fill_embarked(Embarked,default_embark):

    if type(Embarked)==str: return Embarked 

    elif np.isnan(Embarked): return default_embark



    assert(False)#Cannot be float and not nan
for i,dataset in enumerate(train_test_data):

  train_test_data[i]['Embarked']=dataset.apply(lambda x: fill_embarked(x['Embarked'], 'S'), axis=1)
train_test_data[0]['Embarked'].unique()
plt.figure(num=0)

sns.scatterplot(x='Age',y='Survived',data=train,alpha=0.1) 

plt.figure(num=1)

sns.displot(train, x='Age', hue="Survived",multiple="dodge")

#Age seems important input for prediction for very young + very old

#But 20% of missing values is to high to make an estimation just by the mean or median 
# Erase Age seems to give better performances than Fill with mean

flag_possibility = 0



if flag_possibility==0: # Erase Age

  for i,dataset in enumerate(train_test_data):

    train_test_data[i]=dataset.drop('Age', axis=1) 



elif flag_possibility==1: # Mean Age

  for i,dataset in enumerate(train_test_data):

    train_test_data[i]['Age'].fillna(pd.concat(train_test_data)['Age'].mean(), inplace=True)



else: # Use another NN that estimate AGE (TODO)

  pass
#pd.concat(train_test_data)['Age']
# UNUSABLE INPUTS

useless_columns=['Name', 'Ticket', 'Cabin']

for i,dataset in enumerate(train_test_data):

  # No relation between name + Too many names for a simple NN

  train_test_data[i]=dataset.drop(useless_columns, axis=1) 
train_test_data[0]
# Categories to dummy

list(train_test_data[0].select_dtypes(['object']).columns)
def change_obj_2_dummy(df):

  objects = df.select_dtypes(['object']).columns

  dummies = pd.get_dummies(df[objects],drop_first=True)

  df = df.drop(objects,axis=1)

  df = pd.concat([df,dummies],axis=1)

  return df



for i,dataset in enumerate(train_test_data):

  # No relation between name + Too many names for a simple NN

  train_test_data[i]=change_obj_2_dummy(dataset)
Id_train = train_test_data[0]['PassengerId']

Id_valid = train_test_data[1]['PassengerId']



for i,dataset in enumerate(train_test_data):

  # No relation between name + Too many names for a simple NN

  train_test_data[i]=dataset.drop('PassengerId',axis=1)



# Initiate again Train/Validation

train, valid = train_test_data

label_train = train['Survived']

train = train.drop('Survived',axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, label_train, test_size=0.3, random_state=101)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_valid = scaler.transform(valid)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import load_model



### Start with a litlle NN because of the lack of data --> High risk of overtraining

model = Sequential()

# input layer

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))

# hidden layer

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))

# hidden layer

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.2))

# output layer

model.add(Dense(units=1,activation='sigmoid'))



# Compilation of model

model.compile(loss='binary_crossentropy', optimizer='adam')



# Callbacks

early_stop = EarlyStopping(monitor='val_loss',patience=10)
model.fit(x=X_train, 

          y=y_train, 

          epochs=250,

          batch_size=16,

          validation_data=(X_test, y_test), 

          callbacks=[early_stop]

          )



# TODO: Add a condition only if val_loss or val_acc is better

model.save('titanic.h5')
losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot();
from sklearn.metrics import classification_report,confusion_matrix



predictions = (model.predict(X_test) > 0.5).astype('int32')

print(classification_report(y_test,predictions))

print('\n')

confusion_matrix(y_test,predictions)
predict_submit = (model.predict(X_valid) > 0.5).astype('int32')



df_submit = pd.DataFrame(Id_valid) 

df_submit['Survived']=predict_submit

df_submit.to_csv('submission.csv',index=False)