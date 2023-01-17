import numpy as np

import pandas as pd

from imblearn.over_sampling import SMOTE



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# Importing the dataset

dataset = pd.read_csv('../input/application_train.csv')

#dataset=dataset.dropna()

#dataset.replace(r'\s+',np.nan,regex=True)

#dataset.head()



#dataset.shape

#dataset.dtypes

#dataset.select_dtypes(include=['object']).copy()

dataset
dataset=dataset.dropna()
dataset
dataset.replace(r'\s+',np.nan,regex=True)
dataset
dataset.head()
dataset.shape
dataset.dtypes
dataset.select_dtypes(include=['object']).copy()

from sklearn.preprocessing import LabelEncoder

number=LabelEncoder()

dataset['NAME_INCOME_TYPE']=number.fit_transform(dataset['NAME_INCOME_TYPE'].astype('str'))

#dataset['NAME_FAMILY_STATUS']=number.fit_transform(dataset['NAME_FAMILY_STATUS'].astype('str'))

#dataset['FONDKAPREMONT_MODE']=number.fit_transform(dataset['FONDKAPREMONT_MODE'].astype('str'))

dataset['ORGANIZATION_TYPE']=number.fit_transform(dataset['ORGANIZATION_TYPE'].astype('str'))

dataset['NAME_CONTRACT_TYPE']=number.fit_transform(dataset['NAME_CONTRACT_TYPE'].astype('str'))

#dataset['WALLSMATERIAL_MODE']=number.fit_transform(dataset['WALLSMATERIAL_MODE'].astype('str'))

#dataset['CODE_GENDER']=number.fit_transform(dataset['CODE_GENDER'].astype('str'))

dataset['NAME_EDUCATION_TYPE']=number.fit_transform(dataset['NAME_EDUCATION_TYPE'].astype('str'))

#dataset['NAME_TYPE_SUITE']=number.fit_transform(dataset['NAME_TYPE_SUITE'].astype('str'))

#dataset['WEEKDAY_APPR_PROCESS_START']=number.fit_transform(dataset['WEEKDAY_APPR_PROCESS_START'].astype('str'))

dataset['NAME_HOUSING_TYPE']=number.fit_transform(dataset['NAME_HOUSING_TYPE'].astype('str'))

#dataset['HOUSETYPE_MODE']=number.fit_transform(dataset['HOUSETYPE_MODE'].astype('str'))

#dataset['FLAG_OWN_CAR']=number.fit_transform(dataset['FLAG_OWN_CAR'].astype('str'))

#dataset['EMERGENCYSTATE_MODE']=number.fit_transform(dataset['EMERGENCYSTATE_MODE'].astype('str'))

#dataset['FLAG_OWN_REALTY']=number.fit_transform(dataset['FLAG_OWN_REALTY'].astype('str'))

dataset['OCCUPATION_TYPE']=number.fit_transform(dataset['OCCUPATION_TYPE'].astype('str'))

dataset
pd.value_counts(dataset['TARGET']).plot.bar()

plt.title('Fraud class histogram')

plt.xlabel('Class')

plt.ylabel('Frequency')

dataset['TARGET'].value_counts()
X = dataset.iloc[:, [0,2,7,8,9,12,13,15,18,28,40]].values

y = dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

import keras

from keras.models import Sequential

from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
from sklearn.metrics import accuracy_score 

accuracy_Sequential = accuracy_score(y_test,y_pred) 

accuracy_Sequential
X_test.shape
1604/1721