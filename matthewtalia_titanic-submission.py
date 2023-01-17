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
# Load into Pandas dataframe
titanic_data = pd.read_csv("/kaggle/input/titanic/train.csv")
#pd.set_option('display.max_rows',None)

# Load test data
titanic_test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
titanic_data.head()
# Make copy
titanic_train = titanic_data
titanic_test = titanic_test_data

# Drop PassengerId since non-predictive
titanic_train = titanic_train.drop(['PassengerId'],axis=1)
titanic_test = titanic_test.drop(['PassengerId'],axis=1)

# Extract Cabin information
titanic_train['CabinLetter'] = titanic_train['Cabin'].str.extract(pat='([A-G])')
titanic_train = titanic_train.drop(['Cabin'],axis=1)

titanic_test['CabinLetter'] = titanic_test['Cabin'].str.extract(pat='([A-G])')
titanic_test = titanic_test.drop(['Cabin'],axis=1)

titanic_train['CabinLetter'] = titanic_train['CabinLetter'].fillna('Unknown')
titanic_test['CabinLetter'] = titanic_test['CabinLetter'].fillna('Unknown')

# Combine Siblings and Parents
titanic_train['Family_num'] = titanic_train['SibSp'] + titanic_train['Parch']
titanic_test['Family_num'] = titanic_test['SibSp'] + titanic_test['Parch']
titanic_train = titanic_train.drop(['SibSp','Parch'],axis=1)
titanic_test = titanic_test.drop(['SibSp','Parch'],axis=1)

# Fare per person
titanic_train['Fpp'] = titanic_train['Fare']/(titanic_train['Family_num']+1)
titanic_train = titanic_train.drop(['Fare'],axis=1)
titanic_test['Fpp'] = titanic_test['Fare']/(titanic_test['Family_num']+1)
titanic_test = titanic_test.drop(['Fare'],axis=1)

# Extract ticket numbers
titanic_train['Ticket_num'] = titanic_train['Ticket'].str.extract(pat='(\d{2,}$)')
titanic_test['Ticket_num'] = titanic_test['Ticket'].str.extract(pat='(\d{2,}$)')
# Extract prefactor
titanic_train['Ticket'] = titanic_train['Ticket'].str.replace('LINE','LINE ')
df2_train = titanic_train['Ticket'].str.extract(pat='(^.+\s)')

titanic_test['Ticket'] = titanic_test['Ticket'].str.replace('LINE','LINE ')
df2_test = titanic_test['Ticket'].str.extract(pat='(^.+\s)')

# Cleaning training set
df2_train = df2_train[0].str.replace('[^\w\s]','')
df2_train = df2_train.str.replace(' ','')
df2_train = df2_train.str.replace('SCParis','SCPARIS')
df2_train = df2_train.str.replace('STONO2','SOTONO2')

df2_test = df2_test[0].str.replace('[^\w\s]','')
df2_test = df2_test.str.replace(' ','')
df2_test = df2_test.str.replace('SCParis','SCPARIS')
df2_test = df2_test.str.replace('STONO2','SOTONO2')
df2_test = df2_test.str.replace('STONOQ','SOTONOQ')

# Compare ticket pre in test and train set
df2_train = pd.get_dummies(df2_train)
df2_test = pd.get_dummies(df2_test)

df2_test[df2_train.columns.difference(df2_test.columns)] = 0
df2_train[df2_test.columns.difference(df2_train.columns)] = 0

df2_train.sum(axis = 0, skipna = True).sort_values()
# Keep largest
keep_pre = ['A5','CA','PC','SCPARIS','SOTONO2','SOTONOQ','WC']
df2_train = df2_train[keep_pre]
df2_test = df2_test[keep_pre]

# Add to dataframes??????????????????
titanic_train = pd.concat([titanic_train, df2_train], axis=1)
titanic_test = pd.concat([titanic_test, df2_test], axis=1)

# Drop Ticket feature
titanic_train = titanic_train.drop(['Ticket'],axis=1)
titanic_test = titanic_test.drop(['Ticket'],axis=1)

########################### Drop some info?
titanic_train = titanic_train.drop(['CabinLetter'],axis=1)
titanic_test = titanic_test.drop(['CabinLetter'],axis=1)
# Extract titles from names
title = '(Mr\.)|(Mrs\.)|(Miss\.)|(Master\.)'#'|(Dr\.)|(Mlle\.)|(Don\.)|(Lady.)|(Sir\.)|(Col\.)|(Rev\.)|(Capt\.)|(Countess\.)'
df_train = titanic_train['Name'].str.extract(pat=title)
df_test = titanic_test['Name'].str.extract(pat=title)
df_train = df_train[df_train.columns[0:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)
df_test = df_test[df_test.columns[0:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)
df_train = df_train.replace(r'^\s*$', np.nan, regex=True)
df_test = df_test.replace(r'^\s*$', np.nan, regex=True)
df_train = df_train.fillna('Unknown')
df_test = df_test.fillna('Unknown')

df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

df_test[df_train.columns.difference(df_test.columns)] = 0
df_train[df_test.columns.difference(df_train.columns)] = 0

# Drop Name feature
titanic_train = titanic_train.drop(['Name'],axis=1)
titanic_test = titanic_test.drop(['Name'],axis=1)

# Add to dataframe
titanic_train = pd.concat([titanic_train, df_train], axis=1)
titanic_test = pd.concat([titanic_test, df_test], axis=1)

df_train.sum(axis = 0, skipna = True)
df_test.sum(axis = 0, skipna = True) 
# Convert rest of categorical variables to numeric
titanic_train = pd.get_dummies(titanic_train, columns=['Sex','Embarked'])
titanic_test = pd.get_dummies(titanic_test, columns=['Sex','Embarked'])

titanic_train
# Set target variable and drop from dataframe
titanic_target = titanic_train['Survived'].to_numpy()
titanic_train = titanic_train.drop(['Survived'],axis=1)

titanic_test.shape
# Standardize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
titanic_std = scaler.fit_transform(titanic_train)
titanic_test = scaler.fit_transform(titanic_test)

# Deal with NaN values
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(titanic_std)
titanic_std = imp.transform(titanic_std)
imp.fit(titanic_test)
titanic_test = imp.transform(titanic_test)
from keras.models import Sequential
from keras.layers import Dense

from keras.optimizers import SGD, Adam

model = Sequential()

model.add(Dense(100, activation='relu', input_shape=(22,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

opt = Adam(learning_rate=5e-3)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(titanic_std, titanic_target,epochs=150, batch_size=10, verbose=0)
score = model.evaluate(titanic_std, titanic_target,verbose=1)
# Accuracy scores and metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
confusion_matrix(titanic_target,model.predict_classes(titanic_std))
# Dataframe with predictions
titanic_test_data['Survived'] = model.predict_classes(titanic_test)
titanic_test_data[['PassengerId','Survived']]
titanic_test_data[['PassengerId','Survived']].to_csv(r'\kaggle\working\submission.csv', index = False)