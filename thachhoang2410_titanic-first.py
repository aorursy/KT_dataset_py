# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print("Training shape: ", train.shape)

print("Training info: ")

train.info()

print("\n-----------------------------------------\n")

print("Test shape: ", test.shape)

print("Test info: ")

test.info()
print("Check NaN values in Test set:")

isnull = test.isnull().sum().reset_index()

#isnull[isnull>0]

isnull.columns = ['Feature', 'Total_null']

total_null = isnull[isnull['Total_null']>0]

total_null
print("Check NaN values in Train set:")

isnull = train.isnull().sum().reset_index()

#isnull[isnull>0]

isnull.columns = ['Feature', 'Total_null']

total_null = isnull[isnull['Total_null']>0]

total_null
traintest = pd.concat([train, test], axis=0, sort=False)
traintest[traintest.duplicated()].index
pd.concat([traintest.nunique(dropna=False), traintest.count(), traintest.nunique()/traintest.count()], axis=1)
train = train.replace('male', 0)

train = train.replace('female', 1)

test = test.replace('male', 0)

test = test.replace('female', 1)
import seaborn as sns

import math



cols = traintest.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'Embarked']).columns

nrows = math.ceil(len(cols)/3)

fig, axs = plt.subplots(ncols=3, nrows=nrows)

fig.set_size_inches(16, 7*nrows)



index = 0

for col in cols:

    sns.kdeplot(train[col], ax=axs[math.floor(index/3), index%3], label='Train_'+col)

    sns.kdeplot(test[col], ax=axs[math.floor(index/3), index%3], label='Test_'+col)

    index += 1



#sns.kdeplot(train.Age, ax=axs[1], label='Train_Age')

#sns.kdeplot(test.Age, ax=axs[1],label='Test_Age')



#sns.kdeplot(train.SibSp, ax=axs[2], label='Train_SibSp')

#sns.kdeplot(test.SibSp, ax=axs[2], label='Test_SibSp')

# plot survival vs sex

temp = train[['Survived', 'Sex']].groupby('Survived').count().plot.bar()
train['Age'].describe()
temp = train[['Survived', 'Age']].groupby('Age').count().plot(figsize=(16,7), kind='bar')

plt.title('Number of people per Age group')
temp = train[['Survived', 'Age']].groupby('Age').sum().plot(figsize=(16,7), kind='bar')

plt.title('Number of people survived per Age group')
temp = train[['Survived', 'Age']].groupby('Age').sum()

temp2 = train['Age'].value_counts()



plt.figure(figsize=(16,7))

plt.bar(temp.index, align='center', height=temp['Survived'].astype('float')/temp2)

plt.title('The probability of Survived over Age group')
temp = train[['Survived', 'SibSp']].groupby('SibSp').sum().plot(kind='bar', figsize=(16,7))

plt.title('Number of people survived per SibSp group')
#temp = train[['Survived', 'SibSp']].groupby('SibSp').sum().plot(kind='bar', figsize=(16,7))



temp = train[['Survived', 'SibSp']].groupby('SibSp').sum()

temp2 = train[['Survived', 'SibSp']].groupby('SibSp').count()



plt.figure(figsize=(16,7))

plt.bar(temp.index, align='center', height=temp['Survived'].astype('float')/temp2['Survived'])

plt.title('The probability of Survived over SibSp group')
temp2 = train[['Survived', 'Parch']].groupby('Parch').count().plot(figsize=(16,7), kind='bar')

plt.title('Number of people per Parch group')
temp = train[['Survived', 'Parch']].groupby('Parch').sum().plot(kind='bar', figsize=(16,7))

plt.title('Number of people survived per Parch group')
#temp = train[['Survived', 'SibSp']].groupby('SibSp').sum().plot(kind='bar', figsize=(16,7))



temp = train[['Survived', 'Parch']].groupby('Parch').sum()

temp2 = train[['Survived', 'Parch']].groupby('Parch').count()



plt.figure(figsize=(16,7))

plt.bar(temp.index, align='center', height=temp['Survived'].astype('float')/temp2['Survived'])

plt.title('The probability of Survived over Parch group')
train['SibSp_parch'] = train.SibSp + train.Parch

test['SibSp_parch'] = test.SibSp + train.Parch

print(train['SibSp_parch'].describe())

temp = train[['Survived', 'SibSp_parch']].groupby('SibSp_parch').sum()

temp2 = train[['Survived', 'SibSp_parch']].groupby('SibSp_parch').count()



plt.figure(figsize=(16,7))

plt.bar(temp.index, align='center', height=temp['Survived'].astype('float')/temp2['Survived'])

plt.title('The probability of Survived over family size')
temp2 = train[['Survived', 'SibSp_parch']].groupby('SibSp_parch').count().plot(figsize=(16,7), kind='bar')

plt.title('Number of people per Family size')
# merge Train and Test dataframe to get one-hot encoding

# add a column to indicate whether the row is in Train or Test set

train['is_train'] = 1

train['origin_index'] = train.index 

test['is_train'] = 0

test['origin_index'] = test.index 



traintest = pd.concat([train, test], axis=0, sort=False)
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))



num_cols = ['Age', 'Fare', 'SibSp_parch']

# one-hot coding categorical features

cat_cols = ['Sex', 'Cabin', 'Embarked', 'Pclass', 'SibSp', 'Parch']



# Generate new feature

train['SibSp_parch'] = train.SibSp + train.Parch

test['SibSp_parch'] = test.SibSp + train.Parch



# create a temp set

scaled_train = train[num_cols+cat_cols].copy()

scaled_test = test[num_cols+cat_cols].copy()



one_hot_data = pd.get_dummies(traintest[cat_cols+['is_train', 'origin_index']], columns=cat_cols)

train_one_hot = one_hot_data[one_hot_data.is_train==1]

test_one_hot = one_hot_data[one_hot_data.is_train==0]



scaled_train = scaled_train.merge(train_one_hot, left_index=True, right_on='origin_index')

scaled_test = scaled_test.merge(test_one_hot, left_index=True, right_on='origin_index')



scaled_train = scaled_train.drop(columns=cat_cols + ['origin_index', 'is_train'])

scaled_test = scaled_test.drop(columns=cat_cols+ ['origin_index', 'is_train'])



# replace all NaN value to -1

#train.fillna(-1, inplace=True)

#test.fillna(-1, inplace=True)

scaled_train = scaled_train.fillna(scaled_train.mean())

scaled_test = scaled_test.fillna(scaled_train.mean())



y_train = train['Survived']

X_train = scaled_train



X_test = scaled_test



# fit scaler

X_train[num_cols] = min_max_scaler.fit_transform(X_train[num_cols])

#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

X_test[num_cols] = min_max_scaler.fit_transform(X_test[num_cols])
print("X_train.shape: ", X_train.shape)

print("y_train.shape: ", y_train.shape)



print("X_test.shape: ", X_test.shape)
# split train/test set

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression
x_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
model_svm_linear = SVC(kernel='linear', C=1).fit(x_train, Y_train)

print("Training score: ", model_svm_linear.score(x_train, Y_train))

print("Validation score: ", model_svm_linear.score(X_val, Y_val))
model_svm_sigmoid = SVC(kernel='sigmoid').fit(x_train, Y_train)

print("Training score: ", model_svm_sigmoid.score(x_train, Y_train))

print("Validation score: ", model_svm_sigmoid.score(X_val, Y_val))
model_svm_rbf = SVC(kernel='rbf').fit(x_train, Y_train)

print("Training score: ", model_svm_rbf.score(x_train, Y_train))

print("Validation score: ", model_svm_rbf.score(X_val, Y_val))
model_logistic = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train, Y_train)

print("Training score: ", model_logistic.score(x_train, Y_train))

print("Validation score: ", model_logistic.score(X_val, Y_val))
model = SVC(kernel='linear', C=1).fit(X_train, y_train)

print("Training score: ", model_svm_linear.score(X_train, y_train))



class_predict = model.predict(X_test)
test_ID = test['PassengerId']
temp = {'PassengerID': test_ID, 'Survived': class_predict}

result = pd.DataFrame(temp)
result.to_csv('result.csv', index=False)
# import library

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.callbacks import EarlyStopping

from sklearn import preprocessing
model_neuron = Sequential() 

#model_neuron.add(Dense(output_dim=512, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dense(output_dim=256, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dropout(0.5))

model_neuron.add(Dense(output_dim=128, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dropout(0.5))

model_neuron.add(Dense(output_dim=64, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dropout(0.5))

model_neuron.add(Dense(output_dim=32, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dense(output_dim=16, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dense(output_dim=1, input_shape=(X_train.shape[1],), activation='sigmoid'))

model_neuron.compile(loss='mse', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

model_neuron.summary()
history = model_neuron.fit(X_train, y_train, nb_epoch=10000, validation_split=0.2, callbacks=[EarlyStopping(patience=10)])
import matplotlib.pyplot as plt

# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
model_neuron = Sequential() 

#model_neuron.add(Dense(output_dim=512, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dense(output_dim=256, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dense(output_dim=128, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dense(output_dim=64, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dense(output_dim=32, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dense(output_dim=16, input_shape=(X_train.shape[1],), activation='relu'))

model_neuron.add(Dense(output_dim=1, input_shape=(X_train.shape[1],), activation='sigmoid'))

model_neuron.compile(loss='mse', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

model_neuron.summary()



history = model_neuron.fit(X_train, y_train, nb_epoch=200, callbacks=[EarlyStopping(patience=10)])
class_predict = model_neuron.predict_classes(X_test)

class_predict = class_predict.reshape((class_predict.shape[0],))

test_ID = test['PassengerId']

temp = {'PassengerID': test_ID, 'Survived': class_predict}

result = pd.DataFrame(temp)

result.to_csv('result_neuron.csv', index=False)
# fill nan value to easy separate them

cabins_list = traintest.Cabin.fillna('NaN')

cabins = []

for value in cabins_list:

    if value != 'NaN':

        cabins += value.split(' ')

        

unique_cabins = set(cabins)

print("Number unique in cabins: ", len(unique_cabins))

print("Number of Cabins: ", len(cabins))

print("Number of nan in Cabin: ", traintest.Cabin.isnull().sum())

print("Number of row not nan in Cabin: ", traintest.Cabin.shape[0] - traintest.Cabin.isnull().sum())

       