# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json

import seaborn as sns

sns.set(style="darkgrid")

import matplotlib.pyplot as plt

%matplotlib inline
# loading files

breed_labels = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/breed_labels.csv")

color_labels = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/color_labels.csv")

state_labels = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/state_labels.csv")



train = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/train/train.csv")

test = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/test/test.csv")

test_ids = test['PetID']

columns = train.columns
train.head()
# functions 

def preprocess(train, column_name, table):

    d = dict(zip(range(len(table)), table))

    train[column_name].replace(d, inplace=True)

    

def id_to_state(id):

    return state_labels['StateName'][state_labels['StateID'] == id].values[0]



def simple_plot(train, column_name, plot_title=None):

    ax = sns.countplot(train[column_name])

    if plot_title is not None:

        ax.set(title = plot_title)

    set_values(ax)

    

def set_values(ax):

    for p in ax.patches:

        ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 7), textcoords = 'offset points')

        

def id_to_breed(id):

    if id != 0:

        return breed_labels['BreedName'][breed_labels['BreedID'] == id].values[0]

    return 0



# cleaning data

preprocessing_table = {

    'MaturitySize': ['Not Specified', 'Small', 'Medium', 'Large', 'Extra Large'],

    'FurLength': ['Not Specified', 'Short', 'Medium', 'Long'],

    'Vaccinated': ['Yes', 'No', 'Not Sure'],

    'Dewormed': ['Yes', 'No', 'Not Sure'],

    'Sterilized': ['Yes', 'No', 'Not Sure'],

    'Health': ['Not Specified', 'Healthy', 'Minor Injury', 'Serious Injury'],

    'Type': [None, 'Dog', 'Cat'],

    'Gender': [None, 'Male', 'Female', 'Group'],

    'Color1': ['Not defined', 'Black', 'Brown', 'Golden', 'Yellow', 'Cream', 'Gray', 'White'],

    'Color2': ['Not defined', 'Black', 'Brown', 'Golden', 'Yellow', 'Cream', 'Gray', 'White'],

    'Color3': ['Not defined', 'Black', 'Brown', 'Golden', 'Yellow', 'Cream', 'Gray', 'White'],

    

}
def clean(train):

    for key in preprocessing_table:

        preprocess(train, key, preprocessing_table[key])



    train['Name'] = train['Name'][pd.notnull(train['Name'])].apply(lambda x: 'Not defined' if ('name' in x.lower() or len(x) < 3) else x)

    train['Name'].fillna('Not defined', inplace = True)



    names = train['State'].unique().tolist()

    names_states = [id_to_state(i) for i in names]

    d = dict(zip(names, names_states))

    train['State'].replace(d, inplace=True)



clean(train)

clean(test)
train.head()
plt.figure(figsize=(23,17))



plt.subplot(2,2,1)

sns.countplot(train['Type'])



plt.subplot(2,2,2)

sns.countplot(train['Gender'])



plt.subplot(2,2,3)

ax = sns.kdeplot(data=train['Age'], shade=True, gridsize = 30)

_ = ax.set(title='Age distribution', ylabel='Distribution', xlabel='Age - months')
num = 10

mixed_breed_class = 307



plt.figure(figsize=(20,20))



indexes, values = train['Breed1'][(train['Type'] == 'Dog')].value_counts().index[:num], train['Breed1'].value_counts()[:num]

names = [id_to_breed(i) for i in indexes]

s = pd.Series(data={'values': values.values, 'names': names})

ax = sns.catplot(x = 'values', y = 'names' , kind='bar', data = s)

_ = ax.set(title=f'Dog breed classes top {num}', ylabel='Dog breed', xlabel='Count')

    

indexes, values = train['Breed1'][(train['Type'] == 'Cat')].value_counts().index[:num], train['Breed1'].value_counts()[:num]

names = [id_to_breed(i) for i in indexes]

s = pd.Series(data={'values': values.values, 'names': names})

ax = sns.catplot(x = 'values', y = 'names' , kind='bar', data = s)

_ = ax.set(title=f'Cat breed classes top {num}', ylabel='Cat breed', xlabel='Count')



pure_breeded = train['Breed1'].apply(lambda x: 0 if id_to_breed(x) in ['Mixed Breed', 'Domestic Short Hard', 'Domestic Medium Hair', 'Domestic Long Hair'] else 1)

print(f'Pure breeded pets: {sum(pure_breeded)}\nNot pure breeded pets: {len(pure_breeded)-sum(pure_breeded)}')
plt.figure(figsize=(25,7.5))



plt.subplot(1,3,1)

ax = sns.countplot(train['Color1'])

ax.set(title='First color')



plt.subplot(1,3,2)

ax = sns.countplot(train['Color2'])

ax.set(title='Second color')



plt.subplot(1,3,3)

ax = sns.countplot(train['Color3'])

_ = ax.set(title='Third color')
plt.figure(figsize=(25,15))



plt.subplot(2,3,1)

simple_plot(train, 'MaturitySize')



plt.subplot(2,3,2)

simple_plot(train, 'FurLength')



plt.subplot(2,3,3)

simple_plot(train, 'Vaccinated')



plt.subplot(2,3,4)

simple_plot(train, 'Dewormed')



plt.subplot(2,3,5)

simple_plot(train, 'Sterilized')



plt.subplot(2,3,6)

simple_plot(train, 'Health')
plt.figure(figsize=(15,7))

simple_plot(train, 'Quantity', 'Number of pets in profile')
fee = pd.concat([train['Fee'][train['Fee'] == 0], pd.qcut(train['Fee'][train['Fee'] != 0], 5).sort_values()], axis=0)

plt.figure(figsize=(10,5))

ax = sns.countplot(fee)

ax.set(title = 'Fee amount for pet')

set_values(ax)
plt.figure(figsize=(20,5))

ax = sns.countplot(train['State'])

ax.set(title = 'State location in Malaysia')

set_values(ax)
plt.figure(figsize=(20,6))



plt.subplot(1,2,1)

simple_plot(train, 'VideoAmt', 'Number of pet videos uploaded')

ax = sns.countplot(train['VideoAmt'])

ax.set(title = 'Number of pet videos uploaded')

set_values(ax)



plt.subplot(1,2,2)

ax = sns.countplot(train['PhotoAmt'].sort_values().apply(lambda x: 'Over 10' if x > 10 else x))

ax.set(title = 'Number of pet photos uploaded')

set_values(ax)
def get_magnitude_avg(x, w):

    if os.path.exists(f"/kaggle/input/petfinder-adoption-prediction/train_sentiment/{x['PetID']}.json"):

        j = json.load(open(f"/kaggle/input/petfinder-adoption-prediction/train_sentiment/{x['PetID']}.json"))

        summ, num = 0, 0

        for sent in j['sentences']:

            summ += sent['sentiment'][w]

            num += 1

        return summ/num

    else:

        return None

            

train['Description_magnitude'] = train.apply(lambda x: get_magnitude_avg(x, 'magnitude'), axis=1)            

train['Description_score'] = train.apply(lambda x: get_magnitude_avg(x, 'score'), axis=1)
plt.figure(figsize=(20,6))



plt.subplot(1,2,1)

ax = sns.distplot(a=train['Description_magnitude'], kde = False)

ax.set(title='Magnitude - strength of emotion in description.', xlabel='Description magnitude', ylabel='count')



plt.subplot(1,2,2)

ax = sns.distplot(a=train['Description_score'], kde = False)

_ = ax.set(title='Score - emotional leaning of the description -1 - negative, 1 - positive', xlabel='Description score', ylabel='count')
print(f'Columns in train data before preprocessing:\n\n {train.columns.to_list()}')
def limit_column(train, test, column_name, values):

    train[column_name] = train[column_name].apply(lambda x: x if x in values else 'Other')

    test[column_name] = test[column_name].apply(lambda x: x if x in values else 'Other')



limit_column(train, test, 'Breed1', test['Breed1'].value_counts().index[:3].to_list())

limit_column(train, test, 'Breed2', test['Breed2'].value_counts().index[:4].to_list())

limit_column(train, test, 'State', test['State'].value_counts().index[:3].to_list())
# preprocessing to fit model

from sklearn import preprocessing



def preprocess_all(train):

    train['Name'] = train['Name'].apply(lambda x: 'Defined' if x != 'Not defined' else x)



    # normalization

    normalization_columns = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']

    x = train[normalization_columns].values

    scaler = preprocessing.MinMaxScaler()

    x_scaled = scaler.fit_transform(x)

    train[normalization_columns] = pd.DataFrame(x_scaled, columns=normalization_columns)



    #one hot encoding

    dummies_columns = ['Type', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Breed1', 'Breed2', 'State', 'Name']

    train = pd.get_dummies(train, columns=dummies_columns)

    

    train.drop(['RescuerID', 'Description', 'PetID'], axis = 1, inplace = True)

    

    return train



train_labels = train['AdoptionSpeed']    

train.drop(['AdoptionSpeed', 'Description_magnitude', 'Description_score'], axis = 1, inplace = True)
train = preprocess_all(train)

test = preprocess_all(test)



X_train, Y_train, X_test = train.values, train_labels.values, test.values

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(f'Columns in train data after preprocessing:\n\n {train.columns.to_list()}')
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv1D, Flatten



model = Sequential()



model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape = (X_train[0].shape[0], 1)))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Conv1D(filters=256, kernel_size=2, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Conv1D(filters=512, kernel_size=2, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Conv1D(filters=1024, kernel_size=2, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Flatten())



model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(5, activation='softmax'))



model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=10, validation_split=0.2)
plt.figure(figsize=(10,7))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['Train loss', 'Validation loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss (categorical crossentropy)')
from sklearn.metrics import confusion_matrix



plt.figure(figsize=(10,7))

train_predictions = model.predict_classes(X_train)

matrix = confusion_matrix(Y_train, train_predictions)

ax = sns.heatmap(matrix, annot=True, fmt='d', linewidths=0.25)

_ = ax.set(xlabel='Predicted class', ylabel='Actual class')
from sklearn.metrics import cohen_kappa_score



sc = cohen_kappa_score(train_predictions, Y_train, weights = 'quadratic')

print(f'Quadratic kappa score: {sc}')
model = Sequential()



model.add(Flatten(input_shape = (X_train[0].shape[0], 1)))



model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(1024, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(5, activation='softmax'))



model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=15, validation_split=0.2)
plt.figure(figsize=(10,7))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['Train loss', 'Validation loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss (categorical crossentropy)')
from sklearn.metrics import cohen_kappa_score



train_predictions = model.predict_classes(X_train)

sc = cohen_kappa_score(train_predictions, Y_train, weights = 'quadratic')

print(f'Quadratic kappa score: {sc}')
test_predictions = model.predict_classes(X_test)

my_submission = pd.DataFrame({'PetID': range(len(test_predictions)), 'AdoptionSpeed': test_predictions})

my_submission['PetID'] = test_ids

my_submission.to_csv('submission.csv', index=False)