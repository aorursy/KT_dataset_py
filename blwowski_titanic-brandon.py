import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import operator



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import ModelCheckpoint



from sklearn.model_selection import train_test_split



import os

print(os.listdir("../working/"))



os.system("rm ../working/*.hdf5")
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

true_result = pd.read_csv('../input/gender_submission.csv')
import re

def clean_variable(df):

    df.Age = df.Age.fillna(-0.5)

    df.Age = pd.cut(df.Age, (-1,0,5, 13, 20, 30, 45, 65, 100), labels=['Unknown', 'Baby', 'Child', 'Teen', 'Young_Adult', 'Adult', 'Senior', 'Old'])

    df.Embarked = df.Embarked.fillna('N')

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    df.Fare = df.Fare.fillna(-0.5)

    df.Fare = pd.qcut(df['Fare'], 6)

    df['Title'] = df.Name.apply(lambda x: x.split(', ')[1].split('.')[0])

    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

    df['Single'] = 1

    df['Single'].loc[df['Family_Size'] > 1] = 0 

    df.Family_Size = pd.cut(df.Family_Size, (-1, 2, 4, 6, 8, 10), labels=['S','M', 'L', 'XL', 'XXL'])

    df= df.drop(['Name', 'SibSp', 'Parch'], axis = 1)

    return df



train = clean_variable(train)

test = clean_variable(test)

train.head()
# normalize the titles

normalized_titles = {

    "Capt":"Capt",        "Col":"Officer",    "Major":"Officer",    "Dr":"Officer",              "Rev":"Officer",

    "Jonkheer":"Royalty",    "Don":"Royalty",    "Sir" :"Royalty",     "the Countess":"Royalty",    "Dona":"Royalty",    "Lady" :"Royalty",

    "Mme":"Mrs",             "Ms":"Mrs",         "Mrs" :"Mrs",

    "Mlle":"Miss",           "Miss" :"Miss",

    "Mr" :"Mr",

    "Master" :"Master"

    }

# map the normalized titles to the current titles 

train.Title = train.Title.map(normalized_titles)

test.Title = test.Title.map(normalized_titles)
from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Title','Embarked']#,'Ticket'

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test

    

train, test = encode_features(train, test)

train.head()
### Create Dummy Variables for Categorical Data

def dummy_column(df, column_name):

    df_dummy = pd.get_dummies(df[column_name], drop_first=True, prefix=column_name )

    df = df.join(df_dummy)

    df = df.drop([column_name], axis=1)

    return df



train = dummy_column(train, 'Pclass')

train = dummy_column(train, 'Sex')

train = dummy_column(train, 'Age')

train = dummy_column(train, 'Family_Size')

train = dummy_column(train, 'Embarked')

train = dummy_column(train, 'Title')

train = dummy_column(train, 'Cabin')

train = dummy_column(train, 'Fare')





test = dummy_column(test, 'Pclass')

test = dummy_column(test, 'Sex')

test = dummy_column(test, 'Age')

test = dummy_column(test, 'Family_Size')

test = dummy_column(test, 'Embarked')

test = dummy_column(test, 'Title')

test = dummy_column(test, 'Cabin')

test = dummy_column(test, 'Fare')



test_list  = [col for col in train.columns if col not in  test.columns ]

train_list = [col for col in test.columns if col not in  train.columns ]



for missing_col in test_list:

    test[str(missing_col)] = 0

    

for missing_col in train_list:

    train[str(missing_col)] = 0

    

    

column_order = list(train.columns)

test = test[column_order]
X_all = train.drop(['Survived', 'PassengerId','Ticket'], axis=1)

y_all = train['Survived']



num_test = 0.75

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=123)

X_train.head()
print ("Train Shape {}".format(X_train.shape))

print ("Train Label Shape {}".format(y_train.shape))

print ("Test Shape {}".format(X_test.shape))

print ("Test Label Shape {}".format(y_test.shape))
model = Sequential()

model.add(Dense(32, input_dim = 40, activation = 'tanh'))

model.add(Dropout(.5))

model.add(Dense(32, activation = 'tanh'))

model.add(Dropout(.5))

model.add(Dense(16, activation = 'tanh'))

model.add(Dense(8, activation = 'tanh'))

model.add(Dense(4, activation = 'tanh'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]



history = model.fit(X_train, y_train, epochs=150, verbose=1, validation_data=(X_test,y_test), batch_size=128, shuffle=True, callbacks=callbacks_list)
import matplotlib.pyplot as plt

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

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.load_weights("weights.best.hdf5")
ids = test['PassengerId']

predictions = model.predict(test.drop(['PassengerId','Ticket', 'Survived'], axis=1))

final_predictions = []
for each in predictions:

    if each[0] >= .75:

        score = 1

    else:

        score = 0

    final_predictions.append(score)
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': final_predictions })

output.to_csv('mylastsub.csv', index = False)
output.Survived.value_counts()
y_pred_id = output['PassengerId'].tolist()

y_true_id = true_result['PassengerId'].tolist()

print('If PassengerId is in the right order: '+str(y_pred_id == y_true_id))
test