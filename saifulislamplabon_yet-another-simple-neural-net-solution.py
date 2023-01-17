import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



from sklearn import preprocessing

from keras.models import Sequential

from keras.layers import Dense, Dropout
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')



train_data_copy = train_data.copy()

test_data_copy = test_data.copy()
train_data.head()
test_data['Test'] = 1

train_data['Test'] = 0

data = train_data.append(test_data, sort = False)



drop_cols = list()

one_hot_encoding_cols = list()

normalization_cols = list()
data.drop('PassengerId', axis = 1, inplace = True)
data['Pclass'].value_counts(dropna = False)



normalization_cols.append('Pclass')
data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

data.drop('Name', axis = 1, inplace = True)
data['Title'].value_counts()
mapping = {'Col': 'Army', 'Mlle' : 'Miss', 'Major' : 'Army', 'Sir': 'Royal',

          'Mme': 'Mrs', 'Capt' : 'Army', 'Don' : 'Royal', 'Jonkheer' : 'Royal',

          'Ms' : 'Miss', 'Countess' : 'Royal', 'Lady': 'Royal'}

           

data.replace({'Title': mapping}, inplace=True)
one_hot_encoding_cols.append('Title')

data.head()
data['Sex'].value_counts(dropna = False)
label_encoder = preprocessing.LabelEncoder()

data['Sex'] = label_encoder.fit_transform(data['Sex'])



data.head()
data[data['Test'] == 0].groupby(['Title', 'Sex']).Age.mean()
data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Army'), 'Age'] = 56.60



data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Dr'), 'Age'] = 49.00

data.loc[(data.Age.isna()) & (data.Sex == 0) & (data.Title == 'Dr'), 'Age'] = 40.60



data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Master'), 'Age'] = 4.57



data.loc[(data.Age.isna()) & (data.Sex == 0) & (data.Title == 'Miss'), 'Age'] = 21.85



data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Mr'), 'Age'] = 32.36



data.loc[(data.Age.isna()) & (data.Sex == 0) & (data.Title == 'Mrs'), 'Age'] = 35.78



data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Rev'), 'Age'] = 43.16



data.loc[(data.Age.isna()) & (data.Sex == 0) & (data.Title == 'Royal'), 'Age'] = 40.50

data.loc[(data.Age.isna()) & (data.Sex == 1) & (data.Title == 'Royal'), 'Age'] = 42.33
normalization_cols.append('Age')

data.head()
data['SibSp'].value_counts()

normalization_cols.append('SibSp')
data['Parch'].value_counts()

normalization_cols.append('Parch')



data.head()
data.drop('Ticket', axis =1, inplace = True)
normalization_cols.append('Fare')

data['HasCabin'] = data['Cabin'].isnull() == False

data['HasCabin'].replace(False, 0, inplace = True)

data['HasCabin'].replace(True, 1, inplace = True)



data.drop('Cabin', axis =1, inplace = True)
one_hot_encoding_cols.append('Embarked')
data.head()
data = pd.get_dummies(data = data, columns = one_hot_encoding_cols)
data.head()
std = data[data['Test'] == 0][normalization_cols].std(axis = 0)

mean = data[data['Test'] == 0][normalization_cols].mean(axis = 0)



data[normalization_cols] = (data[normalization_cols] - mean) / std
data.head(10)
train_data = data[data['Test'] == 0].drop(columns = ['Test'])



test_data = data[data['Test'] == 1].drop(columns = ['Survived', 'Test'])
train_data.head()

train_data.shape
test_data.head()
X = train_data.iloc[: , 1:].to_numpy()

y = train_data.iloc[:, 0].to_numpy()



print(str(X.shape))

print(str(y.shape))
def create_model():



    model = Sequential()

    model.add(Dense(14, input_dim = 19, activation = 'relu'))

    model.add(Dropout(0.3))

    model.add(Dense(8, activation = 'relu'))

    

    model.add(Dense(1, activation = 'sigmoid'))



    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    

    return model
epochs = 20

model = create_model()

history = model.fit(X, y, epochs=epochs, validation_split = 0.3, batch_size=10)
epochs = 20

model = create_model()

history = model.fit(X, y, epochs=epochs, batch_size=10, verbose = 0)
X_test = test_data.to_numpy()
prediction = model.predict(X_test)
submission = pd.DataFrame(test_data_copy[['PassengerId']])

submission['Survived'] = prediction

submission['Survived'] = submission['Survived'].apply(lambda x: 0 if x < 0.5 else 1)
submission.to_csv('submission.csv', index = False)
submission.head(10)