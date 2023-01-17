import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
from sklearn import preprocessing

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.sample(5)
test.sample(5)
print("Train")
for col in train:
    print(col, "-", np.sum(train[col].isna()))
    
print("\nTest")
for col in test:
    print(col, "-", np.sum(test[col].isna()))
train.Cabin[train.Cabin.notnull()].sample(10)
# Get first character of cabin
cCabin = train.Cabin.fillna("X")
firstChar = cCabin.astype(str).str[0]

# Determine if cabin is correlated with the number of dead
survival = train.Survived
cabinSurvived = pd.concat([firstChar, survival], axis=1)
cabinTypes = cabinSurvived.Cabin.unique()

labels = []
survived = []
totals = []
for cabin in cabinTypes:
    mask = cabinSurvived.Cabin == cabin
    labels.append(cabin)
    survived.append(cabinSurvived.Survived[mask].sum())
    totals.append(cabinSurvived.Survived[mask].count())
    

plt.bar(labels, totals, zorder=1)
plt.bar(labels, survived, zorder=2)
plt.title("Survived Over Total People Given Cabin")
plt.show()
plt.figure()
plt.bar(labels[1:], totals[1:], zorder=1)
plt.bar(labels[1:], survived[1:], zorder=2)
plt.show()
# Get first character of cabin
nnAge = train.Age.notnull()
cAge = train.Age[nnAge]

# Determine if cabin is correlated with the number of dead
survival = train.Survived[nnAge]
cAgeSurvived = pd.concat([cAge, survival], axis=1)
ageTypes = cAgeSurvived.Age.unique()
labels = []
survived = []
totals = []
for age in ageTypes:
    mask = cAgeSurvived.Age == age
    labels.append(age)
    survived.append(cAgeSurvived.Survived[mask].sum())
    totals.append(cAgeSurvived.Survived[mask].count())
    

plt.bar(labels, totals, zorder=1)
plt.bar(labels, survived, zorder=2)
plt.title("Survived Over Total People Given Age")
plt.show()
def convert_titles(data):
    title = data['Title']
    if title in ['Don', 'Jonkheer', 'Rev']:
        return 'Mr'
    if title in ['Major', 'Capt', 'Col']:
        return 'Military'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if data['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
def get_normalized_col(col):
    x = col.values
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    return pd.DataFrame(x_scaled)

# Modify provided data: all NaN are X cabins
def modify_data(data):
    # Create new column 'Title'
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data.apply(convert_titles, axis=1)
    data['Title'] = pd.Categorical(data.Title).codes
    
    # Modify Embarked/Sex to be codes
    data.Embarked = pd.Categorical(data.Embarked).codes
    data.Sex = pd.Categorical(data.Sex).codes
    
    # Convert SibSp and Parch into Family Parameter
    data['Family'] = data['SibSp'] + data['Parch']

    # Drop all non useful columns
    data = data.drop(columns=['Ticket', 'Parch', 'SibSp', 'Name', 'PassengerId', 'Cabin'])
    
    # Need to fix Age NaN, using Pclass as age classification.
    for title in data['Title'].unique():
        ageAvg = data.Age[data.Age.notnull()][data['Title'] == title].sum() / (data['Title'] == title).sum()
        data.loc[data.Age.isna() & (data['Title'] == title), 'Age'] = ageAvg
        
    # Scale integer values       
    data.loc[data.Fare.isna(), 'Fare'] = np.average(data.loc[data.Fare.notnull(), 'Fare'])
    
    # Add new age class column
    data['AgeClass'] = data['Age'] * data['Pclass']
    
    # Normalize larger numbers
    data['AgeClass'] = (data['AgeClass'] - data['AgeClass'].min()) / (data['AgeClass'].max() - data['AgeClass'].min())
    data['Age'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
    data['Fare'] = (data['Fare'] - data['Fare'].min()) / (data['Fare'].max() - data['Fare'].min())
    data['Family'] = (data['Family'] - data['Family'].min()) / (data['Family'].max() - data['Family'].min())
    return data
    
train_set = modify_data(train)
test_set = modify_data(test)

train_set
# Check for isNaN in trainset
print("Train Set")
for col in train_set:
    print(col, "-", np.sum(train_set[col].isna()))

print("\nTest Set")
for col in test_set:
    print(col, "-", np.sum(test_set[col].isna()))
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.utils import to_categorical
from keras import optimizers

from sklearn import model_selection, linear_model, metrics

keras.backend.clear_session()

xset = np.array(train_set.values.tolist())[:, 1:]
yset = np.array(train_set.values.tolist())[:, :1]
submit_x = np.array(test_set)


x_train, x_test, y_train, y_test = model_selection.train_test_split(xset, yset)#, random_state=0)
input_shape = x_train[0].shape

model = Sequential([Dense(8, activation='relu', input_shape=input_shape), BatchNormalization(), Dense(5, activation='relu'), Dense(2, activation='sigmoid')])
opt = optimizers.Adam()

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.build()
model.summary()
model.fit(x_train, to_categorical(y_train), epochs=15)
y_pred = model.predict(x_test)
print("Accuracy score:", metrics.accuracy_score(y_test, np.argmax(y_pred, axis=-1)))
predictions = np.argmax(model.predict(submit_x), axis=-1)

predicted_data = {'PassengerId': test.PassengerId, 'Survived': pd.Series(predictions)}
data = pd.DataFrame(predicted_data, columns = ['PassengerId', 'Survived'])
data
data.to_csv('submission.csv', index=False)