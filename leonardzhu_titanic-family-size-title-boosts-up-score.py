# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
def concat_df(train_data, test_data, removeTarget=True):

    """

    Returns a concatenated df of training and test set

    train_data, dataset for training

    test_data, dataset for testing

    """

    if removeTarget:

        test_data['Survived'] = -1

    train_data["Type"] = "Training"

    test_data["Type"] = "Testing"

    retData = pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

    retData = retData[train_data.columns]

    return retData



def divide_df(all_data, conditionName='Type', trainValue="Training", testValue="Testing", targetName="Survived"):

    """

    Returns divided dfs of training and test set

    Preseve the original order of test_data    

    all_data, completed data set

    """



    if trainValue == 'Others':

        ret_train = all_data[all_data[conditionName]!=testValue].copy()

    else:

        ret_train = all_data[all_data[conditionName]==trainValue].copy()

    ret_test = all_data[all_data[conditionName]==testValue].copy()

    ret_test = ret_test.drop([targetName], axis=1)

    return ret_train.reset_index(drop=True), ret_test.reset_index(drop=True)
train_data = pd.read_csv("/kaggle/input/titanic/train.csv",dtype={"Survived": np.int64})
train_data["Type"] = "Train"
train_data.info()
train_data.head()
train_data[train_data["Embarked"].isna()]
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data["Type"] = "Test"

test_data.head()
print("Train : ", train_data.shape)

print("Test : ", test_data.shape)
all_data = concat_df(train_data, test_data)
print(all_data.info())
all_data['Embarked'] = all_data['Embarked'].fillna('S') 
all_data['Title'] = all_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
all_data['Title'].value_counts()


all_data.loc[all_data['Title'].isin(['Dr', 'Col', 'Rev', 'Major','Lady', 'Jonkheer', 'Don','Sir', 'the Countess', 'Capt','Dona']),'Title'] = 'Nobility'

all_data.loc[all_data['Title'].isin(['Miss', 'Ms', 'Mlle']),'Title'] = 'Miss'

all_data.loc[all_data['Title'].isin(['Mrs', 'Mme']),'Title'] = 'Mrs'
fig, axs = plt.subplots(figsize=(15, 9))

sns.countplot(x='Title', hue='Survived', data=all_data[all_data['Sex'] == 'female'])
fig, axs = plt.subplots(figsize=(15, 9))

sns.countplot(x='Title', hue='Survived', data=all_data[all_data['Sex'] == 'male'])
all_data['Family_Size'] = all_data['SibSp'] + all_data['Parch'] + 1
fig, axs = plt.subplots(figsize=(15, 9))

sns.countplot(x='Family_Size', hue='Survived', data=all_data)


family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

all_data['Family_Size_Grouped'] = all_data['Family_Size'].map(family_map)
train_data, test_data = divide_df(all_data)
print("train ", train_data.shape)

print("test ", test_data.shape)
from sklearn.ensemble import RandomForestClassifier
target = ['Survived']

#features = ['Pclass', 'Sex', 'SibSp', 'Parch','Embarked', 'AgeGroup', 'fareGroup', 'Deck']

features = ['Pclass', 'Sex', 'Embarked', 'Family_Size_Grouped','Title']

train_data_process = train_data[target + features]

#train_data_process = train_data_process.dropna(axis=0)

y = train_data_process[target].values.ravel()

X = pd.get_dummies(train_data_process[features])

print(X.shape)

print(y.shape)
X_test = pd.get_dummies(test_data[features])
X.head()
X_test.head()
trainX, validX, trainY, validY = train_test_split(X, y, test_size=0.2, random_state=123)
model = RandomForestClassifier(random_state=1, max_depth=9,n_estimators=100,  max_features="auto")
model.fit(trainX, trainY)
clf = model
print("OOB - ", clf.score(validX, validY))

print("whole - ", clf.score(X, y))
clf.fit(X,y) #retrain for output
print("whole(retrain) - ", clf.score(X, y))
predictions = clf.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")