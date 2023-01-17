# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



print("Output files path")        

for dirname, _, filenames in os.walk('/kaggle/output'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



def sigmoid(z):

    return 1/(1 + np.exp(-z))
data_train = pd.read_csv("/kaggle/input/titanic/train.csv");

data_test = pd.read_csv("/kaggle/input/titanic/test.csv");



data_train = data_train[pd.notnull(data_train['Survived'])]

data_train = data_train[pd.notnull(data_train['Age'])]

data_train = data_train[pd.notnull(data_train['Fare'])]

data_train = data_train[pd.notnull(data_train['Pclass'])]

data_train = data_train[pd.notnull(data_train['Sex'])]

data_train['SexN'] = data_train['Sex'].map({'female': 1, 'male': 0})



#data_test = data_test[pd.notnull(data_test['Age'])]

data_test['Age'].fillna(data_test['Age'].mean(), inplace=True)

#data_test = data_test[pd.notnull(data_test['Fare'])]

data_test['Fare'].fillna(data_test['Fare'].mean(), inplace=True)

#data_test = data_test[pd.notnull(data_test['Pclass'])]

data_test['Pclass'].fillna(data_test['Pclass'].mean(), inplace=True)

#data_test = data_test[pd.notnull(data_test['Sex'])]

#data_test['Sex'].fillna(data_test['Sex'].mean(), inplace=True)



data_test['SexN'] = data_test['Sex'].map({'female': 1, 'male': 0})



my_data = data_train[['Pclass' , 'Fare', 'Age', 'SexN', 'Survived']]

my_data_test = data_test[['Pclass' , 'Fare', 'Age', 'SexN']] 

my_data_test_with_ID = data_test[['PassengerId','Pclass' , 'Fare', 'Age', 'SexN']] 

X = my_data[my_data.columns[1:5]].values;

Y = my_data[['Survived']].values.flatten()



X_test = my_data_test.values

X_test_with_ID = my_data_test_with_ID.values



clf = tree.DecisionTreeClassifier(random_state=123);

clf = clf.fit(X,Y);



clf.feature_importances_
print(len(X_test))

print(len(data_test))
the_prediction = clf.predict(X_test)

the_result = []



for index, record in data_test.iterrows():

    the_id = record[0]

    temp = []

    temp.append(the_id)

    temp.append(the_prediction[index])

    the_result.append(temp)



f = open("gender_submission_3.csv", "a")

f.write('PassengerId,Survived\n')

for record in the_result:

    the_id = record[0]

    predicted = record[1]

    f.write(str(int(the_id)) + ',' + str(int(predicted)) + '\n' )

f.close()