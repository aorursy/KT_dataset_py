# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#load data as dataframes

test_data = pd.read_csv('../input/test.csv',index_col=False)

train_data = pd.read_csv ('../input/train.csv', index_col=False)
#test_data sample

train_data.head()
#convert names to titles for easier segregation

for index, row in train_data.iterrows():

    obj = re.search("^.*, (.*?)\\..*$",row['Name'])

    title = obj.group(1)

    if title in ['Capt','Col','Major','Dr','Rev','Don','Sir','the Countess','Jonkheer']:

        title = 'Officer'

    train_data = train_data.set_value(index,'Title',title)

    

print(train_data['Title'].value_counts())
#reclubbing some title categories

train_data['Title'].replace('Ms', 'Miss',inplace=True)

train_data['Title'].replace('Mme', 'Mrs',inplace=True)

train_data['Title'].replace('Lady', 'Miss',inplace=True)

train_data['Title'].replace('Dona', 'Miss',inplace=True)

train_data['Title'].replace('Mlle', 'Miss',inplace=True)



print(train_data['Title'].value_counts())
def make_pivot (param1, param2):

    df = train_data

    df_slice = df[[param1, param2, 'PassengerId']]

    slice_pivot = df_slice.pivot_table(index=[param1], columns=[param2],aggfunc=np.size, fill_value=0)

    p_chart = slice_pivot.div(slice_pivot.sum(axis=1), axis=0).plot.bar(stacked=True)

    return slice_pivot

    return p_chart
#some pretty visulisations

make_pivot('Title','Survived')
make_pivot('Sex','Survived')
make_pivot('Parch','Survived')
make_pivot('SibSp','Survived')
#quantising the data

train_data['Sex'].replace('male', 1,inplace=True)

train_data['Sex'].replace('female', 0,inplace=True)

train_data['Title'].replace('Mr',1,inplace=True)

train_data['Title'].replace('Mrs',2,inplace=True)

train_data['Title'].replace('Miss',3,inplace=True)

train_data['Title'].replace('Master',4,inplace=True)

train_data['Title'].replace('Officer',5,inplace=True)

#classifying data into test and train

X_train = train_data[0:500][['Pclass','Sex','SibSp','Parch','Title']]

y_train = train_data[0:500]['Survived']

X_test = train_data[500:][['Pclass','Sex','SibSp','Parch','Title']]

y_test = train_data[500:]['Survived']

#print(X_train.head(30))



clf = RandomForestClassifier()

clf.fit(X_train, y_train)

#testing accuracy

clf.score(X_test,y_test)
#preprocessing test_data

for index, row in test_data.iterrows():

    obj = re.search("^.*, (.*?)\\..*$",row['Name'])

    title = obj.group(1)

    if title in ['Capt','Col','Major','Dr','Rev','Don','Sir','the Countess','Jonkheer']:

        title = 'Officer'

    test_data = test_data.set_value(index,'Title',title)



#reclubbing some title categories

test_data['Title'].replace('Ms', 'Miss',inplace=True)

test_data['Title'].replace('Mme', 'Mrs',inplace=True)

test_data['Title'].replace('Lady', 'Miss',inplace=True)

test_data['Title'].replace('Dona', 'Miss',inplace=True)

test_data['Title'].replace('Mlle', 'Miss',inplace=True)



#quantising the data

test_data['Sex'].replace('male', 1,inplace=True)

test_data['Sex'].replace('female', 0,inplace=True)

test_data['Title'].replace('Mr',1,inplace=True)

test_data['Title'].replace('Mrs',2,inplace=True)

test_data['Title'].replace('Miss',3,inplace=True)

test_data['Title'].replace('Master',4,inplace=True)

test_data['Title'].replace('Officer',5,inplace=True)



X_test_final = test_data[['Pclass','Sex','SibSp','Parch','Title']]
#applying on test_data

y_test_final = clf.predict(X_test_final)

test_data['Survived'] = y_test_final

gensubmission = test_data[['PassengerId', 'Survived']].copy()

gensubmission.Survived = gensubmission.Survived.astype(int)

gensubmission.to_csv('./DSsubmission.csv',index=False)