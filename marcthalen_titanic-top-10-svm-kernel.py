# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_import = pd.read_csv('../input/titanic/test.csv')

train_import = pd.read_csv('../input/titanic/train.csv')



test = np.zeros((0,0))

train = np.zeros((0,0))
# create an array in which we put the data sets that have to be cleaned

data_clean = [test_import, train_import]
# check which data is missing from the datasets



for dataset in data_clean:

    print(dataset.isnull().any())
for dataset in data_clean:

    # datasets are missing data in Age, Fare, Cabin and Embarked

    # inspect the data and clean up the data. Start with first column and work our way up

    # Age and Fare we can fill by taking the median 

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)



    # Embarked and Cabin we can fill with mode (most common)

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    dataset['Cabin'].fillna(dataset['Cabin'].mode()[0], inplace = True)
for dataset in data_clean:

    # now that the data is filled we should see if there are columns that are irrelevant

    # we can drop passengerId as its an index, also cabin and ticket are irrelevant combined with the other features that we already

    # have such as fare.

    dataset.drop(['PassengerId','Cabin', 'Ticket'], axis=1, inplace=True)
# generate new features. features inspired by kernel created by LD Freeman 

for dataset in data_clean:

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    

    # set isAlone to 0 

    dataset['IsAlone'] = 1

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 



    # split title from remainder of the name

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    

    # group non common titles

    title_names = (dataset['Title'].value_counts() < 10)

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

    

    

    # group some features down into groups

    # create 4 different fare groups

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    dataset['AgeBin'] = pd.qcut(dataset['Age'], 4)
label = LabelEncoder()



for dataset in data_clean:    

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])



    

    featuresTest = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']

    featuresTrain = ['Survived','Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']

    test = pd.get_dummies(data_clean[0][featuresTest])

    train = pd.get_dummies(data_clean[1][featuresTrain])
#normally we would split the test set but in this case we already know the model so we train on the whole set.

X_train = train.drop('Survived', axis=1)

Y_train = train['Survived'];
#Create the model and score on 

svc=make_pipeline(StandardScaler(),SVC(random_state=1))

r=[0.0001,0.001,0.1,1,10,50,100]

PSVM=[{'svc__C':r, 'svc__kernel':['linear']},

      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]

GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)

scores_svm=cross_val_score(GSSVM, X_train.astype(float), Y_train,scoring='accuracy', cv=5)

print(np.mean(scores_svm))
# import a fresh set to get the ids from

fresh_import = pd.read_csv('../input/titanic/test.csv')



#Fit the model

GSSVM.fit(X_train, Y_train)

pred=GSSVM.predict(test)

output=pd.DataFrame({'PassengerId':fresh_import['PassengerId'],'Survived':pred})

output.to_csv('submission.csv', index=False)