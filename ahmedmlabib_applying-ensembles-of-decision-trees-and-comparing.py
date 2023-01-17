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
#import the libraries that we are gonna use to visualize our data

import seaborn as sns

import matplotlib.pyplot as plt 

import cufflinks as cf

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.express as px

cf.go_offline()

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import train_test_split

%matplotlib inline
# import the 2 algorithms that we will use and compare between their performance

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
# Load in the train and test datasets

train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

# Store our passenger ID for easy access

PassengerId = test['PassengerId']



train.head()
test.head()
#now let's see what are the featurres that have missing data and try to fill it

train.info()
train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# to fill the data missed inside we wil check for the correlation between the age feature and the rest of features

train.corr()
px.box(x='Sex',y='Age',data_frame=train.dropna())
def impute_age(cols):

    Age=cols[0]

    Sex=cols[1]

    if pd.isnull(Age):



        if Sex == 'male':

            return 29



        else:

            return 27



    else:

        return Age
train['Age'] = train[['Age','Sex']].apply(impute_age,axis=1)

test['Age']=test[['Age','Sex']].apply(impute_age,axis=1)
train.isnull().sum().sort_values(ascending = False)
test.isnull().sum().sort_values(ascending = False)
train['Embarked'].value_counts()
train['Embarked']=train['Embarked'].fillna('S')

test['Fare']=test['Fare'].fillna(test['Fare'].median())
test.head()
#Deal with categorical data and extract information from it

datasets=[train,test]

for dataset in datasets:

    # Mapping Sex and Embarked

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1,'Q' : 2} ).astype(int)
#train.head()

#For the name we can extract the title because it represents an add-feature for our data instead of ignoring the name feature totaly

def getTitles(name):

    name = str(name)

    title = name.split('.')[0]

    title = title.split(',')

    return str(title[1])

for dataset in datasets:

    dataset['Title']=dataset['Name'].apply(getTitles)
train['Title'].value_counts()
def cleanTitle(title):

    if title in [' Mr',' Mrs',' Master',' Miss',' Mlle',' Ms',' Mme']:

        if title in [' Mrs',' Mme']:

            return 'Mrs'

        elif title in[' Miss',' Mlle',' Ms']:

            return 'Miss'

        else:

            return title.strip()      

    else:

        return "Other"

for dataset in datasets:

    dataset['Title'] = dataset['Title'].apply(cleanTitle)

    
#train['Title'].value_counts()

#test['Title'].value_counts()

for dataset in datasets:

    dataset['Title'] = dataset['Title'].map( {'Mr': 0, 'Miss': 1,'Mrs' : 2,'Master' : 3,

                                             'Other' : 4} ).astype(int)

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['Is_Alone']=(dataset['FamilySize']>1)

    dataset['Is_Alone']=dataset['Is_Alone'].map({False: 0, True: 1})
for dataset in datasets:

     dataset.drop(['PassengerId','Name','Ticket','SibSp'], axis = 1,inplace=True)
test.head()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

train_data = sc_X.fit_transform(train.drop('Survived',axis=1))

test_data = sc_X.transform(test)
X_train, X_test, y_train, y_test = train_test_split(train_data, 

                                                    train['Survived'], random_state=101,test_size=0.20)

forest = RandomForestClassifier(n_estimators=400,random_state=40,max_depth=10,max_features=4)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train))) 

print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

y_pred=forest.predict(X_test)

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
#to get the importance of each feature in your data 

features = list(train.columns)

importances = forest.feature_importances_

indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='g')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
test['Survived']=forest.predict(test_data)

test['PassengerId']=PassengerId 

test[['PassengerId', 'Survived']].to_csv('kaggle_submission.csv', index = False)