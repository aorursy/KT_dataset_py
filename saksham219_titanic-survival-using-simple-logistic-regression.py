# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
titanic = pd.read_csv('../input/train.csv')

titanic.head()
titanic.info()
np.median(titanic[titanic['Age'].notnull()]['Age'])
titanic['Age'] = titanic['Age'].fillna(np.median(titanic[titanic['Age'].notnull()]['Age']))
male_index = titanic['Sex'] =='male'

female_index = titanic['Sex'] == 'female'

titanic.loc[male_index,'Sex'] = 1 

titanic.loc[female_index,'Sex'] = 0
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.info()
def get_title(name):

    if '.' in name:

        a = name.split(',')[1]

        title = a.split(' ')[1]

        return(title)

    else:

        return('unknown')
titanic['title'] = titanic['Name'].apply(get_title)

titanic['title'].value_counts()
def title_to_no(title):

    if title in ['Mr.','Major.','Col.']:

        return 1

    elif title in ['Mrs.','Mme.']:

        return 2

    elif title in ['Master.']:

        return 3

    elif title in ['Miss.','Mlle.','Ms.']:

        return 4

    else:

        return 5
titanic['title'] = titanic['title'].apply(title_to_no)

titanic['title'].value_counts()
titanic.head()
sns.countplot('Survived',data=titanic,hue='Embarked')
embark = pd.get_dummies(titanic['Embarked'],drop_first=True)

titanic.drop('Embarked',axis=1,inplace=True)

titanic = pd.concat([titanic,embark],axis=1)

titanic.head()
titanic['Family'] = titanic['SibSp'] + titanic['Parch'] 
survived = titanic[titanic['Survived']==1]

died = titanic[titanic['Survived']!=1]
plt.hist([survived['Age'],titanic['Age']],bins=20,stacked =True,label=['Survived','total'])

plt.legend()
plt.hist([survived['Pclass'],titanic['Pclass']],bins=20,stacked =True,label=['Survived','total'])

plt.legend()
corrs_survived = titanic.corr()['Survived']

corrs_survived
corrmat = titanic.corr()

sns.heatmap(corrmat)

plt.hist([survived['Fare'],titanic['title']],bins=200,stacked =True,label=['Survived','total'])

plt.xlim(0,100)

plt.legend()

plt.show()
test = pd.read_csv('../input/test.csv')
test.info()
test['Age'] = test['Age'].fillna(np.median(test[test['Age'].notnull()]['Age']))

male_index = test['Sex'] =='male'

female_index = test['Sex'] == 'female'

test.loc[male_index,'Sex'] = 1 

test.loc[female_index,'Sex'] = 0

test['Embarked'] = test['Embarked'].fillna('S')

test['title'] = test['Name'].apply(get_title)

test['title'] = test['title'].apply(title_to_no)

embark = pd.get_dummies(test['Embarked'],drop_first=True)

test.drop('Embarked',axis=1,inplace=True)

test = pd.concat([test,embark],axis=1)

test['Family'] = test['SibSp'] + test['Parch'] 
test.count()
test = test.drop(['PassengerId','Name','SibSp','Parch','Fare','Ticket','Cabin'],axis = 1)

train = titanic.drop(['PassengerId','Name','SibSp','Parch','Fare','Ticket','Cabin'],axis=1)
test.head()
train.head()
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(train[['Pclass','Sex','Age','title','Q','S']],train['Survived'])

predictions = model.predict(test[['Pclass','Sex','Age','title','Q','S']])

predictions[0:10]
from sklearn.metrics import mean_squared_error

actual = pd.read_csv('../input/gendermodel.csv')

equal = actual['Survived'] == predictions

correct = len(equal[equal == True])

accuracy =  correct/len(actual)
print(accuracy *100)