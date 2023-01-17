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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.feature_selection import SelectFromModel



# To ignore unwanted warnings

import warnings

warnings.filterwarnings('ignore')
# Loading the Titanic

train = pd.read_csv('../input/machine-learning-on-titanic-data-set/train.csv')

test = pd.read_csv('../input/machine-learning-on-titanic-data-set/test.csv')

gender_submission = pd.read_csv('../input/machine-learning-on-titanic-data-set/gender_submission.csv')
train.head()
test.head()
len(train)
train.isnull().sum().sort_values(ascending=False).head(4)
 #missing amount for train set

missing= train.isnull().sum().sort_values(ascending=False)

percentage = ((train.isnull().sum()/ train.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([missing, percentage], axis=1, keys=['Missing', '%Missing'])

missing_data.head(4)
train[train.Age.isnull()]
#categorical feature analysis :

ax = sns.countplot(y=train.Survived, data=test)

plt.title('Distribution of Survived')

plt.xlabel('Number of passengers')



total = len(train['Survived'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
ax = sns.countplot(y=train.Pclass, data=test)

plt.title('Distribution of  Pclass')

plt.xlabel('Number of passengers')



total = len(train['Pclass'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
ax = sns.countplot(y=test.Pclass, data=test)

plt.title('Distribution of  Pclass')

plt.xlabel('Number of passengers')



total = len(test['Pclass'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
"""GENDER DISTRIBUTION"""



ax = sns.countplot(y=train.Sex, data=test)

plt.title('Distribution of Sex (Train data)')

plt.xlabel('Number of passengers')



total = len(train['Sex'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
"""GENDER DISTRIBUTION"""



ax = sns.countplot(y=test.Sex, data=test)

plt.title('Distribution of Sex (Test data)')

plt.xlabel('Number of passengers')



total = len(test['Sex'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
"""SIBLINGS"""





ax = sns.countplot(y=train.SibSp, data=test)

plt.title('Distribution of Number of Siblings/Spouses (Train data)')

plt.xlabel('Number of passengers')



total = len(train['SibSp'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
"""SIBLINGS/SPOUSE"""



ax = sns.countplot(y=test.SibSp, data=test)

plt.title('Distribution of Number of Siblings/Spouses  (Test data)')

plt.xlabel('Number of passengers')



total = len(test['SibSp'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
"""Parent/children"""





ax = sns.countplot(y=train.Parch, data=test)

plt.title('Distribution of Number of Parent/children (Train data)')

plt.xlabel('Number of passengers')



total = len(train['Parch'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
"""Embarked"""



ax = sns.countplot(y=train.Embarked, data=test)

plt.title('Distribution of Boarding points(Train data)')

plt.xlabel('Number of passengers')



total = len(train['Embarked'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()



# Titanic started from Southampton (S), Wnt to Cherbourg (C) then finally to Queenstown (Q)
train_v2=train.copy()

test_v2=test.copy()
train_v2.head
'''cabin flag'''

train_v2['Cabin_flag']=np.where(train_v2.Cabin.isnull(),0,1)

test_v2['Cabin_flag']=np.where(test_v2.Cabin.isnull(),0,1)
"""Gender flag"""

train_v2['Gender']=np.where(train_v2.Sex=='male',1,0)

test_v2['Gender']=np.where(test_v2.Sex=='male',1,0)

"""EMBARKED FLAG"""

train_v2['EM_flag']=np.where(train_v2.Embarked=='S',1,np.where(train_v2.Embarked=='C',2,3))

test_v2['EM_flag']=np.where(test_v2.Embarked=='S',1,np.where(test_v2.Embarked=='C',2,3))
train_v2.head
"""Cabin no cabin counts"""

ax = sns.countplot(y=train_v2.Cabin_flag, data=test)

plt.title('Distribution of Boarding points(Train data)')

plt.xlabel('Number of passengers')



total = len(train_v2['Cabin_flag'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()

pd.crosstab(train_v2.Survived,train_v2.Cabin_flag)
pd.crosstab(train_v2.Survived,train_v2.Cabin_flag).apply(lambda x:x/sum(x)*100)
pd.crosstab(train_v2.Survived,train_v2.Sex).apply(lambda x:x/sum(x)*100)
pd.crosstab(train_v2.Survived,train_v2.Sex).apply(lambda x:x/sum(x)*100)
"""CORRELATION PLOT"""



x= train_v2.select_dtypes(include=[np.number])#to check only numeric columns

x.corr() 



corr = train_v2.corr()

plt.figure(figsize=(10,10)) 

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

           cmap='viridis', annot=True)

plt.show()



#High negative correl between pclass and cabin means that a passenger in 1st class facility had cabin --we can chose one out of them 

#high neg correl btween plcass and fare, good class(1st) ->good price ->we can chose one

#out of the above 3 cabin flag makes the most sense to keep because it is has better correl with the survived passengers

#strong neg correl in gender and survived indicates that more number of females survived
train_v2.columns
# splitting the data to train and Validation:



y= train_v2[['Survived']]

X= train_v2[[ 'SibSp','Parch', 'Fare', 'EM_flag', 'Cabin_flag', 'Gender']] #removed age because of null values in 18% of the data..





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
"""Logistic Regression Model Fitting"""

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train, y_train)



#Predicting the test set results and calculating the accuracy

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
"""actual test data prediction"""



X_train=train_v2[[ 'SibSp','Parch', 'Fare', 'EM_flag', 'Cabin_flag', 'Gender']]

X_test=train_v2[[ 'SibSp','Parch', 'Fare', 'EM_flag', 'Cabin_flag', 'Gender']]

y_train= train_v2[['Survived']]
"""Logistic Regression Model Fitting"""

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train, y_train)



#Predicting the test set results and calculating the accuracy

y_pred = logreg.predict(X_test)
test_v2.columns
Output=test_v2[['PassengerId']].copy()

Output['Survived']=pd.DataFrame(y_pred)

pd.DataFrame(Output).to_csv('/personal/Python Projects/Titanic/data/Predicted_log_v1.csv', index=False)