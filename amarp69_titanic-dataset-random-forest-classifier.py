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
#data analysis libraries

import numpy as np

import pandas as pd



#visualization libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#ignore warnings

import warnings

warnings.filterwarnings('ignore')





from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler,RobustScaler

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix 

from sklearn.naive_bayes import GaussianNB 

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn import model_selection

from sklearn.svm import LinearSVC

from sklearn import metrics

from sklearn.svm import SVC
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
train.Sex=train.Sex.astype('category').cat.codes

test.Sex=test.Sex.astype('category').cat.codes
train.info()
test.info()
#checking for total null values

train.isnull().sum() 
#checking for total null values

test.isnull().sum() 
# drop the Cabin column becuse more the 77% data missing 

# drop Name and Ticket column as well 

train.drop(labels = ["Cabin","Name","Ticket"], axis=1, inplace=True) 

test.drop(labels = ["Cabin","Name","Ticket"], axis=1, inplace=True) 



# fill Age null value with median

train['Age'].fillna(train['Age'].mean(), inplace=True) 

test['Age'].fillna(train['Age'].mean(), inplace=True) 

test['Fare'].fillna(test['Fare'].mean(),inplace=True)



# drop any missing value

train=train.dropna()  



train.head()
test.head()
# Visualizations of Feature vs. Target

fig=plt.figure()

ax1=plt.subplot(321)

sns.countplot(x = 'Survived', hue = 'Sex', data = train, ax=ax1)



ax2=plt.subplot(322)

sns.countplot(x = 'Survived', hue = 'Pclass', data = train, ax=ax2)



ax3=plt.subplot(323)

sns.countplot(x = 'Survived', hue = 'SibSp', data = train, ax=ax3)

ax3.legend(loc=1, title='Sibling/Spouse Count', fontsize='x-small')



ax4=plt.subplot(324, sharey=ax3)

sns.countplot(x = 'Survived', hue = 'Parch', data = train, ax=ax4)

ax4.legend(loc=1, title='Parent/Children Count', fontsize='x-small')



ax5=plt.subplot(325)

sns.countplot(x = 'Survived', hue = 'Embarked', data = train, ax=ax5)

ax5.legend(loc=1, title='Embarked')



fig.set_size_inches(8,12)

fig.show();
#sort the ages into logical categories

bins = [0, 5, 12, 18, 24, 35, 60, 80]

labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

fig1=plt.figure(figsize=(10,5))

sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show();
tab1=pd.crosstab(train.Pclass,train.Survived,margins=True)

print(tab1)

print("----------------------------------------------")

tab2=pd.crosstab(train.Sex,train.Survived,margins=True)

print(tab2)
# Encoding Catagorical Values

train.drop(labels = ["AgeGroup"], axis=1, inplace=True) 

train_df=train.copy()

test_df=test.copy()

train_df = pd.get_dummies(train_df, columns=['Embarked', 'Pclass'], drop_first=True)

test_df = pd.get_dummies(test_df, columns=['Embarked', 'Pclass'], drop_first=True)



train_df.head()
# Correlations

fig=plt.figure(figsize=(8,8))

sns.heatmap(train_df.corr(), annot=True, cbar_kws={'label': 'Correlation coeff.'}, cmap="RdBu")

fig.show();
# Seperating the i/p features from the target variable



X = train_df.drop('Survived', axis=1)  

y = train_df['Survived']



# Prepare an array with all the algorithms

models = []

models.append(('LR', LogisticRegression(solver='liblinear'))) # uses default parameters

models.append(('KNC', KNeighborsClassifier())) # uses default parameters

models.append(('NB', GaussianNB())) # uses default parameters

models.append(('SVC', SVC())) # uses default parameters

models.append(('LSVC', LinearSVC())) # uses default parameters

models.append(('RFC', RandomForestClassifier())) # default n_estimators = 100

models.append(('DTC', DecisionTreeClassifier())) # uses default parameters

models.append(('GBC',(GradientBoostingClassifier()))) # uses default parameters



seed = 10

results = []  # to cross_validation results of each Model

names = []  # to hold the names of the Model



# Scale the i/p feature_set

ss = StandardScaler()

X_scaled = ss.fit_transform(X)



# Run all models and print scores

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed,  shuffle = True)

    cv_results = model_selection.cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = round(cv_results.mean()* 100,2)

    print(f'{name} : {msg}')
from sklearn.model_selection import cross_val_score



X=train_df.drop('Survived',axis=1)

y=train_df['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



RFG=RandomForestClassifier()



RFG.fit(X_train,y_train)

prediction=RFG.predict(X_test)

score=cross_val_score(RFG,X_train,y_train,cv=5)

print("Confusion_matrix:",confusion_matrix(y_test,prediction))

acc_log = round(RFG.score(X_train,y_train) * 100, 2)

acc_log
test_df['Survived'] = RFG.predict(test_df)



test_df.head()
#Export as csv



result = test_df[["PassengerId","Survived"]]

result.to_csv('Titanic-results.csv', index=False, header=True)

print("Your submission was successfully saved!")