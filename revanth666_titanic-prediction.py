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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sb

from collections import Counter

import warnings

warnings.filterwarnings("ignore")
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
print('Train data size',train_data.shape)

print('Test data size',test_data.shape)
train_data.info()
train_data.isnull().sum()
train_data.describe()
train_data['Pclass'].value_counts()
train_data['Survived'].value_counts()
Counter(train_data['Sex'])
train_data['SibSp'].value_counts()
train_data['Parch'].value_counts()
train_data['Embarked'].value_counts()
#train_data = df

#df.head()

df = pd.DataFrame(train_data)

df.head()
rcParams['figure.figsize'] = 10,5

sb.barplot(x = df['Survived'].value_counts().index, y = df['Survived'].value_counts().values)

plt.title('Survival counts')

plt.xlabel('Survived')

plt.ylabel('No of passengers')

plt.show()
rcParams['figure.figsize'] = 10,5

sb.barplot(x = df['Pclass'].value_counts().index, y = df['Pclass'].value_counts().values)

plt.title('Types of passenger class')

plt.xlabel('Class')

plt.ylabel('No of passengers')

plt.show()
rcParams['figure.figsize'] = 10,5

sb.barplot(x = df['Sex'].value_counts().index, y = df['Sex'].value_counts().values)

plt.title('Male and Female counts')

plt.xlabel('Counts')

plt.ylabel('No of passengers')

plt.show()
gender = pd.crosstab(df['Survived'],df['Sex'])

gender
gender.plot(kind="bar",title='No of passengers survived')

plt.show()
rcParams['figure.figsize'] = 10,5

sb.barplot(x = df['Embarked'].value_counts().index, y = df['Embarked'].value_counts().values)

plt.title('port')

plt.xlabel('count')

plt.ylabel('No of passengers')

plt.show()
rcParams['figure.figsize'] = 10,5

sb.barplot(x = df['SibSp'].value_counts().index, y = df['SibSp'].value_counts().values)

plt.title('Number of siblings/spouses aboard')

plt.xlabel('count')

plt.ylabel('No of passengers')

plt.show()
rcParams['figure.figsize'] = 10,5

sb.barplot(x = df['Parch'].value_counts().index, y = df['Parch'].value_counts().values)

plt.title('Number of parents/childrens aboard')

plt.xlabel('count')

plt.ylabel('No of passengers')

plt.show()
rcParams['figure.figsize'] = 10,5

sb.countplot(x = 'Survived',hue = 'Pclass',data = df)

plt.show()
rcParams['figure.figsize'] = 10,5

sb.countplot(x = 'Survived', hue = 'Embarked', data = df)

plt.show()
rcParams['figure.figsize'] = 10,5

sb.countplot(x = 'Survived', hue = 'SibSp', data = df)

plt.show()
rcParams['figure.figsize'] = 10,5

sb.countplot(x = 'Survived', hue = 'Parch', data = df)

plt.show()
rcParams['figure.figsize'] = 10,5

#plt.hist(df['Age'],bins =15,alpha = 0.9)

ax = df['Age'].hist(bins = 15,alpha = 0.9, color = 'green')

ax.set(xlabel = 'Age',ylabel = 'Count',title = 'Visualization of Ages')

plt.show()
rcParams['figure.figsize'] = 10,10

sb.heatmap(df.corr(),annot = True,square = True,linewidths = 2,linecolor = 'black')
delete = ['Ticket','Cabin','Name','Fare','Embarked']
train = train_data.drop(delete,axis = 1)
train.head()
test = test_data.drop(delete,axis =1)
test.head()
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
train['Sex'] = enc.fit_transform(train['Sex'])
train.head(2)
test['Sex'] = enc.fit_transform(test['Sex'])
test.head(2)
train.dtypes
train.isnull().sum()
train.fillna(train['Age'].median(),inplace = True)
train.isnull().sum()
test.fillna(train['Age'].median(),inplace = True)
test.isnull().sum()
test.set_index(['PassengerId'],inplace = True)
test.head(2)
train.set_index(['PassengerId'],inplace = True)
train.head()
X = train[['Pclass','Sex','Age','SibSp','Parch']]

y = train.Survived
X.head(2)
y.head(2)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB,MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from sklearn import model_selection

from sklearn.model_selection import train_test_split,cross_val_score,validation_curve,KFold
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 10,test_size=0.25)
models = []

models.append(('LG', LogisticRegression()))

models.append(('SVC', SVC()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('RFC', RandomForestClassifier()))

models.append(('KNC', KNeighborsClassifier()))

models.append(('MLP', MLPClassifier()))

models.append(('XGB-TREE', XGBClassifier(booster='gbtree')))

models.append(('XGB-DART', XGBClassifier(booster='dart')))

models.append(('GNB', GaussianNB()))
seed = 10

results = []

names = []

output = []

score = 'accuracy'

for name,model in models:

    kfold = model_selection.KFold(n_splits = 5,random_state = seed)

    result = model_selection.cross_val_score(model,np.array(X_train),np.array(y_train),cv=kfold,scoring=score)

    results.append(result)

    names.append(name)

    values = name,result.mean()

    output.append(values)

print(output)
rcParams['figure.figsize'] = 10,5

fig = plt.figure()

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,roc_curve
accuracy_score(y_test,y_predict).round(4)*100
pd.crosstab(y_test,y_predict)
print(classification_report(y_test,y_predict))
auc = roc_auc_score(y_test,y_predict)

print('XGB AUC : %.2f'%auc)
rf_fpr,rf_tpr,_ = roc_curve(y_test,y_predict)

plt.plot(rf_fpr,rf_tpr,marker='_',label = 'XGB')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend()

plt.show()
test_predict = model.predict(test)
test_predict = pd.Series(test_predict)
test.reset_index(inplace = True)
predict = test['PassengerId']
predict = pd.concat([predict,test_predict], axis=1)
predict.rename(columns={0: "Survived"},inplace=True)
predict.to_csv("submission.csv",index=False)
sb.countplot(predict.Survived)