import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

train.head()
test = pd.read_csv('../input/titanic/test.csv')

test.head()
train.shape # shape of the training data
test.shape # shape of the test data
train.isnull().sum() # check the count of missing values in the train data
train.isnull().sum() # check the count of missing values in the test data
import missingno as msno 

msno.bar(train, color = 'steelblue')                  #Plot bar for train data
# Check the column which has more the 30% missing observation in training data

col = train.isnull().sum() > int(0.30 * train.shape[0]) 

col
# Check the column which has more the 30% missing observation in test data

col = train.isnull().sum() > int(0.30 * test.shape[0]) 

col
col_drop = ['Cabin']

# Drop the columns 'cabin' from training and test data as it contains more than 30% missing values

train.drop(col_drop, axis = 1, inplace = True)

test.drop(col_drop, axis = 1, inplace = True)
sns.set_style('dark')

sns.distplot(train['Age'].dropna(),hist = True, kde = True, bins = 50)
df = pd.DataFrame(train['Age'])

df.describe()
skew = (3 *(29.699 - 28))/14.5264  # Skewness = 3(mean - median)/SD

skew
train["Age"].fillna(train["Age"].median(), inplace = True)

test["Age"].fillna(test["Age"].median(), inplace = True)

train["Embarked"].fillna(train["Embarked"].mode()[0], inplace = True)  # 0 index to get mode the column

test["Embarked"].fillna(test["Embarked"].mode(), inplace = True)      # Data is categorical

test["Fare"].fillna(test["Fare"].mean(), inplace = True) # Only few values are missing which can be replaced with mean

sns.heatmap(train.isnull(),cmap = 'magma' )
sns.set_style('darkgrid')

sns.countplot(x = 'Survived',data = train, palette = 'deep')
sns.set_style('darkgrid')

sns.countplot(x = 'Survived', hue = 'Sex' ,data = train, palette = 'deep')
sns.set_style('darkgrid')

sns.countplot(x = 'Survived', hue = 'Pclass' ,data = train, palette = 'cubehelix')
cor = train.corr()

sns.heatmap(cor, annot = True)
train.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Sex1'] = le.fit_transform(train.Sex)
train['Embarked1'] = le.fit_transform(train.Embarked)
train.drop(['Name','Sex','Ticket','Embarked'],axis = 1,inplace = True)

train.rename(columns={"Sex1": "Sex", "Embarked1": "Embarked"},inplace = True)
train.head(5)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
train.drop('Survived',axis=1).head()
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis = 1), train['Survived'], test_size = 0.30, random_state = 100)
logisticModel = LogisticRegression()

logisticModel.fit(X_train,y_train)

import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)  #To suppress warning 
predict = logisticModel.predict(X_test)
confusionmatrix = confusion_matrix(y_test,predict)

confusionmatrix
accuracy=accuracy_score(y_test,predict)

accuracy