import numpy as np                                                 # Implemennts milti-dimensional array and matrices

import pandas as pd                                                # For data manipulation and analysis

#import pandas_profiling

import matplotlib.pyplot as plt                                    # Plotting library for Python programming language and it's numerical mathematics extension NumPy

import seaborn as sns                                              # Provides a high level interface for drawing attractive and informative statistical graphics

%matplotlib inline

sns.set()

import warnings

warnings.filterwarnings('ignore')





pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 200)

#pd.reset_option('^display.', silent=True)
data = pd.read_csv("../input/titanic-machine-learning-from-disaster/train.csv")     # Importing training dataset using pd.read_csv
data.head(10)
data.info()
# Finding the distribution of Embarked across unique values

data.groupby(['Embarked'])['Embarked'].count()
data.Embarked.mode()
# Finding the mode of Embarked

data['Embarked'].mode()[0]
# Filling Null values of Embarked with its mode

data.Embarked = data.Embarked.fillna(data['Embarked'].mode())
data.info()
data.Age.fillna(data.Age.median(), inplace = True)

data.Fare.fillna(data.Fare.median(), inplace = True)
data.drop('Cabin', axis = 1,inplace = True)
data.head(10)
# Creating a new feature/column : "Family Size". By adding the number of children and parents

data['FamilySize'] = data['SibSp'] + data['Parch']+1
data.head()
drop_cols = ['Name','Ticket','SibSp','Parch','PassengerId']
data.drop(drop_cols, axis = 1, inplace=True)

data.head(10)
# how many people survived

data['Survived'].sum()
data.shape[0]
# % of people survived

data['Survived'].sum()/data.shape[0]
# what is the distribution of male and female

data.groupby(['Sex'])['Sex'].count().sort_values(ascending=False)#.plot('bar')
# From where did the people embark

data.groupby(['Embarked'])['Embarked'].count().sort_values(ascending=False)
# How many people stayed in each of the classes

data.groupby(['Pclass'])['Pclass'].count().sort_values(ascending=False)
data.head()
data.groupby(['Embarked'])['Survived'].sum().sort_values(ascending=False)
data.groupby(['Embarked'])['Survived'].count()
data.FamilySize.sum()
(data.groupby(['Embarked'])['Survived'].sum()/data.groupby(['Embarked'])['Survived'].count()).plot(kind='bar')
data.groupby(['Pclass'])['Survived'].sum()
(data.groupby(['Pclass'])['Survived'].sum()/data.groupby(['Pclass'])['Survived'].count()).plot(kind='bar')
data['Fare'].mean()
#data['combination'] = data['Embarked'].astype(str) + '-' + data['Pclass'].astype(str)
data.head()
data.groupby(['Survived'])['Fare'].mean()
data.groupby(['Survived'])['Age'].mean()
sns.pairplot(data[["Fare","Age","Pclass","Survived"]],vars = ["Fare","Age","Pclass"],\

                                                             hue="Survived", dropna=True, height = 3, aspect = 1.5)

plt.title('Pair Plot')
data.head()
cat_cols = ['Sex','Embarked']
data_with_dummies = pd.get_dummies(data, columns=cat_cols, drop_first=True)
data_with_dummies.head()
corr = data_with_dummies.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr, square = True, annot = True,vmin=-1, vmax=1) #  vmin=-1, vmax=1

plt.title('Correlation between features')
data_with_dummies.head()
data_with_dummies.columns
features = ['Pclass', 'Age', 'Fare', 'FamilySize', 'Sex_male','Embarked_Q', 'Embarked_S']



target = ['Survived']
X = data_with_dummies[features]

y = data_with_dummies[target]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print('Train cases as below')

print('X_train shape: ',X_train.shape)

print('y_train shape: ',y_train.shape)

print('\nTest cases as below')

print('X_test shape: ',X_test.shape)

print('y_test shape: ',y_test.shape)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)
y_pred_train = logreg.predict(X_train)  
print(y_pred_train)
y_pred_test = logreg.predict(X_test)                 # make predictions on the testing set
print(y_pred_test)
probabilities = logreg.predict_proba(X_test)
print(probabilities)
type(probabilities)
probabilities_1 = probabilities[:,1]

probabilities_0 = probabilities[:,0]
print(probabilities_1)
from sklearn.metrics import accuracy_score

print('Accuracy score for test data is:', accuracy_score(y_test,y_pred_test))
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred_test)

print(conf_matrix)

c_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred_test))



print(c_matrix)
c_matrix.index = ['Actual Died','Actual Survived']

c_matrix.columns = ['Predicted Died','Predicted Survived']

print(c_matrix)