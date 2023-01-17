# Load in libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

%matplotlib inline
# Import train.csv and preview
df_train = pd.read_csv('../input/train.csv')
df_train.head()
# Import test.csv and preview
df_test = pd.read_csv('../input/test.csv')
df_test.head()
print(f"train dataset: {df_train.isnull().sum()}")
print(f"Total number of row is {df_train.shape[0]}")
print(f"test dataset: {df_test.isnull().sum()}")
print(f"Total number of row is {df_test.shape[0]}")
# Check data distribition of Age columns
fig = plt.figure(figsize=(5,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
df_train['Age'].hist(ax =ax1)
df_test['Age'].hist(ax =ax2)
# Impute medians for missing values in age columns
df_train['Age'][df_train['Age'].isnull() == True ] = df_train['Age'].median()
df_test['Age'][df_test['Age'].isnull() == True ] = df_test['Age'].median()
# Compare child survavl rate than adult male and female
df_train['Identity'] = df_train['Sex']
df_train['Identity'] [df_train['Age'] < 16]='child'

df_test['Identity'] = df_test['Sex']
df_test['Identity'] [df_test['Age'] < 16]='child'

# Identity visualization
sns.barplot(x = 'Identity', y = 'Survived', data=df_train)
# Created new columns 'Minor' to mark child
df_train['Minor']=df_train['Age'].astype(int)
df_train['Minor'][df_train['Age'] <= 16]=1
df_train['Minor'][df_train['Age'] > 16]=0

df_test['Minor']=df_test['Age'].astype(int)
df_test['Minor'][df_test['Age'] <= 16]=1
df_test['Minor'][df_test['Age'] > 16]=0
# Check data distribition
sns.countplot(df_train['Embarked'])
df_train['Embarked'][df_train['Embarked'].isnull() == True] = 'S'
# Fare for df_test
df_test['Fare'].hist()
df_test['Fare'][df_test['Fare'].isnull() == True]=df_test['Fare'].median()
# Fare
sns.kdeplot(df_train['Fare'][df_train['Survived'] == 1], shade=True)
sns.kdeplot(df_train['Fare'][df_train['Survived'] == 0], shade=True)
plt.legend(['Survived', 'Died'])
# Family
# We can have only one column to describe if the passagers traveled with families

df_train['Family'] = df_train["Parch"] + df_train["SibSp"]
df_train['Family'][df_train['Family'] > 0] = 1
df_train['Family'][df_train['Family'] == 0] = 0

df_test['Family'] = df_test["Parch"] + df_test["SibSp"]
df_test['Family'][df_test['Family'] > 0] = 1
df_test['Family'][df_test['Family'] == 0] = 0
# Pclass
sns.barplot(x = 'Pclass', y = 'Survived', data=df_train)
# Sex
df_train['Sex'][df_train['Sex'] == 'male'] = 1
df_train['Sex'][df_train['Sex'] == 'female'] = 0

df_test['Sex'][df_test['Sex'] == 'male'] = 1
df_test['Sex'][df_test['Sex'] == 'female'] = 0
df_train1=pd.get_dummies(df_train, columns=['Pclass'])
df_train2=pd.get_dummies(df_train1, columns=['Embarked'])
df_train = df_train2
df_test1=pd.get_dummies(df_test, columns=['Pclass'])
df_test2=pd.get_dummies(df_test1, columns=['Embarked'])
df_test = df_test2
# Drop columns 'Ticket', 'Name', 'Cabin', 'Parch', 'SibSp','Identity'
df_train = df_train.drop(['Ticket', 'Name', 'Cabin', 'Parch', 'SibSp','Identity'], axis = 1)
df_test = df_test.drop(['Ticket', 'Name', 'Cabin', 'Parch', 'SibSp', 'Identity'], axis = 1)

# Check cleaning result
print(f"train dataframe: {df_train.isnull().sum()}")
print(f"test dataframe: {df_test.isnull().sum()}")
#train data
df_train.head()
#test data
df_test.head()
cols = ['Sex','Age','Fare','Minor','Family','Pclass_1','Pclass_2','Pclass_3','Embarked_C','Embarked_Q','Embarked_S']
x_train = df_train[cols]
y_train = df_train["Survived"]
x_test  = df_test[cols]
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
logreg.score(x_train, y_train)
