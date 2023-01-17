# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# get data to dataframe variables
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

# data visualization
train_df

# data visualization
test_df
#inspecting train dataframe

train_df.shape
train_df.info()
train_df.isnull().sum()

#inspecting test dataframe

test_df.shape
test_df.info()
test_df.isnull().sum()
train_df.columns
# drop unnecessary columns: Name, Ticket, Cabin from train and test data
#Note: Survived is not present in the test data

train_df = train_df[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']]
test_df = test_df[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']]
#Checking the name of the columns in the dataframe
print(train_df.columns)
test_df.columns
#checking the null value in the train and test data
print(train_df.isnull().sum())
print(test_df.isnull().sum())
#Inspecting the datatype of the data
print(train_df.dtypes)
print(test_df.dtypes)
# Looking at the data distribution
train_df.describe()
# Looking at the data distribution
test_df.describe()
# checking the distribution of the data
train_df.hist()

#Age
train_df['Age'].hist()
train_df['Age'].describe()
# visualizing the data where Age is null
train_df_null_age = train_df[train_df['Age'].isnull()]
train_df_null_age


#comparing the histograms
train_df_null_age.hist()

train_df.hist()
# percentage of the null to the total data of the age
train_df['Age'].isnull().sum()/len(train_df.index)*100

# Want to see the survived vs Age plot
sns.barplot(x='Survived', y='Age', data=train_df,estimator=np.count_nonzero)
plt.title("Age vs Survived")
sns.boxplot(x='Survived', y='Age', data=train_df)
plt.title("Age vs Survived")
sns.violinplot(x='Survived', y='Age', data=train_df)
plt.title("Age vs Survived")
randon_age_array = np.random.randint(0,80,177)
train_df['Age'][train_df['Age'].isnull()] = randon_age_array
train_df['Age']

train_df['Age'].hist()
train_df['Age'].isnull().sum()
# coverting float to int
train_df['Age'] = train_df['Age'].astype(int)

train_df['Age'].hist(bins=70)
#Sex
train_df.groupby('Sex')['Sex'].count()
plt.subplots(1,2,figsize=(15,5))

plt.subplot(1,2,1)
sns.barplot(x='Sex', y='Survived', data=train_df,estimator=np.count_nonzero)
plt.title("Sex")

plt.subplot(1,2,2)
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title("Sex")
#Embarked
train_df[(train_df.Embarked.isnull())]
# there is not a significant null value in Embarked
# Will delete that rows from the train_df dataframe
train_df = train_df[~(train_df.Embarked.isnull())]
train_df['Embarked'].isnull().sum()
train_df.groupby('Embarked')['Embarked'].count()
plt.subplots(1,2,figsize=(15,5))

plt.subplot(1,2,1)
sns.barplot(x='Embarked', y='Survived', data=train_df,estimator=np.count_nonzero)
plt.title("Embarked")

plt.subplot(1,2,2)
sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.title("Embarked")
# Pclass
train_df.groupby('Pclass')['Pclass'].count()
plt.subplots(1,2,figsize=(15,5))

plt.subplot(1,2,1)
sns.barplot(x='Pclass', y='Survived', data=train_df,estimator=np.count_nonzero)
plt.title("Pclass")

plt.subplot(1,2,2)
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title("Pclass")
# Parch, SibSp
plt.subplots(1,2,figsize=(15,5))

plt.subplot(1,2,1)
sns.countplot(x='Parch', data=train_df)
plt.title('Parch')

plt.subplot(1,2,2)
sns.countplot(x='SibSp', data=train_df)
plt.title('SibSp')

plt.show()
# combining the columns SibSp and Parch and making one column family.
train_df['Family'] = train_df['SibSp'] + train_df['Parch']
train_df[train_df['Family'] > 0]
sns.countplot(x='Family', data=train_df)
plt.title('Family')
# replacing non zero value by 1
def decode_family(x):
    if x == 0:
        return 0
    else:
        return 1
    


train_df['Family'] = train_df.Family.apply(decode_family)
# drop unnecessary columns SibSp and Parch now
train_df = train_df.drop(['SibSp','Parch'], axis =1)
train_df.info()
plt.subplots(1,2,figsize=(15,5))

plt.subplot(1,2,1)
sns.barplot(x='Family', y='Survived', data=train_df,estimator=np.count_nonzero)
plt.title("Family")

plt.subplot(1,2,2)
sns.barplot(x='Family', y='Survived', data=train_df)
plt.title("Family")
# Do same operation on test_df

# combining the columns SibSp and Parch and making one column family.
test_df['Family'] = test_df['SibSp'] + test_df['Parch']

# drop unnecessary columns SibSp and Parch now
test_df = test_df.drop(['SibSp','Parch'], axis =1)

test_df['Family'] = test_df.Family.apply(decode_family)
test_df.Age.isnull().sum()
randon_age_array = np.random.randint(0,80,86)
test_df['Age'][test_df['Age'].isnull()] = randon_age_array
test_df['Age']
#Creating train data and test data
X_train = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked','Family']]
Y_train = train_df['Survived']

X_test =test_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family']]
print(X_train.columns)
print(X_test.columns)

sns.pairplot(train_df,hue="Survived",size = 3,diag_kind="kde")
