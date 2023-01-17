## Hi all..I am very new to Kaggle any suggestions would be very helpful
## In my this version of notebook I am focussed on Exploratory Data Analysis and Feature Engineering
## The remaining part that deals with  Predictive Modelling will upload in upcoming Versions*
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics 
from sklearn.metrics import confusion_matrix,roc_auc_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/train.csv")
test_data  = pd.read_csv("/kaggle/input/test.csv")
train_data.shape
test_data.shape
train_data.columns
train_data.head()
train_data.describe(include = 'all')
train_data.info()
## Checking the NA values in the dataset
train_data.isnull().sum()
(687/891) *100 ## So Cabin has 77% of missing values So we can drop it for further analysis as most of the data is missing.
## Seaborn Relplot also gives good visualization
## A Relplot is a Figure-level interface for drawing relational plots
sns.relplot(x="Sex", y="Age", hue="Survived", data=train_data)
## Rechecking using statistics
x = train_data.Sex[train_data.Survived == 1]
x.value_counts()
## We will plot Seaborn FacetGrid which is Multi-plot grid for plotting conditional relationships.

## We will plot it using vars Sex,Embark,Survived and PClass
bins = np.arange(0, 65, 5)
g = sns.FacetGrid(train_data, row="Embarked")
g = g.map(sns.pointplot, 'Pclass', 'Survived','Sex', palette=None,  order=None, hue_order=None )
g.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train_data)
##Checking the Class Distribution wrt Survived
x1 = train_data.Survived[train_data.Pclass == 3]
x2 = train_data.Survived[train_data.Pclass == 2]
x3 = train_data.Survived[train_data.Pclass == 1]

x3.value_counts()
x2.value_counts()
x1.value_counts()
## It can be seen from the above statistics that most of the people belonging to Class 3 did not survive
## It can also be rechecked by plotting FacetGrid as below ->

grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
## It can be seen that Pclass 3 has very less rate of survival
## Need to check relation between SibSp and Survived and relation between Parch and Survived
## We will check it by plotting the Seaborn Barplot
sns.barplot(y = 'Survived', x = 'SibSp',data = train_data, edgecolor = 'w')
plt.show()

sns.barplot(y = 'Survived', x = 'Parch',data = train_data, edgecolor = 'w')
plt.show()

grid1 = sns.FacetGrid(train_data, col='Survived', row='SibSp', size=2.2, aspect=1.6)
grid1.map(plt.hist, 'Age', alpha=.5, bins=20)
grid1.add_legend();
## It can be seen that the Survival is better for Sibsp = 0 whereas Sibsp with value 3,4,5 have least and SibSp with value as 8 no survival
sns.relplot(x="SibSp", y="Parch", hue="Survived", data=train_data)
grid2 = sns.FacetGrid(train_data, col='Survived', row='Parch', size=2.2, aspect=1.6)
grid2.map(plt.hist, 'Age', alpha=.5, bins=20)
grid2.add_legend();
train_data.isnull().sum()
## Age,Cabin, Embarked have missing values in train data
test_data.isnull().sum()
## Age,Fare,Cabin have missin values
data = [train_data,test_data]
train_data.Embarked.value_counts()
## Most Commom value is 'S' so fillna with 'S'
train_data['Embarked'] = train_data['Embarked'].fillna('S')
train_data.Embarked.value_counts()
test_data.Fare.describe()
## So we will fill the missing value with the mean
test_data['Fare'] = test_data['Fare'].fillna(35.627188)
test_data.Fare.isnull().sum()
train_data.Age.describe()
## In both train_data and test_data Age attribute has missing values 
##We will fill na with the mean Age in both the data
mean_age = train_data.Age.mean()
mean_age
train_data['Age'] = train_data['Age'].fillna(mean_age)
train_data.Age.isnull().sum()
test_data.Age.describe()
mean_age1 = test_data.Age.mean()
mean_age1
test_data['Age'] = test_data['Age'].fillna(mean_age1)
test_data.Age.isnull().sum()
## Now the attribute Cabin has the maximum nbr of missing values 
## In Cabin the letter refers to the Deck which might be helpful for Survived attr as in it might be the case 
## that people belonging to certain Deck have higher rate of survival
train_data.Cabin.describe()
train_data.Cabin.value_counts()
train_data['Cabin'] = train_data['Cabin'].str.extract('(\w)')

train_data.Cabin.value_counts()
train_data.Cabin.describe()
## Lets plot the relation between Cabin and Survived
grid3 = sns.FacetGrid(train_data, col='Survived', row='Cabin', size=2.2, aspect=1.6)
grid3.map(plt.hist, 'Age', alpha=.5, bins=20)
grid3.add_legend();
## It can be seen from the above Graph that Passengers with Deck no as A and F have higher rate for survival
test_data.Cabin.describe()
test_data['Cabin'] = test_data['Cabin'].str.extract('(\w)')
test_data.Cabin.value_counts()
test_data.Cabin.describe()
## Lets fill na for Cabin in both the data with value as 'T'
train_data['Cabin'] = train_data['Cabin'].fillna('T')
test_data['Cabin'] = test_data['Cabin'].fillna('T')
train_data.Cabin.value_counts()
test_data.Cabin.value_counts()
train_data.describe(include = 'all')
## Here the vars Sex,Cabin and Embarked need to be converted to numeric categorical
## So we will convert them using LabelEncoder object
le = LabelEncoder()

train_data.Sex       = le.fit_transform(train_data.Sex)
train_data.Cabin     = le.fit_transform(train_data.Cabin)
train_data.Embarked  = le.fit_transform(train_data.Embarked)

train_data.Sex       = train_data.Sex.astype('category')
train_data.Cabin     = train_data.Cabin.astype('category')
train_data.Embarked  = train_data.Embarked.astype('category')
train_data.describe(include = 'all')
## Similarly we need to convert attr for test_data
test_data.Sex       = le.fit_transform(test_data.Sex)
test_data.Cabin     = le.fit_transform(test_data.Cabin)
test_data.Embarked  = le.fit_transform(test_data.Embarked)

test_data.Sex       = test_data.Sex.astype('category')
test_data.Cabin     = test_data.Cabin.astype('category')
test_data.Embarked  = test_data.Embarked.astype('category')
train_data.Pclass.value_counts()
train_data.Pclass.describe()
## Pclass needs to be categorized
train_data.Pclass = train_data.Pclass.astype('category')
test_data.Pclass  = test_data.Pclass.astype('category')
test_data.Pclass.describe()
##SibSp and Parch would make more sense as a combined feature, that shows the total number of relatives, a person has on the Titanic.
## I will create it below and also a feature that shows if someone is not alone.
data = [train_data, test_data]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 1
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 0
train_data.columns
train_data.relatives.head()
train_data.Ticket.describe()
##Since the Ticket attribute has 681 unique tickets, it will be a bit tricky to convert them into useful categories. So we will drop it from the dataset.
train_data.drop('Ticket', axis = 1, inplace = True)
test_data.drop('Ticket', axis = 1, inplace = True)
train_data.columns
## Also PassengerID is not significant we can drop it
train_data.drop('PassengerId', axis = 1, inplace = True)
test_data.drop('PassengerId', axis = 1, inplace = True)
train_data.columns
train_data.Name.head()
## We will use the Name feature to extract the Titles from the Name..That might be helpful in predicting Survived
## We will extract it using Series.str.extract 
train_data['Name'] =  train_data['Name'].str.extract(pat = "(Mr|Master|Mrs|Miss|Major|Rev|Lady|Dr|Mme|Mlle|Col|Capt)\\.")
test_data['Name'] =  test_data['Name'].str.extract(pat = "(Mr|Master|Mrs|Miss|Major|Rev|Lady|Dr|Mme|Mlle|Col|Capt)\\.")
train_data.Name.describe()
test_data.Name.describe()
train_data.Name.value_counts()
test_data.Name.value_counts()
train_data.Name.isnull().sum()
test_data.Name.isnull().sum()
## As we can see that most of the Name accounts for Miss,Mr,Mrs,Master we can group the others to Rare
train_data['Name'] = train_data['Name'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Mme','Mlle',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data['Name'].replace(['Mr','Mrs','Miss','Master','Rare'],[1,2,3,4,5],inplace=True)
train_data.Name = train_data.Name.fillna(0)
train_data.Name.describe()
train_data.Name = train_data.Name.astype('category')
train_data.Name.describe()
train_data.Name.value_counts()
test_data['Name'] = test_data['Name'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Mme','Mlle',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_data['Name'].replace(['Mr','Mrs','Miss','Master','Rare'],[1,2,3,4,5],inplace=True)
test_data.Name = test_data.Name.fillna(0)
test_data.Name.describe()
test_data.Name = test_data.Name.astype('category')
test_data.Name.describe()
test_data.Name.value_counts() 
## We need to convert the ‘age’ feature. First we will convert it from float into integer. 
## Then we will create the new ‘AgeGroup” variable, by categorizing every age into a group using the pandas qcut function for data binning
## The pandas documentation describes qcut as a “Quantile-based discretization function.” This basically means that qcut tries to divide up the underlying data into equal sized bins. 
## The function defines the bins using percentiles based on the distribution of the data, not the actual numeric edges of the bins.
train_data.Age.describe()
train_data.Age = train_data.Age.astype(int) 
test_data.Age = test_data.Age.astype(int) 
train_data.Age.describe()
train_data['Age_new'] = pd.qcut(train_data['Age'], q = 6,duplicates = 'drop')
train_data.Age_new.value_counts()
train_data.head()
## Now we need to convert the Age to category
le1 = LabelEncoder()
train_data.Age_new = le1.fit_transform(train_data.Age_new)
train_data.Age_new = train_data.Age_new.astype('category')
train_data.Age_new.value_counts()
test_data['Age_new'] = pd.qcut(test_data['Age'], q = 6, duplicates = 'drop')
test_data.Age_new.value_counts()
test_data.Age_new = le1.fit_transform(test_data.Age_new)
test_data.Age_new = test_data.Age_new.astype('category')
test_data.Age_new.value_counts()
## For the ‘Fare’ feature, we need to do the same as with the ‘Age’ feature
train_data.Fare.describe()
## Min value is 0 Max is 512 
## We need to categorize Fare value with cut
train_data['New_Fare'] = pd.qcut(train_data['Fare'], q = 6)
train_data.New_Fare.value_counts()
train_data.New_Fare = le1.fit_transform(train_data.New_Fare)
train_data.New_Fare = train_data.New_Fare.astype('category')
train_data.New_Fare.value_counts()
test_data['New_Fare'] = pd.qcut(test_data['Fare'], q = 6)
test_data.New_Fare.value_counts()
test_data.New_Fare = le1.fit_transform(test_data.New_Fare)
test_data.New_Fare = test_data.New_Fare.astype('category')
test_data.New_Fare.value_counts()
train_data.head()
## Drop the columns which are not significant
train_data.drop('Age', axis = 1, inplace = True)
train_data.drop('Fare', axis = 1, inplace = True)
test_data.drop('Age', axis = 1, inplace = True)
test_data.drop('Fare', axis = 1, inplace = True)
train_data.drop('not_alone', axis = 1, inplace = True)
test_data.drop('not_alone', axis = 1, inplace = True)
train_data.columns
test_data.columns
