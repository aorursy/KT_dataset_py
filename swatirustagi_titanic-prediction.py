#importing libraries

import pandas as pd

import numpy as np



#Visualisation Libraries

import matplotlib.pyplot as plt

import seaborn as sns



#modelling libraries

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



#evaluation metrics

from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score



import warnings

warnings.simplefilter("ignore")

%matplotlib inline
#reading data

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

train.head(2)
test.head(2)
#dropping the PId from train and test dataset

train.drop("PassengerId", axis = 1, inplace = True)

test.drop("PassengerId", axis = 1, inplace = True)
print("Row and Columns in Train Data are", train.shape[0], "&", train.shape[1], "respectively.")

print("Row and Columns in Test Data are", test.shape[0], "&", test.shape[1], "respectively.")
train.describe()
print("Data type in Train Dataset.\n\n", train.dtypes)

print("*"*50)

print("Data type in Test Dataset.\n\n", test.dtypes)
train.corr()
test.corr()
# Missing values

def missing_values_table(df):

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

        print ("Your selected dataframe has " + str(df.shape[1]))   

        print("There are " + str(mis_val_table_ren_columns.shape[0])+" columns that have missing values.")

        return mis_val_table_ren_columns
#checking on the missing values in train dataset

missing_values_train= missing_values_table(train)

missing_values_train.style.background_gradient(cmap='Paired_r')
#checking on the missing values in train dataset

missing_values_test= missing_values_table(test)

missing_values_test.style.background_gradient(cmap='Paired_r')
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

sns.countplot(x = "Pclass", hue= "Survived", data = train, dodge = True)
print(train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean())

sns.countplot(x = "Survived", hue= "Sex", data = train, dodge = True)
print(train[['SibSp','Survived']].groupby(['SibSp'], as_index=False).count())

sns.countplot(x = "SibSp", hue = "Survived", data = train, dodge = True)
bins = np.arange(0, 85, 5)

g = sns.FacetGrid(train, col='Survived', row = 'Sex')

g.map(plt.hist, 'Age', bins=bins)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
sns.boxplot(x = 'Pclass', y = 'Age', data = train)
sns.boxplot(x = 'Pclass', y = 'Age', data = test)
sns.heatmap(train.isnull(), yticklabels= False)
sns.heatmap(test.isnull(), yticklabels= False)
train.groupby(['Survived','Sex','Pclass'])['Age'].mean()
age_groupby_train = train.groupby(['Survived','Sex','Embarked','Pclass'])['Age'].mean()

age_groupby_train
age_groupby_test = test.groupby(['Sex','Embarked','Pclass'])['Age'].mean()

age_groupby_test
train['Age'].fillna(value = -1,inplace =True)

test['Age'].fillna(value = -1,inplace =True)
def fill_value_train(df):

    for row in range(len(age_groupby_train.index)):

        df.loc[(df['Survived'] == age_groupby_train.index[row][0]) &

                 (df['Sex']== age_groupby_train.index[row][1]) &

                 (df['Embarked']== age_groupby_train.index[row][2])&

                 (df['Pclass']== age_groupby_train.index[row][3])&

                 (df['Age']==-1),'Age']=age_groupby_train.values[row]
def fill_value_test(df):

    for row in range(len(age_groupby_test.index)):

        df.loc[(df['Sex']== age_groupby_train.index[row][1]) &

                 (df['Embarked']== age_groupby_train.index[row][2])&

                 (df['Pclass']== age_groupby_train.index[row][3])&

                 (df['Age']==-1),'Age']=age_groupby_train.values[row]
fill_value_train(train)
fill_value_test(test)
sns.heatmap(train.isnull(), yticklabels= False)
sns.heatmap(test.isnull(), yticklabels= False)
Sex_train = pd.get_dummies(train['Sex'], drop_first=True)

Embark_train = pd.get_dummies(train['Embarked'], drop_first= True)

Pcls_train = pd.get_dummies(train['Pclass'], drop_first= True)



Sex_test = pd.get_dummies(test['Sex'], drop_first=True)

Embark_test = pd.get_dummies(test['Embarked'], drop_first= True)

Pcls_test = pd.get_dummies(test['Pclass'], drop_first= True)
train = pd.concat([train, Sex_train, Embark_train, Pcls_train], axis = 1)

train.head(2)
test = pd.concat([test, Sex_test, Embark_test, Pcls_test], axis = 1)

test.head(2)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'Cabin', 'Fare'], axis = 1, inplace = True)

train.head(2)
test.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'Cabin', 'Fare'], axis = 1, inplace = True)

test.head(2)
print("Missing Values of Age in Train Dataset :", train.isnull().any())

print("Missing Values of Age in Test Dataset :", test.isnull().any())
X = train[['Pclass','Age','SibSp', 'Parch', 'male', 'Q','S']]

y = train['Survived']



X_test =test[['Pclass','Age','SibSp', 'Parch', 'male', 'Q','S']]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)

lr = LogisticRegression(random_state= 42, fit_intercept=True,intercept_scaling=1)

lr.fit(X_train,y_train)

y_pred = lr.predict(X_val)
print("Train Data Accuracy {0} ".format(lr.score(X_train, y_train)))

print("Test Data Accuracy {0} ".format(lr.score(X_val, y_val)))

print("Confusion Matrix \n",confusion_matrix(y_val,y_pred))

print("Classification Matrix \n", classification_report(y_val,y_pred))
y_test = lr.predict(X_test)

y_test
dummy = pd.Series(y_test)

type(dummy)
frame = { 'y_predicted': dummy}

result = pd.DataFrame(frame) 

result.head()
test.head()
test_result = pd.merge(test, result, left_index=True, right_index=True)
test_result.shape
test_result.head()