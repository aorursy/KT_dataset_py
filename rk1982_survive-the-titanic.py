import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import warnings



warnings.filterwarnings('ignore')
# Import train and test dataset



df_train = pd.read_csv('../input/titanic/train.csv')





df_test = pd.read_csv('../input/titanic/test.csv')

df_train.shape , df_test.shape
df_train.head()
df_test.head()
df_train.info()
df_test.info()
df_train.describe()
df_test.describe()
# check the type of variables in dataset



df_train.dtypes
# find categorical variables





categorical = [var for var in df_train.columns if df_train[var].dtype=='O']





print('There are {} categorical variables\n'.format(len(categorical)))





print('The categorical variables are :', categorical)
# view the categorical variables



df_train[categorical].head()
# find numerical variables



numerical = [var for var in df_train.columns if df_train[var].dtype!='O']



print('There are {} numerical variables\n'.format(len(numerical)))



print('The numerical variables are :', numerical)
# preview the numerical variables



df_train[numerical].head()
# check missing values in variables in training data



df_train.isnull().sum()
# check missing values in variables in test data



df_test.isnull().sum()
df_train['Sex'].value_counts()
df_test['Sex'].value_counts()
# label minors as child, and remaining people as female or male



def label_child(passenger):

    

    # take the age and sex

    age, sex = passenger

    

    # compare age, return child if under 16, otherwise leave sex

    if age < 16:

        return 'child'

    else:

        return sex
# create a new column `person` which specify the person as male, female or child



df_train['Person'] = df_train[['Age', 'Sex']].apply(label_child, axis = 1)



df_test['Person'] = df_test[['Age', 'Sex']].apply(label_child, axis = 1)
# check the distribution in `Person` variable in training data



df_train['Person'].value_counts()
# check the distribution in `Person` variable in test data



df_test['Person'].value_counts()
# print number of labels in Pclass variable



print('Pclass contains', len(df_train['Pclass'].unique()), 'labels')
# view labels in Pclass variable



df_train['Pclass'].unique()
# check frequency distribution of values in Pclass variable



df_train['Pclass'].value_counts()
# Person segregated by class in training set



sns.factorplot('Pclass', data = df_train, hue = 'Person', kind = 'count')
# distribution of age in training dataset



df_train['Age'].hist(bins=25, grid=False)

# distribution of age in test dataset





df_test['Age'].hist(bins=25, grid=False)
# age segregated by Person



fig = sns.FacetGrid(df_train, hue = 'Person', aspect = 4)

fig.map(sns.kdeplot, 'Age', shade = True)

fig.add_legend()
# view the median age of people in training and test set



for df1 in [df_train, df_test]:

    print(df1.groupby('Person')['Age'].median())
# impute missing values with respective median values



for df1 in [df_train, df_test]:

    df1['Age'] = df1['Age'].fillna(df1.groupby('Person')['Age'].transform('median'))
df_train['Age'].isnull().sum()        
df_test['Age'].isnull().sum()
# print number of labels in Cabin variable



print('Cabin contains', len(df_train['Cabin'].unique()), 'labels in training set')



print('\nCabin contains', len(df_test['Cabin'].unique()), 'labels in test set')
df_train['CabinLetter'] = df_train['Cabin'].str.get(0)



df_test['CabinLetter'] = df_test['Cabin'].str.get(0)
# print number of labels in CabinLetter variable



print('CabinLetter contains', len(df_train['CabinLetter'].unique()), 'labels in training set\n')



print('CabinLetter contains', len(df_test['CabinLetter'].unique()), 'labels in test set')

# view labels in CabinLetter variable in training set



df_train['CabinLetter'].unique()
# view labels in CabinLetter variable in test set



df_test['CabinLetter'].unique()
# view labels in Cabin variable in training set



df_train['Cabin'].unique()
df_train['CabinLetter'].isnull().sum()
df_test['CabinLetter'].isnull().sum()
sns.factorplot('CabinLetter', data = df_train, hue = 'Person', kind = 'count')
sns.factorplot('CabinLetter', data = df_test, hue = 'Person', kind = 'count')
# impute missing values in CabinLetter with respective mode values



for df1 in [df_train, df_test]:

    df1['CabinLetter'] = df1['CabinLetter'].fillna(df1['CabinLetter'].mode().iloc[0])
for df1 in [df_train, df_test]:

    print(df1['CabinLetter'].isnull().sum())
df_train.drop('Cabin', axis = 1, inplace = True)



df_test.drop('Cabin', axis = 1, inplace = True)
# check distribution of `Embarked` variable in training set



df_train['Embarked'].value_counts()
# check distribution of `Embarked` variable in test set



df_test['Embarked'].value_counts()
# where did people from different classes get on board



sns.factorplot('Embarked', data = df_train, hue= 'Pclass', kind = 'count')
for df1 in [df_train, df_test]:

    print(df1['Embarked'].isnull().sum())
df_train['Embarked'].fillna('S', inplace = True)
for df1 in [df_train, df_test]:

    print(df1['Fare'].isnull().sum())
df_test['Fare'].fillna(df_test.Fare.mean(), inplace =True)
df_train.isnull().sum()
df_test.isnull().sum()
for df1 in [df_train, df_test]:

    df1['Familyman'] = df1.Parch + df1.SibSp

    df1['Familyman'].loc[df1['Familyman'] > 0] = 'Yes'

    df1['Familyman'].loc[df1['Familyman'] == 0] = 'No'
# check the frequency distribution of `Familyman` variable



for df1 in [df_train, df_test]:

    print(df1['Familyman'].value_counts())
def man_with_spouse(passenger):

    sex, sibsp = passenger

    if sex == 'male' and sibsp > 0:

        return 1

    else:

        return 0
# create a new variable `man_and_spouse` to check whether a man is travelling with spouse



for df1 in [df_train, df_test]:

    df1['man_and_spouse'] = df1[['Sex', 'SibSp']].apply(man_with_spouse, axis = 1)
def woman_with_child(passenger):

    age, sex, parch = passenger

    if age > 20 and sex == 'female' and parch > 0:

        return 1

    else:

        return 0
# create a new variable `is_mother` to check whether a woman is travelling with child



for df1 in [df_train, df_test]:

    df1['is_mother'] = df1[['Age', 'Sex', 'Parch']].apply(woman_with_child, axis = 1)
# Preview the train dataset again



df_train.head()
# preview the test dataset again



df_test.head()
# drop PassengerId variable 



for df1 in [df_train, df_test]:

    df1.drop('PassengerId', axis=1, inplace=True)

# drop Ticket variable 



for df1 in [df_train, df_test]:

    df1.drop('Ticket', axis=1, inplace=True)
# drop Name variable 



for df1 in [df_train, df_test]:

    df1.drop('Name', axis=1, inplace=True)
X_train = df_train.drop(['Survived'], axis=1)



y_train = df_train.Survived



X_test = df_test
X_train.head()
X_test.head()
# encode sex variable



for df1 in [X_train, X_test]:

    df1['Sex']  = pd.get_dummies(df1.Sex, drop_first=True)
X_train.Sex.unique()
X_test.Sex.unique()
# import category encoders



import category_encoders as ce
# encode categorical variables with ordinal encoding



encoder = ce.OneHotEncoder(cols=['Embarked', 'Person', 'CabinLetter', 'Familyman'])



X_train = encoder.fit_transform(X_train)



X_test = encoder.transform(X_test)
X_train.head()
cols = X_train.columns
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



X_train = scaler.fit_transform(X_train)



X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
from xgboost import XGBClassifier
xgb_final = XGBClassifier()
xgb_final.fit(X_train, y_train)
y_pred = xgb_final.predict(X_test)
test_df = pd.read_csv('../input/titanic/test.csv')

test_df.head()
submission = pd.DataFrame({

                        "PassengerId": test_df['PassengerId'],

                        "Survived": y_pred

                          })
submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook



filename = 'Titanic Predictions.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)