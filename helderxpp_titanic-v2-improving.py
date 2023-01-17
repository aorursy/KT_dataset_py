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
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import StratifiedKFold



import string

import warnings

warnings.filterwarnings('ignore')



SEED = 42
def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set on axis 0

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



def divide_df(all_data):

    # Returns divided dfs of training and test set

    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)



df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_all = concat_df(df_train, df_test)



df_train.name = 'Training Set'

df_test.name = 'Test Set'

df_all.name = 'All Set' 



dfs = [df_train, df_test]



print('Number of Training Examples = {}'.format(df_train.shape[0]))

print('Number of Test Examples = {}\n'.format(df_test.shape[0]))

print('Training X Shape = {}'.format(df_train.shape))

print('Training y Shape = {}\n'.format(df_train['Survived'].shape[0]))

print('Test X Shape = {}'.format(df_test.shape))

print('Test y Shape = {}\n'.format(df_test.shape[0]))

print(df_train.columns)

print(df_test.columns)
df_all

def display_missing(df):    

    for col in df.columns.tolist():          

        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))

    print('\n')

    

for df in dfs:

    print('{}'.format(df.name))

    display_missing(df)
df_all_corr = df_all.corr().abs().unstack().sort_values(kind="quicksort", ascending=True).reset_index()

df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_all_corr[df_all_corr['Feature 1'] == 'Age']
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']

age_by_pclass_sex
# Filling the missing values in Age with the medians of Sex and Pclass groups

df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
df_all[df_all['Embarked'].isnull()]

# Filling the missing values in Embarked with S

df_all['Embarked'] = df_all['Embarked'].fillna('S')
df_all[df_all['Fare'].isnull()]
df_all['Fare'] = df_all['Fare'].fillna(df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0])
df_train.groupby(['Parch','Pclass']).Survived.mean()*100
# Creating Deck column from the first letter of the Cabin column (M stands for Missing)

df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df_all=df_all.drop(['Cabin'],axis=1)

df_all
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')

df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')

df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')
df_all
df_train, df_test = divide_df(df_all)
# Seaborn visualization library

import seaborn as sns

# Create the default pairplot

sns.pairplot(df_train,hue='Survived')
plt.figure(figsize=(10,10))

sns.heatmap(df_train.corr(),

            vmin=-1,

            annot=True);



plt.title('Correlation matrix')

plt.show()

#test set

plt.figure(figsize=(10,10))

sns.heatmap(df_test.corr(),

            vmin=-1,

            annot=True);



plt.title('Correlation matrix')

plt.show()
#split fares in quartiles

df_all['Fare'] = pd.qcut(df_all['Fare'], 13)

df_all
fig, axs = plt.subplots(figsize=(25, 10))

sns.countplot(x='Fare', hue='Survived', data=df_all)

plt.legend(['Not Survived', 'Survived'])

plt.show()
df_all['Age'] = pd.qcut(df_all['Age'], 7)
fig, axs = plt.subplots(figsize=(25, 10))

sns.countplot(x='Age', hue='Survived', data=df_all)

plt.legend(['Not Survived', 'Survived'])

plt.show()
df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1

df_all
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)

df_all.drop(['Family_Size'],inplace=True,axis=1)

df_all
fig, axs = plt.subplots(figsize=(15, 7))

sns.barplot(x=df_all['Family_Size_Grouped'].value_counts().index, y=df_all['Family_Size_Grouped'].value_counts().values)

plt.show()
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')

df_all
fig, axs = plt.subplots(figsize=(12, 9))

sns.countplot(x='Ticket_Frequency', hue='Survived', data=df_all)

plt.show()
df_all['Title']=df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df_all.Title.unique()
df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')

df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

df_all.Title.unique()
fig, axs = plt.subplots(figsize=(12, 9))

sns.countplot(x='Title', hue='Survived', data=df_all)

plt.show()
df_all
df_train = df_all.loc[:890]

df_test = df_all.loc[891:]

dfs = [df_train, df_test]
non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']



for df in dfs:

    for feature in non_numeric_features:        

        df[feature] = LabelEncoder().fit_transform(df[feature])

  



        
df_test.drop(['Name','PassengerId','Ticket','Survived'], axis=1,inplace=True)  

df_train.drop(['Name','PassengerId','Ticket'], axis=1,inplace=True)  
df_test
X_train=df_train.drop(['Survived'],axis=1)

y_train=df_train['Survived'].values



X_test=df_test



print('X_train shape: {}'.format(X_train.shape))

print('y_train shape: {}'.format(y_train.shape))

print('X_test shape: {}'.format(X_test.shape))
X_train_scaled = StandardScaler().fit_transform(X_train)

X_test_scaled = StandardScaler().fit_transform(X_test)
single_best_model = RandomForestClassifier(criterion='gini', 

                                           n_estimators=1100,

                                           max_depth=5,

                                           min_samples_split=4,

                                           min_samples_leaf=5,

                                           max_features='auto',

                                           oob_score=True,

                                           random_state=SEED,

                                           n_jobs=-1,

                                           verbose=1)



from sklearn.model_selection import GridSearchCV



param_grid = {

                 'n_estimators': [1200, 1000, 1100,1300,1400,1250,1150],

                 'max_depth': [2,3, 5,6, 7,8, 9]

             }

grid_clf = GridSearchCV(single_best_model, param_grid, cv=10)

grid_clf.fit(X_train, y_train)
grid_clf.best_score_ 
grid_clf.best_params_
rfc_prediction = grid_clf.predict(X_test)

Y_pred=rfc_prediction.astype(int)
Y_pred
submission = pd.DataFrame({

        "PassengerId": df_all.loc[891:]['PassengerId'],

        "Survived": Y_pred

    })

submission.to_csv('submission_gxboost_search_ird_cv5.csv', index=False)