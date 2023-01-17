import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)



import warnings

warnings.filterwarnings('ignore')



import seaborn as sns



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score,GridSearchCV,RepeatedStratifiedKFold

from sklearn.feature_selection import SelectKBest, f_classif
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

df = train.append(test)
df.head()
print(train.shape)

print(test.shape)
print(train.isnull().sum())

print()

print(test.isnull().sum())
#0=Male; 1=Female

df['Sex'] = pd.get_dummies(df['Sex'])
df['Family_Size'] = df['Parch'] + df['SibSp']
#Finding Passenger's Title

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')

df['Title'] = df['Title'].replace(['Mme',"Countess","Lady","Dona"], 'Mrs')    

df['Title'] = df['Title'].replace(['Capt',"Col","Don","Jonkheer","Major", "Rev","Sir", "Master"],"Other")



df.loc[((df.Title == "Dr") & (df.Sex==1)), 'Title'] = "Mrs"

df.loc[((df.Title == "Dr") & (df.Sex==0)), 'Title'] = "Mr"



titles = list(df.Title.unique())

for title in titles:

    age = df.groupby('Title')['Age'].median().loc[title]

    df.loc[(df.Age.isnull()) & (df.Title == title),'Age'] = age
title_mapping = {"Mr": 0, "Miss": 1, 

                 "Mrs": 2, "Other":3}



df.replace({'Title': title_mapping}, inplace=True)
df["Age"].fillna(df.Age.median(), inplace=True)

df['Fare'].fillna(value = df[df.Pclass==3]['Fare'].median(), inplace = True)
# Quantile-based discretization function

df['Fare_Bin'] = pd.qcut(df['Fare'], 5, labels=False)

df['Age_Bin'] = pd.qcut(df['Age'], 4, labels=False)
# This feature is inspired by https://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever

df['Last_Name'] = df['Name'].apply(lambda x: str.split(x, ",")[0])

df['Fare'].fillna(df['Fare'].mean(), inplace=True)



DEFAULT_SURVIVAL_VALUE = 0.5

df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE



for grp, grp_df in df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    

    if (len(grp_df) != 1):

        # A Family group is found.

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin==0.0):

                df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0



print("Number of passengers with family survival information:", 

      df.loc[df['Family_Survival']!=0.5].shape[0])
for _, grp_df in df.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0

                        

print("Number of passenger with family/group survival information: " 

      +str(df[df['Family_Survival']!=0.5].shape[0]))



# # Family_Survival in TRAIN_DF and TEST_DF:

train['Family_Survival'] = df['Family_Survival'][:891]

test['Family_Survival'] = df['Family_Survival'][891:]
features = ['Survived','Title', 'Pclass','Sex','Family_Size','Family_Survival','Fare_Bin','Age_Bin']

df = df[features]
train = df[:len(train)]



x_train = train.drop('Survived', axis=1)

y_train = train['Survived'].astype(int)



x_test = df[len(train):].drop('Survived', axis=1)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

x_train = pd.DataFrame(scaler.fit_transform(x_train))

x_test = pd.DataFrame(scaler.fit_transform(x_test))
# Model

model = KNeighborsClassifier()



# KFold

kfold = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=420)
# GSCV

from sklearn.model_selection import GridSearchCV



param_grid = {

    "n_neighbors":np.arange(2,20,2),

    "weights":["uniform","distance"],

    "leaf_size":np.arange(1,50,5)

}



gscv = GridSearchCV(model, cv=kfold, param_grid=param_grid, scoring='accuracy')

gscv_result = gscv.fit(x_train,y_train)
# GSCV Result

best_acc_score=gscv_result.best_score_

best_params=gscv_result.best_params_



print("Accuracy: {:6f}".format(best_acc_score))

print("Params: {}".format(best_params))
# Model training and prediction

result = gscv.predict(x_test)
# Exporting result

test_new = pd.read_csv('../input/titanic/test.csv')



submission = pd.DataFrame({'PassengerId': test_new['PassengerId'], 'Survived': result})

submission.to_csv('submission.csv', index = False)

submission.head()