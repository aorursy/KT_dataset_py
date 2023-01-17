import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import StackingCVClassifier

from sklearn.pipeline import Pipeline



import warnings

warnings.filterwarnings('ignore')
X_train = pd.read_csv('../input/titanic/train.csv')

X_test = pd.read_csv('../input/titanic/test.csv')

y_train = X_train['Survived']

df = pd.concat((X_train, X_test)).reset_index(drop=True)

# Store our passenger ID for easy access

PassengerId = X_test['PassengerId']
# Creat new feature based on passengers's titles

df['Title'] = df['Name'].str.split(', ', expand = True)[1].str.split('.',expand=True)[0]

title_changes = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'the Countess':'Mrs','Ms': 'Miss', 'Dona': 'Mrs'}

df.replace({'Title': title_changes}, inplace=True)



# Fill up missing values of 'Age', group by 'Title' first

df['Age'] = df.groupby('Title')['Age'].apply(lambda x: x.fillna(x.median()))



# Create new feature - Family Size

df['Family_Size'] = df['Parch'] + df['SibSp']



# Create new feature - Last name

df['Last_Name'] = df['Name'].apply(lambda x: str.split(x, ",")[0])



# Fill up missing values of Fare

df['Fare'] = df.groupby(['Pclass'])['Fare'].apply(lambda x: x.fillna(x.median()))



# Bin fare and age values

df['FareBin'] = pd.qcut(df['Fare'], 5)

df['AgeBin'] = pd.qcut(df['Age'], 4)
DEFAULT_SURVIVAL_VALUE = 0.5

df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE



for grp, grp_df in df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):   

    if (len(grp_df) != 1):

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
df.drop(['Survived', 'Age', 'Fare', 'Title', 'Last_Name', 'Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
# Encode categorical features

non_numeric_features = ['Sex', 'FareBin', 'AgeBin']

for feature in non_numeric_features:        

    df[feature] = LabelEncoder().fit_transform(df[feature])
# Separate train and test sets

X_train = df.iloc[:len(y_train), :]

X_test = df.iloc[len(y_train):, :]

X_train.shape, X_test.shape
# Scale train and test data

std_scaler = StandardScaler()

X_train = std_scaler.fit_transform(X_train)

X_test = std_scaler.transform(X_test)
# Just initialize the pipeline with any estimator you like    

pipe = Pipeline(steps=[('classifier', SVC())])



# Add a dict of classifier and related parameters in list

params_grid = [{

                'classifier':[SVC()],

                'classifier__C': [0.1, 1, 10, 100],

                'classifier__gamma': [0.001, 0.0001]

                },

              {

                'classifier': [DecisionTreeClassifier()],

                'classifier__max_depth': [2,3,4],

                'classifier__max_features': [None, "auto", "sqrt", "log2"]

              },

              {

                'classifier': [KNeighborsClassifier()],

                'classifier__n_neighbors': [6,7,8,9,10,11,12,14,16,18,20],

                'classifier__algorithm': ['auto'],

                'classifier__weights': ['uniform', 'distance'],

                'classifier__leaf_size': list(range(1,50,5))

              }, 

              {

                'classifier': [LogisticRegression()],

                'classifier__C': [0.1, 1, 10, 100],

              },

              {

                'classifier': [RandomForestClassifier()],

                'classifier__n_estimators': [50, 100, 150, 500],

                'classifier__max_depth': [2, 3, 4, 5],

              },

              {

                'classifier': [GradientBoostingClassifier()],

                'classifier__learning_rate': [0.1, 0.2, 0.5],

                'classifier__n_estimators': [50, 100, 150],

                'classifier__max_depth': [2,3]

              },

              {

                'classifier': [XGBClassifier()],

                'classifier__learning_rate': [0.1, 0.2, 0.3],

                'classifier__n_estimators': [100, 150, 500, 1000],

                'classifier__max_depth': [3,4,5],

                'classifier__gamma': [0, 0.2, 0.5]  

              },

              {

                'classifier': [GaussianNB()]

              }]
gd = GridSearchCV(pipe, params_grid, cv=10, scoring='roc_auc')

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
gd.best_estimator_.fit(X_train, y_train)

y_pred_1 = gd.best_estimator_.predict(X_test)
best_model = KNeighborsClassifier(algorithm='auto', 

                                  leaf_size=26,

                                  metric='minkowski', 

                                  metric_params=None,

                                  n_neighbors=18, 

                                  p=2,

                                  weights='uniform')
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df['PassengerId'] = PassengerId

submission_df['Survived'] = y_pred

submission_df.to_csv('submissions.csv', header=True, index=False)