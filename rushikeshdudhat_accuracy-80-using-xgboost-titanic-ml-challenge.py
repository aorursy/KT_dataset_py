# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

from xgboost import XGBClassifier

import xgboost as xgb

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingClassifier
X = pd.read_csv('/kaggle/input/titanic/train.csv')

X_test = pd.read_csv('/kaggle/input/titanic/test.csv')

X.dropna(axis=0,subset=['Survived'],inplace=True)

y = X.Survived
num_col = [col for col in X.columns if X[col].dtypes != 'object']

print(num_col)

num_col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
#plot the countplots

import seaborn as sns

#define the figure and subplotts

fig,axs = plt.subplots(nrows = 3, ncols = 2, figsize=(20,20))

for i, feature in enumerate(num_col,1):

    plt.subplot(3,2,i)

    sns.countplot(x=feature, hue = 'Survived', data=X)

    plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size':20})
total_data = X.append(X_test)

total_data.describe()
missing_cols = [cols for cols in total_data.columns if total_data[cols].isna().any()]

print(missing_cols)
print(total_data['Age'].isna().sum())

print(total_data['Cabin'].isna().sum())

print(total_data['Embarked'].isna().sum())

print(total_data['Fare'].isna().sum())
#Predicting the unknown values of Age

#total_data.groupby('Embarked')['Survived'].count()

total_data.loc[total_data['Embarked'].isnull() == True, 'Embarked'] = 'S'

total_data.loc[total_data['Fare'].isnull()==True, 'Fare'] = 14.5

total_age_data_known = total_data.loc[total_data['Age'].isnull() == False]

total_age_data_unknown = total_data.loc[total_data['Age'].isnull() == True]

total_age_data_unknown.drop('Age',axis=1, inplace=True)


features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']

#LE = ['Sex','Embarked']

#total_age_data_known['family_size'] = total_age_data_known['SibSp']+total_age_data_known['Parch']+1

age_train, age_valid = train_test_split(total_age_data_known, train_size=0.6, random_state=0)

age_train_y = age_train['Age']

age_valid_y = age_valid['Age']

total_age_known_y = total_age_data_known['Age']

"""encoder = LabelEncoder()

for col in LE:

    age_train[col] = encoder.fit_transform(age_train[col])

    age_valid[col] = encoder.transform(age_valid[col])

    total_age_data_known[col] = encoder.transform(total_age_data_known[col])

    total_age_data_unknown[col] = encoder.transform(total_age_data_unknown[col])"""

age_train = pd.get_dummies(age_train[features])

age_valid = pd.get_dummies(age_valid[features])

total_age_known = pd.get_dummies(total_age_data_known[features])

total_age_unknown = pd.get_dummies(total_age_data_unknown[features])

#total_age_unknown.drop('Age', axis=1, inplace = True)



total_age_unknown.columns
xg_model_age = XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=4, subsample=None,

                        colsample_bytree=0.8, gamma=10,reg_alpha = 1)

xg_model_age.fit(age_train, age_train_y, early_stopping_rounds=5, eval_set=[(age_valid,age_valid_y)],

             verbose=False)

predict=np.round(xg_model_age.predict(age_valid))

score = metrics.mean_absolute_error(age_valid_y,predict)

print(score)

scores = -1*cross_val_score(xg_model_age, total_age_known, total_age_known_y, 

                         cv = 5, scoring ='neg_mean_absolute_error')

print(scores)

print(sum(scores)/5)

#plt.figure(figsize=(10,5)) 

#xgb.plot_importance(xg_model, ax=plt.gca())
#Train on whole data and get values for unknown ages

xg_model_age.fit(total_age_known, total_age_known_y, early_stopping_rounds=5, 

             eval_set=[(age_valid,age_valid_y)], verbose=False)

predict=np.round(xg_model_age.predict(total_age_unknown))

#Create a new dataframe with all the features

output = pd.DataFrame({'PassengerId': total_age_data_unknown.PassengerId, 'Survived': total_age_data_unknown.Survived,

                       'Pclass': total_age_data_unknown.Pclass, 'Name': total_age_data_unknown.Name, 

                       'Sex': total_age_data_unknown.Sex, 'Age': predict, 'SibSp': total_age_data_unknown.SibSp,

                       'Parch': total_age_data_unknown.Parch, 'Ticket': total_age_data_unknown.Ticket,

                       'Fare': total_age_data_unknown.Fare.astype(float), 'Cabin': total_age_data_unknown.Cabin,

                       'Embarked': total_age_data_unknown.Embarked})

full_data = total_age_data_known.append(output)
#Feature engineering on full data

#full_data['Fare'] = pd.qcut(full_data['Fare'], 13)

#full_data['Age'] = pd.qcut(full_data['Age'], 10)

full_data['Ticket_Frequency'] = full_data.groupby('Ticket')['Ticket'].transform('count')

full_data['known_cabin'] = 0

full_data.loc[full_data['Cabin'].isnull() == False,'known_cabin']=1

full_data.loc[full_data['Cabin'].isnull() == True,'known_cabin']=0

#full_data.groupby('known_cabin')['Survived'].count()

full_data['Age_cat']=0

full_data.loc[full_data['Age'] < 3, 'Age_cat'] = 0

full_data.loc[full_data['Age'] > 3, 'Age_cat'] = 1

full_data.loc[full_data['Age'] > 14, 'Age_cat'] = 2

full_data.loc[full_data['Age'] > 24, 'Age_cat'] = 3

full_data.loc[full_data['Age'] > 34, 'Age_cat'] = 4

full_data.loc[full_data['Age'] > 44, 'Age_cat'] = 5

full_data.loc[full_data['Age'] > 54, 'Age_cat'] = 6

full_data.loc[full_data['Age'] > 64, 'Age_cat'] = 7

#Create a seperate feature for family size

full_data['family_size']=full_data['SibSp']+full_data['Parch']+1

#family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

#full_data['Family_Size_Grouped'] = full_data['family_size'].map(family_map)

'''LE=['Age','Family_Size_Grouped','Fare']

encoder = LabelEncoder()

for col in LE:

    full_data[col] = encoder.fit_transform(full_data[col])'''
full_data.dtypes
full_data['title'] = full_data['Name'].str.extract('([A-Za-z]+)\.',expand = False)

full_data['title'] = full_data['title'].replace('Mlle','Miss') 

full_data['title'] = full_data['title'].replace('Ms','Mrs') 

full_data['title'] = full_data['title'].replace('Mme','Mrs')

full_data['title'] = full_data['title'].replace(['Mlle', 'Master','Dr','Rev','Col','Major','Dona',

                                                'Capt','Lady','Jonkheer','Countess','Don','Sir'],'Rare') 

x=1

full_data['title_num'] = 0

for unique in full_data['title'].unique().tolist():

    full_data.loc[full_data['title'] == unique,'title_num'] = x

    x+=1

full_data.drop('title',axis=1,inplace=True)

#full_data.describe()
#full_data['Age_Fare'] = full_data.loc[full_data['Age']<6, ['Age_Fare']] = 1

full_data['Age_Fare'] = 1

full_data.loc[full_data['Fare']<60 , ['Age_Fare']] = 0

full_data.loc[full_data['Age']>65, ['Age_Fare']] = 0

full_data.loc[full_data['Age']<6, ['Age_Fare']] = 1

#full_data['Age_Fare'] = full_data.loc[full_data['Age_Fare'].isnull()==True , ['Age_Fare']] = 1

#full_data['Age_Fare'].sum()
#Split full data into train and test sets again

X_new = full_data.loc[full_data['Survived'].isna()==False]

X_test_new = full_data.loc[full_data['Survived'].isna()==True]

print('Length of training data::',len(X_new),'\n' 'Length of test data::',len(X_test_new))

#X_new.groupby('Age_cat')['Survived'].count()

X_new_copy = X_new.copy()

X_test_new_copy=X_test_new.copy()
#Drop Survived 

y_new_copy = X_new_copy.Survived

#X_new_copy.drop(['Survived'], axis =1, inplace=True)

missing_cols = [cols for cols in X_new_copy.columns if X_new_copy[cols].isna().any()]

print(missing_cols) #Check for missing cols
X_new_copy.columns
#num_cols = [col for col in X_new_copy.columns if X_new_copy[col].dtypes in ['int64','float64']]

#num_cols = ['Pclass', 'Fare', 'known_cabin', 'Age_cat','family_size','title_num', 'SibSp', 'Parch',

#            'Ticket_Frequency', 'Age']

num_cols = ['Pclass', 'Fare', 'known_cabin', 'Age_cat','family_size','title_num', 

            'Ticket_Frequency']

print(num_cols)

low_card_cat_columns = [col for col in X_new_copy.columns if (X_new_copy[col].dtypes == 'object' and X_new_copy[col].nunique()<10)]

print(low_card_cat_columns)
#Split data into train-test

X_train,X_valid,y_train,y_valid = train_test_split(X_new_copy,y_new_copy,train_size=0.6, test_size=0.4, random_state=3)



OH_X_train = pd.get_dummies(X_train[low_card_cat_columns])

OH_X_valid = pd.get_dummies(X_valid[low_card_cat_columns])

OH_X_test = pd.get_dummies(X_test_new_copy[low_card_cat_columns])

OH_X_full = pd.get_dummies(X_new_copy[low_card_cat_columns])



X_train_num = X_train[num_cols]

X_test_num = X_test_new_copy[num_cols]

X_valid_num = X_valid[num_cols]

X_full_num = X_new_copy[num_cols]



X_train_num = pd.concat([X_train_num,OH_X_train], axis=1)

X_test_num = pd.concat([X_test_num,OH_X_test], axis=1)

X_valid_num = pd.concat([X_valid_num,OH_X_valid], axis=1)

X_full_num = pd.concat([X_full_num,OH_X_full], axis=1)

X_full_num.describe()
#XGB MODEL

xg_model = XGBRegressor(n_estimators=350, learning_rate=0.1, max_depth=None, subsample=0.7,

                       colsample_bytree=None,gamma=1,reg_alpha = None)

xg_model.fit(X_train_num, y_train, early_stopping_rounds=5, eval_set=[(X_valid_num,y_valid)], verbose=False)

#xg_model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05)

#xg_model.fit(X_train_num, y_train)

predict=np.round(xg_model.predict(X_valid_num))

#predict=xg_model.predict(X_valid_num)

score = metrics.roc_auc_score(y_valid,predict)

print(score)

plt.figure(figsize=(10,5)) 

xgb.plot_importance(xg_model, ax=plt.gca())

#Fine-tune the hyper-parameters using Cross-Validation

scores = cross_val_score(xg_model, X_full_num, y_new_copy, cv = 5, scoring ='roc_auc')

print(scores)

print(sum(scores)/5)
#Fitting the model with full training data

xg_model.fit(X_full_num, y_new_copy, early_stopping_rounds=5, eval_set=[(X_valid_num,y_valid)], verbose=False)

#xg_model.fit(X_full_num, y_new_copy)

test_preds=np.round(xg_model.predict(X_test_num))

test_preds = test_preds.astype(int)
output = pd.DataFrame({'PassengerId': X_test_new_copy.PassengerId,

                       'Survived': test_preds})

output.to_csv('submission.csv', index=False)