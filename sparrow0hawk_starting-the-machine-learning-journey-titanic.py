# data handling libraries
import numpy as np
import pandas as pd

# data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# import dataset for training model
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

# check its right
train_df.head()
print('Non null data info')
train_df.info()
print('-------------')
test_df.info()
# understand distribution of null values
train_df.isnull().sum(), print('------'),test_df.isnull().sum()
print(train_df.describe())
plt.figure(figsize=(6,12))
plt.subplot(5,1,1)
sns.barplot(x='Sex',y='Survived', data=train_df)
# fares versus survived
train_df['Fare'] = train_df['Fare'].fillna(-0.5)
bins = [-1, 0, 20 ,41 , 60, 81, 100, np.inf]
labels = ['Unknown','1-20','21-41','42-60','61-81','82-100','101+']
train_df['FareGroup'] = pd.cut(train_df['Fare'],bins,labels = labels)
plt.subplot(5,1,2)
sns.barplot(x='FareGroup',y='Survived', data=train_df)
plt.subplot(5,1,3)
sns.barplot(x='Embarked',y='Survived', data=train_df)
plt.subplot(5,1,4)
sns.barplot(x='SibSp',y='Survived', data=train_df)



# parents/children aboard
plt.subplot(5,1,5)
sns.barplot(x='Parch',y='Survived', data=train_df)
plt.tight_layout()
train_df['SibSpBool'] = (train_df['SibSp'].apply(lambda x: 1 if x>0 else 0))

train_df['ParchBool'] = (train_df['Parch'].apply(lambda x: 1 if x>0 else 0))

train_df['CabinBool'] = (train_df['Cabin'].notnull().astype('int'))

plt.figure(figsize=(4,6))
plt.subplot(3,1,1)
sns.barplot(x='ParchBool',y='Survived', data=train_df)
plt.subplot(3,1,2)
sns.barplot(x='SibSpBool',y='Survived', data=train_df)
plt.subplot(3,1,3)
sns.barplot(x='CabinBool',y='Survived', data=train_df)
plt.tight_layout()
# dictionary to map values in column to numbers
sex_map = {'male' : 0, 'female' : 1}
# replace using the map dictionary
train_df['Sex'] = train_df['Sex'].replace(sex_map)

# map for fare groups
fare_map = {'Unknown' : 0,'1-20' : 1,'21-41' : 2,'42-60' :3 ,'61-81' : 4,'82-100' : 5,'101+' : 6}
# replace using the map dictionary
train_df['FareGroup'] = train_df['FareGroup'].replace(fare_map)
# age imputer
def age_imputer(dataf_to_impute,dataf_to_ref):
    # create groupby table of titles and mean age
    title_age = dataf_to_ref[['Title','Age']][dataf_to_ref['Age'].notnull()].groupby('Title').mean()

    # for loop to impute mean values from above table
    for Id in dataf_to_impute['PassengerId'][dataf_to_impute['Age'].isnull()]:
        for tle in dataf_to_impute['Title'][dataf_to_impute['PassengerId'] == Id]:
            dataf_to_impute['Age'][dataf_to_impute['PassengerId'] == Id] = title_age.loc[tle].sum()

# function for returning titles
def title_maker(dataframe):
    dataframe['Title'] = (dataframe['Name'][dataframe['Name'].notnull()]).apply(lambda x: list(list(x.split(','))[1].split(' '))[1])
# check their are currently NA values in age
train_df['Age'].isna().sum()
# run the functions and impute the NAs
title_maker(train_df)

age_imputer(train_df, train_df)

# check the NAs are gone
train_df['Age'].isna().sum()
# lets plot it versus survived
bins = [-1,0 ,5 ,12 , 18, 24, 35, 60, np.inf]
labels = ['Unknown','Baby','Child','Student','Teenager','Young Adult','Adult','Senior']
train_df['AgeGroup'] = pd.cut(train_df['Age'],bins,labels = labels)

sns.barplot(x='AgeGroup',y='Survived', data=train_df)
age_map = {'Unknown' : 0,
           'Baby' : 1,
           'Child' : 2,
           'Student' : 3,
           'Teenager' : 4,
           'Young Adult' : 5,
           'Adult' : 6,
           'Senior' : 7}

train_df['AgeGroup'] = train_df['AgeGroup'].replace(age_map)
# using seaborn to see correlations (plots both negative and positive)
corrmap = train_df[['PassengerId','Survived','Pclass','FareGroup','AgeGroup','SibSpBool','ParchBool','Sex','CabinBool']].corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmap, vmax=.8, square=True)
train_df.columns
Y = train_df['Survived'].values.ravel()

# all simplified variables
X_new = train_df[['Pclass','FareGroup','AgeGroup','SibSpBool','ParchBool','Sex','CabinBool']]

# original variables that are numeric
X_orig = train_df[['Pclass','Sex','Age','SibSp','Parch','Fare','CabinBool']]
# using cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

model_RF = RandomForestClassifier(random_state=0)

my_pipeline = make_pipeline(model_RF)

scores1_RF = cross_val_score(my_pipeline,X_orig,Y,scoring = 'accuracy',cv=5)

scores2_RF = cross_val_score(my_pipeline,X_new,Y,scoring = 'accuracy', cv=5)

Y_preds1 = cross_val_predict(my_pipeline,X_orig,Y)

print('Score for model with original features : {}'.format(scores1_RF*100))

print('Score for model with simplified features : {}'.format(scores2_RF*100))
# using cross validation
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

model = XGBClassifier(random_state=0)

my_pipeline = make_pipeline(model)

scores1_XG = cross_val_score(my_pipeline,X_orig,Y,scoring = 'accuracy',cv=5)

scores2_XG = cross_val_score(my_pipeline,X_new,Y,scoring = 'accuracy', cv=5)

Y_preds1 = cross_val_predict(my_pipeline,X_orig,Y)

print('Score for model with original features : {}'.format(scores1_XG*100))

print('Score for model with simplified features : {}'.format(scores2_XG*100))
# using cross validation
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

model = SVC(random_state=0, gamma="auto")

my_pipeline = make_pipeline(model)

scores1_SV = cross_val_score(my_pipeline,X_orig,Y,scoring = 'accuracy',cv=5)

scores2_SV = cross_val_score(my_pipeline,X_new,Y,scoring = 'accuracy', cv=5)

Y_preds1 = cross_val_predict(my_pipeline,X_orig,Y)

print('Score for model with original features : {}'.format(scores1_SV*100))

print('Score for model with simplified features : {}'.format(scores2_SV*100))
# attempt at using ANN to solve this problem
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

my_pipeline = make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(7,7,7),random_state=0,max_iter=1000))

scores1_ANN = cross_val_score(my_pipeline,X_orig,Y,scoring = 'accuracy',cv=5)

scores2_ANN = cross_val_score(my_pipeline,X_new,Y,scoring = 'accuracy', cv=5)

Y_preds1 = cross_val_predict(my_pipeline,X_orig,Y)

print('Score for model with original features : {}'.format(scores1_ANN*100))

print('Score for model with simplified features : {}'.format(scores2_ANN*100))
plt.figure(figsize=(12,6))
plt.boxplot([scores1_RF,scores2_RF,scores1_XG,scores2_XG,scores1_SV,scores2_SV,scores1_ANN,scores2_ANN],
           labels=['Original RF','Simplified RF',
                  'Original XGBoost','Simplified XGBoost',
                  'Original SVC','Simplified SVC',
                  'Original ANN','Simplified ANN'])
# initially check how many NA in Age
# make a copy
test_df1 = test_df
# check NA
test_df1['Age'].isna().sum()
# make title column to help age imputer
title_maker(test_df1)
# impute age with function
age_imputer(test_df1,train_df)
# check NAs are gone!
test_df1['Age'].isna().sum()
# Engineer features as done on test set

# age to AgeGroups
bins = [-1,0 ,5 ,12 , 18, 24, 35, 60, np.inf]

labels = ['Unknown','Baby','Child','Student','Teenager','Young Adult','Adult','Senior']

test_df1['AgeGroup'] = pd.cut(test_df1['Age'],bins,labels = labels)

age_map = {'Unknown' : 0,
           'Baby' : 1,
           'Child' : 2,
           'Student' : 3,
           'Teenager' : 4,
           'Young Adult' : 5,
           'Adult' : 6,
           'Senior' : 7}

test_df1['AgeGroup'] = test_df1['AgeGroup'].replace(age_map)

# Sex to binary
sex_map = {'male' : 0, 'female' : 1}

test_df1['Sex'] = test_df1['Sex'].replace(sex_map)

#Cabin to CabinBool
test_df1['CabinBool'] = (test_df1['Cabin'].notnull().astype('int'))

# fare to FareGroup
test_df1['Fare'] = test_df1['Fare'].fillna(-0.5)
bins = [-1, 0, 20 ,41 , 60, 81, 100, np.inf]
labels = ['Unknown','1-20','21-41','42-60','61-81','82-100','101+']
test_df1['FareGroup'] = pd.cut(test_df1['Fare'],bins,labels = labels)
fare_map = {'Unknown' : 0,'1-20' : 1,'21-41' : 2,'42-60' :3 ,'61-81' : 4,'82-100' : 5,'101+' : 6}

test_df1['FareGroup'] = test_df1['FareGroup'].replace(fare_map)

# ParchBool
test_df1['ParchBool'] = (test_df1['Parch'].apply(lambda x: 1 if x>0 else 0))

#SibSp to SibSpBool
test_df1['SibSpBool'] = (test_df1['SibSp'].apply(lambda x: 1 if x>0 else 0))
test_df1.head()
test_df1_pred = test_df1[X_new.columns]

test_df1_pred.head()
ids = test_df1['PassengerId']

model = SVC(random_state=0, gamma="auto")

model.fit(X_new,Y)

test_df1_pred = test_df1[X_new.columns]

predictions = model.predict(test_df1_pred)

output = pd.DataFrame({'PassengerId' : ids,
                      'Survived' : predictions})

output.to_csv('submission.csv', index=False)