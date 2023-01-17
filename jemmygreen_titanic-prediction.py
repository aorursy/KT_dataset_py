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
# load the dataset

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

submission_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
#Observe the head of the training dataset

train_data.head()
test_data.head()
# Checking the size of the dataset

train_data.shape
test_data.shape
submission_data.head()
# Checking the info of the training dataset

train_data.info()

#Another way of checking null values

missing_values = train_data.isnull().sum().sort_values(ascending=False)

percentage = train_data.isnull().sum()/train_data.isnull().count().sort_values(ascending=False)*100

missing_data = pd.concat([missing_values, percentage], axis=1, keys=['missing_values', 'percentage'])

missing_data.head(5)
#Fill the Nulls in the Age column

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

#Fill the Nulls in the cabin column

train_data['Cabin'] = train_data['Cabin'].fillna(train_data['Cabin'].mode()[0])

#Fill the Nulls in the Embarked column

train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
#Checking for any Null values

train_data.isnull().sum().max()
train_data.head()
train_data.drop(['Name'], axis=1, inplace=True)
train_data.columns
#Separate the Categorical columns from the Numerical columns

train_categorical_cols = train_data.select_dtypes('object')

train_categorical_cols

# Unique values in the categorical columns

train_data['Ticket'].unique()
train_data['Cabin'].unique()

train_data['Embarked'].value_counts()
train_data['Cabin'].value_counts()
train_categorical_cols.head()
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

Features = train_categorical_cols['Sex']

enc.fit(Features)

Features = enc.transform(Features)

print(Features)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

ohe.fit(Features.reshape(-1,1))

Features = ohe.transform(Features.reshape(-1,1)).toarray()

Features[:10,:]

def encode_string(cat_feature):

    

    ## First encode the strings to numeric categories

    enc = LabelEncoder()

    enc.fit(cat_feature)

    enc_cat_feature = enc.transform(cat_feature)

    

    ## Now, apply one hot encoding

    ohe = OneHotEncoder()

    encoded = ohe.fit(enc_cat_feature.reshape(-1,1))

    return encoded.transform(enc_cat_feature.reshape(-1,1)).toarray()

    



cat_columns = ['Embarked']

for col in cat_columns:

    temp = encode_string(train_data[col])

    Features = np.concatenate([Features, temp], axis = 1)

print(Features.shape)

Features[:2,:]
import seaborn as sns

import matplotlib.pyplot as plt
#correlation matrix

corrmat = train_data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 6 #number of variables for heatmap

cols = corrmat.nlargest(k, 'Survived')['Survived'].index

cm = np.corrcoef(train_data[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train_data.head()

train_numerical_cols = train_data.select_dtypes(exclude='object')

train_numerical_cols
Features = np.concatenate([Features, np.array(train_data[[ 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])], axis = 1)

print(Features[:2,:])

print(Features.shape)
train_y = np.array(train_data[['Survived']])

train_y
test_data.info()
test_data.shape
test_data.size
# Fill the Nulls with 

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

test_data['Cabin'] = test_data['Cabin'].fillna(test_data['Cabin'].mode()[0])
test_data.isnull().sum().max()
test_data.drop(['Name',], axis=1, inplace=True)
test_categorical_values = test_data.select_dtypes('object')

test_categorical_values
test_numerical_data = test_data.select_dtypes(exclude="object")

test_numerical_data
enc = LabelEncoder()

Testing = test_categorical_values['Sex']

enc.fit(Testing)

Testing = enc.transform(Testing)

print(Testing)
ohe = OneHotEncoder()

ohe.fit(Testing.reshape(-1,1))

Testing = ohe.transform(Testing.reshape(-1,1)).toarray()

Testing[:10,:]
def encode_string(cat_feature):

    

    ## First encode the strings to numeric categories

    enc = LabelEncoder()

    enc.fit(cat_feature)

    enc_cat_feature = enc.transform(cat_feature)

    

    ## Now, apply one hot encoding

    ohe = OneHotEncoder()

    encoded = ohe.fit(enc_cat_feature.reshape(-1,1))

    return encoded.transform(enc_cat_feature.reshape(-1,1)).toarray()

    



cat_columns = ['Embarked']

for col in cat_columns:

    temp = encode_string(test_data[col])

    Testing = np.concatenate([Testing, temp], axis = 1)

print(Testing.shape)

Testing[:2,:]
Testing = np.concatenate([Testing, np.array(test_data[[ 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])], axis = 1)

print(Testing[:2,:])

print(Testing.shape)
from sklearn.linear_model import LogisticRegression

#split the data

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state = 42)
log_model = LogisticRegression()

log_model.fit(Features, train_y)
import numpy as np

prediction = log_model.predict(Testing)
probabilities = log_model.predict_proba(Testing)

print(probabilities[:15,:])
prediction
prediction.size
prediction.dtype
pred=pd.DataFrame(prediction)

subdf = submission_data

datasets=pd.concat([subdf['PassengerId'],pred],axis=1)

datasets.columns=['PassengerId','Survived']

datasets.to_csv('sample_submission.csv',index=False)
foo = pd.read_csv('sample_submission.csv', index_col=0)

foo
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state=42)

classifier.fit(Features, train_y)

prediction2 = classifier.predict(Testing)
prediction2
pred2=pd.DataFrame(prediction2)

subdf2 = submission_data

datasets=pd.concat([subdf['PassengerId'],pred2],axis=1)

datasets.columns=['PassengerId','Survived']

datasets.to_csv('sample_submission2.csv',index=False)
foo2 = pd.read_csv('sample_submission2.csv', index_col=0)

foo2
from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(classifier.get_params())
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in the randomforest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]



# Number of features to consider at every split

max_features = ['auto', 'sqrt']



# Maximum depth of a tree

max_depth = [int(x) for x in np.linspace (start = 10, stop=150, num = 15)]

max_depth.append(None)



# Minimum number of samples to split a node

min_samples_split = [2, 4, 6, 8, 10]



# Minimum number of samples at each leaf node

min_samples_leaf = [2, 4, 6, 8, 10]



# Method of selecting samples for training each tree

bootstrap = [True, False]





# Create a random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



pprint(random_grid)
classifier_random = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, 

                                      n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)



# Fit the training datasets to the randomsearch model

classifier_random.fit(Features, train_y)
classifier_random.best_params_
cf_best = RandomForestClassifier(max_depth = 70, n_estimators = 1400, random_state = 42, bootstrap=False,

                                min_samples_split = 10, min_samples_leaf=4, max_features = 'auto')
cf_best.fit(Features, train_y)
predictions3 = cf_best.predict(Testing)
predictions3
pred3=pd.DataFrame(predictions3)

subdf3 = submission_data

datasets=pd.concat([subdf3['PassengerId'],pred3],axis=1)

datasets.columns=['PassengerId','Survived']

datasets.to_csv('sample_submission3.csv',index=False)
foo3 = pd.read_csv('sample_submission3.csv', index_col=0)

foo3
import xgboost as xgb

xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(Features, train_y)
predictions4 = xgb_classifier.predict(Testing)

predictions4
pred4=pd.DataFrame(predictions4)

subdf4 = submission_data

datasets=pd.concat([subdf4['PassengerId'],pred4],axis=1)

datasets.columns=['PassengerId','Survived']

datasets.to_csv('sample_submission4.csv',index=False)
foo4 = pd.read_csv('sample_submission4.csv', index_col=0)

foo4
xgb_classifier
estimator = xgb.XGBClassifier(

    objective= 'binary:logistic',

    nthread=4,

    seed=42

)
parameters = {

    'max_depth': range (2, 10, 1),

    'n_estimators': range(60, 220, 40),

    'learning_rate': [0.1, 0.01, 0.05]

}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(

    estimator=estimator,

    param_grid=parameters,

    scoring = 'roc_auc',

    n_jobs = 10,

    cv = 10,

    verbose=True

)
grid_search.fit(Features, train_y)
grid_search.best_estimator_
cf_best2 = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.01, max_delta_step=0, max_depth=3,

              min_child_weight=1, monotone_constraints='()',

              n_estimators=180, n_jobs=4, nthread=4, num_parallel_tree=1,

              random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

              seed=42, subsample=1, tree_method='exact', validate_parameters=1,

              verbosity=None)
cf_best2.fit(Features, train_y)
predictions5 = cf_best2.predict(Testing)
predictions5
pred5=pd.DataFrame(predictions5)

subdf5 = submission_data

datasets=pd.concat([subdf5['PassengerId'],pred5],axis=1)

datasets.columns=['PassengerId','Survived']

datasets.to_csv('sample_submission5.csv',index=False)
foo5 = pd.read_csv('sample_submission5.csv', index_col=0)

foo5