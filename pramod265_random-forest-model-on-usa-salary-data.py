# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_excel(os.path.join(dirname, "train.xlsx"))

test = pd.read_excel(os.path.join(dirname, "test.xlsx"))

#shape of DF

print("Training data ",train.shape)

print("Testing data ",test.shape)



train.head()
nans = train.shape[0] - train.dropna().shape[0]

print ("%d rows have missing values in the train data" %nans)



nand = test.shape[0] - test.dropna().shape[0]

print ("%d rows have missing values in the test data" %nand)
# Check columns with missing data



train.isnull().sum()
# number of unique values from character variables.

cat = train.select_dtypes(include=['O'])

cat.apply(pd.Series.nunique)
#WorkClass

train.workclass.value_counts(sort=True)

train.workclass.fillna('Private',inplace=True)



#Occupation

train.occupation.value_counts(sort=True)

train.occupation.fillna('Prof-specialty',inplace=True)



#Native Country

train['native.country'].value_counts(sort=True)

train['native.country'].fillna('United-States',inplace=True)



train.isnull().sum()
# check the target variable to investigate if this data is imbalanced or not.

#check proportion of target variable

train.target.value_counts()/train.shape[0]
# we are dividing by train.shape[0] which no, of rows to get percentage of values



pd.crosstab(train.education, train.target,margins=True)/train.shape[0]

# pd.crosstab(train.education, train.target,margins=True)/train.shape[0]*100

# another way is to get well defined percenatge is by multiplying above function with 100
from sklearn import preprocessing



for x in train.columns:

    if train[x].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[x].values))

        train[x] = lbl.transform(list(train[x].values))



train.head()
#<50K = 0 and >50K = 1

train.target.value_counts()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

# from sklearn.cross_validation import cross_val_score



from sklearn.metrics import accuracy_score



y = train['target']

del train['target']



X = train

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,

                                                 random_state=1,stratify=y)



#train the RF classifier

clf = RandomForestClassifier(n_estimators = 500, max_depth = 6)

clf.fit(X_train,y_train)



#     RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

#                 max_depth=6, max_features='auto', max_leaf_nodes=None,

#                 min_impurity_split=1e-07, min_samples_leaf=1,

#                 min_samples_split=2, min_weight_fraction_leaf=0.0,

#                 n_estimators=500, n_jobs=1, oob_score=False, random_state=None,

#                 verbose=0, warm_start=False)



clf.predict(X_test)
#make prediction and check model's accuracy

prediction = clf.predict(X_test)

acc =  accuracy_score(np.array(y_test),prediction)

print ('The accuracy of Random Forest is {}'.format(acc))