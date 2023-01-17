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
import matplotlib as mt

%matplotlib inline
data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
data.info()
data.describe()
#Pregnancies -- can be 0 but other values can't be zero --- they are missing values

features_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI']
data[features_zeros] = data[features_zeros].replace(0, np.nan)
data.isnull().sum()
# Replace nan by looking into data distributions

data.hist(figsize = (20,20))
# replace with Mean or Meadian

# Replacing with median as data distribution is either left or right Skewed

data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace=True)

data['Insulin'].fillna(data['Insulin'].median(), inplace=True)

data['BMI'].fillna(data['BMI'].median(), inplace=True)
# Replacing with mean as data looks normal distribution

data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace=True)

data['Glucose'].fillna(data['Glucose'].mean(), inplace=True)
print('Outcome 1 - ', data['Outcome'][data['Outcome']==1].count())

print('Outcome 0 - ', data['Outcome'][data['Outcome']==0].count())

print('Count - ', data['Outcome'].count())

data['Outcome'].value_counts().plot(kind='bar')
# split the data either using manual split -- 80% outcome(1) and 80% outcome(0)

outcome_1 = data[data['Outcome']==1]

outcome_0 = data[data['Outcome']==0]

train_1 = outcome_1.sample(frac=0.8)

train_0 = outcome_0.sample(frac=0.8)

train = pd.concat([train_1, train_0], axis =0)

test = data.loc[~data.index.isin(train.index)]
X_train = train.drop(['Outcome'], axis =1)

Y_train = train['Outcome']

X_test = test.drop(['Outcome'], axis =1)

Y_test = test['Outcome']
features = X_train.columns
#scale the data -- standard scaling

for feature in features:

    mean = X_train[feature].mean()

    std = X_train[feature].std()

    X_train[feature] = (X_train[feature] - mean)/std

    X_test[feature] = (X_test[feature] -mean)/std

    

    
from sklearn.neighbors import KNeighborsClassifier

test_scores = []

train_scores = []

for i in range(3,100):

    knn = KNeighborsClassifier(i)

    knn.fit(X_train,Y_train)

    train_scores.append(knn.score(X_train,Y_train))

    test_scores.append(knn.score(X_test,Y_test))
max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))