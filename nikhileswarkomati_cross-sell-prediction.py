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
train = pd.read_csv('../input/crosssell-prediction/train.csv_VsW9EGx/train.csv', index_col = 'id')

test = pd.read_csv('../input/crosssell-prediction/test.csv_yAFwdy2/test.csv', index_col = 'id')

train.sample(5)
test.sample(5)
print("size of training data", train.shape)

print("size of testing data", test.shape)
train.describe()
train.isnull().sum()


def percConvert(ser):

    return ser/float(ser['All'])

pd.crosstab(train['Gender'], train['Response'], margins = True).apply(percConvert, axis=1)
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np


sns.boxplot(train['Annual_Premium'])
sns.distplot(train['Age'], kde = False)
sns.distplot(np.log(train['Age']), kde = False)
sns.kdeplot(train['Age'])
sns.kdeplot(np.log(train['Age']))
response_0 = train.loc[(train['Response'] == 0), ['Age', 'Annual_Premium']]

response_1 = train.loc[(train['Response'] == 1), ['Age', 'Annual_Premium']]
print(response_0['Age'].mean(), response_1['Age'].mean())

print(response_0['Annual_Premium'].mean(), response_1['Annual_Premium'].mean())
sns.scatterplot(x = train['Age'], y = train['Annual_Premium'], hue = train['Response'])
cat_cols = [col for col in train.columns if train[col].dtype == 'object']

for each_cat_col in cat_cols:

    print("Distribution of", each_cat_col)

    print(train[each_cat_col].value_counts())

    print("-------------------")


def percConvert(ser):

    return ser/float(ser['All'])

for each_cat_col in cat_cols:

    print("Distribution of", each_cat_col)

    print(train[each_cat_col].value_counts())

    print("++++")

    print(pd.crosstab(train[each_cat_col], train['Response'], margins = True).apply(percConvert, axis=1))

    print("-----------------------------------------------------")
train['Region_Code'].unique()
sns.distplot(train['Region_Code'], kde = False)
import seaborn as sns
sns.distplot(train['Policy_Sales_Channel'], kde = False)
train['Policy_Sales_Channel'].value_counts()
train['Region_Code'].value_counts()
li = train['Vintage'].unique()

sorted(li)
sns.distplot(np.log(train['Vintage']), kde = False)
sns.violinplot(x = train['Response'], y = train['Region_Code'])
sns.heatmap(train.corr(), annot = True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



train['Gender'] = le.fit_transform(train['Gender'])

test['Gender'] = le.transform(test['Gender'])



train['Vehicle_Age'] = le.fit_transform(train['Vehicle_Age'])

test['Vehicle_Age'] = le.transform(test['Vehicle_Age'])



train['Vehicle_Damage'] = le.fit_transform(train['Vehicle_Damage'])

test['Vehicle_Damage'] = le.transform(test['Vehicle_Damage'])
train['Age'] = np.log(train['Age'])





test['Age'] = np.log(test['Age'])



train.head()
target = 'Response'

predictors = [x for x in train.columns if x != target]

from sklearn import model_selection, metrics

def modelfit(alg, dtrain, dtest, predictors, target, filename):

    #Fit the algorithm on the data

    alg.fit(dtrain[predictors], dtrain[target])

    print("Fit Done")   

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[predictors])

    print("Predict Done")



    #Perform cross-validation:

    cv_score = model_selection.cross_val_score(alg, dtrain[predictors], dtrain[target], scoring = 'roc_auc')

    

    #Print model report:

    print("\nModel Report")

    print("ACC : %.4g" % metrics.roc_auc_score(dtrain[target].values, dtrain_predictions))

    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    

    #Predict on testing data:

    predicted = list(alg.predict_proba(dtest[predictors]))

    #Export submission file:

    li = [ele[1] for ele in predicted]

    submission = pd.DataFrame({ 'id': list(dtest.index), 'Response': li})

    submission.to_csv(filename, index=False)
test.shape
from lightgbm import LGBMClassifier



alg1 = LGBMClassifier(learning_rate = 0.05, max_depth = 100, n_estimators = 395, objective = 'binary', boosting_type = 'gbdt')

modelfit(alg1, train, test, predictors, target, './sub1.csv')
from bayes_opt import BayesianOptimization

from skopt  import BayesSearchCV 