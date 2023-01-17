import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import linear_model

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
# Loading datasets in constants

train_data = pd.read_csv('../input/train.csv')

valid_data = pd.read_csv('../input/valid.csv')

test_data = pd.read_csv('../input/test.csv') 
# Concat frames valid and test data

test_total = pd.DataFrame()

test_total = pd.concat([valid_data, test_data])

# Saving id of the final result

test_total_id = test_total['ID']

# Dropping id from datasets

train_data.drop(['ID'], axis=1, inplace=True) 

test_total.drop(['ID'], axis=1, inplace=True) 
# Visualizing all features

train_data.columns
# Visualizing object features

feature_mask = train_data.dtypes== object

cols = train_data.columns[feature_mask].tolist()

cols
# Visualizing int features

feature_mask = train_data.dtypes== int 

cols = train_data.columns[feature_mask].tolist()

cols
# Checking if there's some missing data 

total = train_data.isnull().sum().sort_values(ascending = False)

percentual = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending = False)

pd.concat([total, percentual], axis=1, keys=['Total', 'Percent']).transpose()
#Visualizing frequency of values of the colunm sex

train_data.SEX.value_counts()
#converting to categorical type 

train_data['SEX'][train_data['SEX'] == 1] = 'male'

train_data['SEX'][train_data['SEX'] == 2] = 'female'



test_total['SEX'][test_total['SEX'] == 1] = 'male'

test_total['SEX'][test_total['SEX'] == 2] = 'female'
#Visualizing frequency of values of the colunm education

train_data.EDUCATION.value_counts()
#correcting ranges of values and converting to categorical type in education column

wv_ed_train = (train_data.EDUCATION == 0) | (train_data.EDUCATION == 5) | (train_data.EDUCATION == 6) 

wv_ed_testT = (test_total.EDUCATION == 0) | (test_total.EDUCATION == 5) | (test_total.EDUCATION == 6) 



train_data.loc[wv_ed_train, 'EDUCATION'] = 4

test_total.loc[wv_ed_testT, 'EDUCATION'] = 4



train_data['EDUCATION'][train_data['EDUCATION'] == 1] = 'postGraduate'

train_data['EDUCATION'][train_data['EDUCATION'] == 2] = 'university'

train_data['EDUCATION'][train_data['EDUCATION'] == 3] = 'highSchool'

train_data['EDUCATION'][train_data['EDUCATION'] == 4] = 'otherLevel'



test_total['EDUCATION'][test_total['EDUCATION'] == 1] = 'postGraduate'

test_total['EDUCATION'][test_total['EDUCATION'] == 2] = 'university'

test_total['EDUCATION'][test_total['EDUCATION'] == 3] = 'highscool'

test_total['EDUCATION'][test_total['EDUCATION'] == 4] = 'otherLevel'
#Visualizing frequency of values of the colunm marriage

train_data.MARRIAGE.value_counts()
#correcting ranges of values and converting to categorical type in marriage column

train_data.loc[train_data.MARRIAGE == 0, 'MARRIAGE'] = 3

test_total.loc[test_total.MARRIAGE == 0, 'MARRIAGE'] = 3



train_data['MARRIAGE'][train_data['MARRIAGE'] == 1] = 'married'

train_data['MARRIAGE'][train_data['MARRIAGE'] == 2] = 'single'

train_data['MARRIAGE'][train_data['MARRIAGE'] == 3] = 'otherStatus'

test_total['MARRIAGE'][test_total['MARRIAGE'] == 1] = 'married'

test_total['MARRIAGE'][test_total['MARRIAGE'] == 2] = 'single'

test_total['MARRIAGE'][test_total['MARRIAGE'] == 3] = 'otherStatus'
#Visualizing frequency of values of the PAY colunms



train_data.PAY_0.value_counts()
train_data.PAY_2.value_counts()
train_data.PAY_3.value_counts()
train_data.PAY_4.value_counts()
train_data.PAY_5.value_counts()
train_data.PAY_6.value_counts()
#Correcting ranges of values of the pay columns

wv_pay0_train = (train_data.PAY_0 == -2) | (train_data.PAY_0 == 4) | (train_data.PAY_0 == 5) | (train_data.PAY_0 == 6) | (train_data.PAY_0 == 7) | (train_data.PAY_0 == 8)

wv_pay0_testT = (test_total.PAY_0 == -2) | (test_total.PAY_0 == 4) | (test_total.PAY_0 == 5) | (test_total.PAY_0 == 6) | (test_total.PAY_0 == 7) | (test_total.PAY_0 == 8)

train_data.loc[wv_pay0_train, 'PAY_0'] = 0

test_total.loc[wv_pay0_testT, 'PAY_0'] = 0



wv_pay2_train = (train_data.PAY_2 == -2) | (train_data.PAY_2 == 4) | (train_data.PAY_2 == 5) | (train_data.PAY_2 == 6) | (train_data.PAY_2 == 7) | (train_data.PAY_2 == 8)

wv_pay2_testT = (test_total.PAY_2 == -2) | (test_total.PAY_2 == 4) | (test_total.PAY_2 == 5) | (test_total.PAY_2 == 6) | (test_total.PAY_2 == 7) | (test_total.PAY_2 == 8)

train_data.loc[wv_pay2_train, 'PAY_2'] = 0

test_total.loc[wv_pay2_testT, 'PAY_2'] = 0



wv_pay3_train = (train_data.PAY_3 == -2) | (train_data.PAY_3 == 4) | (train_data.PAY_3 == 5) | (train_data.PAY_3 == 6) | (train_data.PAY_3 == 7) | (train_data.PAY_3 == 8)

wv_pay3_testT = (test_total.PAY_3 == -2) | (test_total.PAY_3 == 4) | (test_total.PAY_3 == 5) | (test_total.PAY_3 == 6) | (test_total.PAY_3 == 7) | (test_total.PAY_3 == 8)

train_data.loc[wv_pay3_train, 'PAY_3'] = 0

test_total.loc[wv_pay3_testT, 'PAY_3'] = 0



wv_pay4_train = (train_data.PAY_4 == -2) | (train_data.PAY_4 == 4) | (train_data.PAY_4 == 5) | (train_data.PAY_4 == 6) | (train_data.PAY_4 == 7) | (train_data.PAY_4 == 8)

wv_pay4_testT = (test_total.PAY_4 == -2) | (test_total.PAY_4 == 4) | (test_total.PAY_4 == 5) | (test_total.PAY_4 == 6) | (test_total.PAY_4 == 7) | (test_total.PAY_4 == 8)

train_data.loc[wv_pay4_train, 'PAY_4'] = 0

test_total.loc[wv_pay4_testT, 'PAY_4'] = 0



wv_pay5_train = (train_data.PAY_5 == -2) | (train_data.PAY_5 == 4) | (train_data.PAY_5 == 5) | (train_data.PAY_5 == 6) | (train_data.PAY_5 == 7) | (train_data.PAY_5 == 8)

wv_pay5_testT = (test_total.PAY_5 == -2) | (test_total.PAY_5 == 4) | (test_total.PAY_5 == 5) | (test_total.PAY_5 == 6) | (test_total.PAY_5 == 7) | (test_total.PAY_5 == 8)

train_data.loc[wv_pay5_train, 'PAY_5'] = 0

test_total.loc[wv_pay5_testT, 'PAY_5'] = 0



wv_pay6_train = (train_data.PAY_6 == -2) | (train_data.PAY_6 == 4) | (train_data.PAY_6 == 5) | (train_data.PAY_6 == 6) | (train_data.PAY_6 == 7) | (train_data.PAY_6 == 8)

wv_pay6_testT = (test_total.PAY_6 == -2) | (test_total.PAY_6 == 4) | (test_total.PAY_6 == 5) | (test_total.PAY_6 == 6) | (test_total.PAY_6 == 7) | (test_total.PAY_6 == 8)

train_data.loc[wv_pay6_train, 'PAY_6'] = 0

test_total.loc[wv_pay6_testT, 'PAY_6'] = 0
# Visualizing the correlation of the columns with the target

sns.set()

plt.figure(figsize=(10,10))

sns.heatmap(train_data.corr()[['default payment next month']], square = True, cmap='RdYlGn')

plt.show()
#Dropping LIMIT_BAL colunm

train_data.drop(['LIMIT_BAL'], axis=1, inplace=True) 

test_total.drop(['LIMIT_BAL'], axis=1, inplace=True) 
#Converting all categorical features with get dummies

new_train_data = pd.get_dummies(train_data)

new_test_total = pd.get_dummies(test_total)

#Select target and features

target = new_train_data['default payment next month']

features = new_train_data.drop('default payment next month', axis=1)



#Generate decision tree and score

tree = DecisionTreeClassifier(random_state=0, max_depth=7)

tree = tree.fit(features, target)



tree.score(features, target)
# Make prediction and generate result csv file

predict = tree.predict(new_test_total)



result = pd.DataFrame(columns=['ID', 'Default'])

result.ID = test_total_id

result.Default = predict



result.to_csv('result.csv', index=False)