# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling as pdp

from sklearn.preprocessing import LabelEncoder, RobustScaler

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit , StratifiedKFold, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import f1_score

from sklearn.preprocessing import StandardScaler, RobustScaler

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

from sklearn.ensemble import GradientBoostingClassifier

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
traindf = pd.read_csv("../input/train_LZdllcl.csv")

testdf = pd.read_csv("../input/test_2umaH9m.csv")
sample = pd.read_csv("../input/sample_submission_M0L0uXE.csv")

sample.head()
pdp.ProfileReport(traindf)
pdp.ProfileReport(testdf)
#Combing data

y = traindf['is_promoted']

traindf_cop = (traindf.copy()).drop(['is_promoted'],axis=1)

alldata = traindf_cop.append(testdf)

alldata = alldata.drop(['employee_id'],axis=1)
#Missing value imputation for categorical variables

alldata['previous_year_rating'] = alldata["previous_year_rating"].fillna(alldata["previous_year_rating"].mode()[0])
pd.crosstab(alldata["education"], alldata["gender"])
alldata["education"] = alldata['education'].fillna("Bachelor's")

#alldata['education'].replace("Master's & above",3,inplace=True)

#alldata['education'].replace("Bachelor's",2,inplace=True)

#alldata['education'].replace("Below Secondary",1,inplace=True)



#alldata['sum_metric'] = alldata['awards_won?']+alldata['KPIs_met >80%'] + alldata['previous_year_rating']

#alldata['tot_score'] = alldata['avg_training_score'] * alldata['no_of_trainings']
alldata['life_spent_in_comp'] = alldata['length_of_service']/alldata['age']

alldata['life_spent_in_comp2'] = alldata['age'] - alldata['length_of_service']

alldata['perf'] = alldata['KPIs_met >80%']*alldata['awards_won?']
age_bins = [10,20,30,40,50,60,100]

age_labels = [1,2,3,4,5,6]  # this  is better than to do label encoding afterwards

alldata['age_binned'] = pd.cut(alldata['age'], bins=age_bins, labels=age_labels)

alldata['age_binned'] = alldata['age_binned'].astype('int64')
cols = ['department','region','gender','recruitment_channel','age_binned']

int_cols = (alldata.dtypes[alldata.dtypes != "object"].index).tolist()
sns.violinplot(x = 'department', y= 'is_promoted',  data=traindf)
sns.barplot(x = 'gender', y= 'is_promoted',  data=traindf)
sns.violinplot(y = 'KPIs_met >80%', x= 'is_promoted',  data=traindf)
#first we will make a copy of this data

alldata_le = alldata.copy()

alldata_oh = alldata.copy()
obj = (alldata_le.dtypes == ("object"))  + (alldata_le.dtypes == "category")
alldata_le.head(2)
alldata_le.dtypes
obj = (alldata_le.dtypes=='object') + (alldata_le.dtypes=='category')
le = LabelEncoder()

col = (alldata_le.dtypes[obj].index).tolist()

for i in range(0,len(col)):

    l = col[i]

    alldata_le[l] = le.fit_transform(alldata_le[l])
alldata_oh = alldata_oh.drop(['region'],axis=1)
obj = (alldata_oh.dtypes=='object') + (alldata_oh.dtypes=='category')

col = (alldata_oh.dtypes[obj].index).tolist()
alldata_oh = pd.get_dummies(alldata_oh,columns=col)
alldata_oh.head(2)
print("Number of features in modified/label encoded dataframe : " , len(alldata.columns))

print("Number of features after One hot encoding : " , len(alldata_oh.columns))
# Let's check distribution of our numeric cols    

m=1

plt.figure(figsize = (15,15))

for i in int_cols:

    plt.subplot(8,4,m)

    sns.distplot(alldata[i],kde = True)

    m = m+1
'''

from scipy.stats import skew

alldata_le_s = alldata_le.copy()

alldata_le_s[int_cols] = np.log1p(alldata_le_s[int_cols])

'''
alldata_oh_s = alldata_oh.copy()

alldata_oh_s[int_cols] = np.log1p(alldata_oh_s[int_cols])
train_data = alldata_oh.iloc[:traindf.shape[0],:]

test_data = alldata_oh.iloc[traindf.shape[0]:,:]

scaler = RobustScaler()

scaler = scaler.fit(train_data[int_cols])

train_data[int_cols] = scaler.transform(train_data[int_cols])

test_data[int_cols] = scaler.transform(test_data[int_cols])
#  split X between training and testing set

x_train, x_test, y_train, y_test = train_test_split(train_data,y, test_size=0.20, shuffle=True)
# Model-1: Using XGBClassifier

xgb = XGBClassifier(n_estimators=600,min_child_weight=5,learning_rate=0.02,

                   gamma=1,subsample=0.8,colsample_bytree=0.8,max_depth=10, random_state=123)

xgb.fit(x_train, y_train, verbose=1)

xgb_pred_prob = xgb.predict_proba(x_test)

xgb_pred = [i[1] for i in xgb_pred_prob]

thresholds = np.linspace(0.01, 0.99, 50)

mcc = np.array([f1_score(y_test, xgb_pred>thr) for thr in thresholds])



best_threshold = thresholds[mcc.argmax()]

print(mcc.max())

print(best_threshold)

#print('F1 score from XGB model: ', f1_score(forest_pred, y_test))
'''

forest = GradientBoostingClassifier(loss='exponential',max_features='auto',n_estimators=500,random_state=22)

forest.fit(x_train, y_train)

forest_pred_prob = forest.predict_proba(x_test)

forest_pred = [i[1] for i in forest_pred_prob]

thresholds = np.linspace(0.01, 0.99, 50)

mcc = np.array([f1_score(y_test, forest_pred>thr) for thr in thresholds])



best_threshold = thresholds[mcc.argmax()]

print(mcc.max())

print(best_threshold)

'''
clf = lgb.LGBMClassifier(max_depth= 8, learning_rate=0.0941, n_estimators=197, 

                         num_leaves= 17, reg_alpha=3.4492 , reg_lambda= 0.0422,random_state=223)

clf.fit(x_train, y_train, verbose=1)

lgb_pred_prob = clf.predict_proba(x_test)

lgb_pred = [i[1] for i in lgb_pred_prob]

thresholds = np.linspace(0.01, 0.99, 50)

mcc = np.array([f1_score(y_test, lgb_pred>thr) for thr in thresholds])



best_threshold = thresholds[mcc.argmax()]

print(mcc.max())

print(best_threshold)
final_prob = (lgb_pred_prob +xgb_pred_prob)/2

final_pred = [i[1] for i in final_prob]

thresholds = np.linspace(0.01, 0.99, 50)

mcc = np.array([f1_score(y_test, final_pred>thr) for thr in thresholds])



best_threshold = thresholds[mcc.argmax()]

print(mcc.max())

print(best_threshold)
#test set

lgb_pred_prob_test = clf.predict_proba(test_data)

xgb_pred_prob_test = xgb.predict_proba(test_data)

final_prob_test = (lgb_pred_prob_test +xgb_pred_prob_test)/2

final_pred_prob = [i[1] for i in final_prob_test]

final_pred = [1 if i>0.33 else 0 for i in final_pred_prob]

sub = pd.DataFrame(data = testdf['employee_id'],columns =['employee_id'])

sub['is_promoted'] = final_pred

sub.to_csv('submission.csv', index=False)