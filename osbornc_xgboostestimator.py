# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from copy import deepcopy

df = pd.read_csv('../input/train.csv')

df_ft = deepcopy(df.drop('LEAVE', axis=1))

df_ft = df_ft.drop('REPORTED_SATISFACTION', axis=1)

df_ft = df_ft.drop('REPORTED_USAGE_LEVEL', axis=1)

df_ft = df_ft.drop('CONSIDERING_CHANGE_OF_PLAN', axis=1)

df_ft = pd.get_dummies(df_ft)



df_test = pd.read_csv('../input/test.csv')

usage_test = deepcopy(df_test.REPORTED_USAGE_LEVEL)

df_test = df_test.drop('REPORTED_SATISFACTION', axis=1)

df_test = df_test.drop('REPORTED_USAGE_LEVEL', axis=1)

df_test = df_test.drop('CONSIDERING_CHANGE_OF_PLAN', axis=1)



print("Train dataset has {} samples and {} features".format(*df.shape))

print("Test dataset has {} samples and {} features".format(*df_test.shape))
df.describe()
import matplotlib.pyplot as plt

log0= df.LEAVE == 0

log1= df.LEAVE == 1

df0 = df[log0]['CONSIDERING_CHANGE_OF_PLAN']+df[log0]['REPORTED_SATISFACTION']+df[log0]['REPORTED_USAGE_LEVEL']

df1 = df[log1]['CONSIDERING_CHANGE_OF_PLAN']+df[log1]['REPORTED_SATISFACTION']+df[log1]['REPORTED_USAGE_LEVEL']

plt.hist([df0,df1],bins=16)

plt.show()
import seaborn as sns



n=len(df)

leave_0=len(df[df['LEAVE']==0])

leave_1=len(df[df['LEAVE']==1])



print("%s%% of customers choose to leave in train dataset." %(leave_1*100/n))

print("%s%% of customers choose not to leave in train dataset." %(leave_0*100/n))



fig, ax = plt.subplots(figsize=(5,4))

sns.countplot(x='LEAVE', data=df)

plt.title("Count of leave")

plt.show()
fig, ax=plt.subplots(1,figsize=(8,6))

df['HOUSE_INCOME_RATIO'] = df.HOUSE/df.INCOME

sns.boxplot(x='LEAVE',y='HOUSE_INCOME_RATIO', data=df)

ax.set_ylim(0,15)

plt.title("LEAVE vs HOUSE_INCOME_RATIO")

plt.show()
colors = ['red','blue']

for i in range(len(colors)):

    plt.scatter(np.log(df[df['LEAVE']==i]['INCOME']), np.log(df[df['LEAVE']==i]['HOUSE']), s = 0.5,c = colors[i] , label = i)

plt.legend()

plt.show()
colors = ['red','blue']

for i in range(len(colors)):

    plt.scatter(np.log(df[df['LEAVE']==i]['INCOME']), df[df['LEAVE']==i]['HANDSET_PRICE'], s = 0.5,c = colors[i] , label = i)

x = np.linspace(10,12)

plt.plot(x,0.7*x**3+0.005*x**4+5*10**(-12)*x**12+10**(-15)*x**16-585)

plt.legend()

plt.show()
fig, ax=plt.subplots(figsize=(16,6))

sns.countplot(x='LEAVE', data=df, hue='OVER_15MINS_CALLS_PER_MONTH')

ax.set_ylim(1,2000)

plt.title("Impact of OVER_15MINS_CALLS_PER_MONTH on LEAVE")

plt.show()
df.REPORTED_USAGE_LEVEL.replace(['very_little', 'little', 'avg', 'high', 'very_high'], [1,2,3,4,5], inplace=True)

df.REPORTED_SATISFACTION.replace(['very_unsat', 'unsat', 'avg', 'sat', 'very_sat'], [1,2,3,4,5], inplace=True)

df.COLLEGE.replace(['zero', 'one'], [0, 1], inplace=True)

df.CONSIDERING_CHANGE_OF_PLAN.replace(['never_thought', 'no', 'perhaps', 'considering', 'actively_looking_into_it'], [1,2,3,4,5], inplace=True)



usage = df['REPORTED_USAGE_LEVEL']

satisfaction = df['REPORTED_SATISFACTION']

college = df['COLLEGE']

change = df['CONSIDERING_CHANGE_OF_PLAN']
features = [

    'COLLEGE',

    'INCOME',

    'OVERAGE',

    'LEFTOVER',

    'HOUSE',

    'HANDSET_PRICE',

    'OVER_15MINS_CALLS_PER_MONTH',

    'AVERAGE_CALL_DURATION',

    'REPORTED_SATISFACTION',

    'REPORTED_USAGE_LEVEL',

    'CONSIDERING_CHANGE_OF_PLAN',

]

for i in range(len(features)):

    corr_df = df[features[i]]

    cor = corr_df.corr(df['LEAVE'], method='pearson')

    print("The correlation between %s and LEAVE" %features[i], cor)
from scipy.stats import chi2_contingency

for i in range(len(features)):

    csq2=chi2_contingency(pd.crosstab(df['LEAVE'], df[features[i]]))

    print("P-value of %s: " %features[i],csq2[1])
corr_df = df[features]  

cor = corr_df.corr(method='pearson')

cor
fig, ax =plt.subplots(figsize=(8, 6))

plt.title("Correlation Plot")

sns.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)

plt.show()
df['OVERAGE'].groupby(df['LEAVE']).mean()
fig, ax=plt.subplots(1,figsize=(8,6))

sns.boxplot(x='COLLEGE',y='CONSIDERING_CHANGE_OF_PLAN', data=df)

ax.set_ylim(0, 8)

plt.title("COLLEGE vs CONSIDERING_CHANGE_OF_PLAN")

plt.show()
df['LEFTOVER_OVERAGE'] = (df['LEFTOVER'].max() - df['OVERAGE'])

df['LEFTOVER_OVERAGE'].groupby(df['LEAVE']).mean()
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



df = pd.read_csv('../input/train.csv')

Y = df['LEAVE']

df = df.drop('REPORTED_SATISFACTION',axis =1)

df = df.drop('REPORTED_USAGE_LEVEL',axis =1)

df = df.drop('CONSIDERING_CHANGE_OF_PLAN',axis =1)



df = pd.get_dummies(df)



df['INCOME_HANDSET_PRICE'] = (0.7*(np.log(df.INCOME))**3 + 0.005*(np.log(df.INCOME))**4-585) - df.HANDSET_PRICE

df['MONTHLY_COST'] = (df.HANDSET_PRICE/12) + df.OVERAGE+df.LEFTOVER+(df.OVER_15MINS_CALLS_PER_MONTH*18) + df.AVERAGE_CALL_DURATION*40

df.loc[df['INCOME']<120000,'REAL_INCOME'] = df.INCOME

df.loc[df['INCOME']>120000,'REAL_INCOME'] = df.INCOME*0.65-15000

df['AGE'] = (df.REAL_INCOME-20000-3000*df.COLLEGE_one)/(1700+150*df.COLLEGE_one) + df.COLLEGE_one*4 +18



df_ft = df.drop('LEAVE', axis = 1)

df_ft = df_ft.drop('REAL_INCOME',axis =1)

df_ft = df_ft.drop('HANDSET_PRICE',axis =1)

df_ft = df_ft.drop('AVERAGE_CALL_DURATION',axis =1)

df_ft = df_ft.drop('OVER_15MINS_CALLS_PER_MONTH',axis =1)



df_ft['HOUSE'] = np.log(df_ft.HOUSE)

df_ft['INCOME'] = np.log(df_ft.INCOME)

df_ft = df_ft.drop('COLLEGE_one',axis =1)

X_train, X_test, y_train, y_test = train_test_split(df_ft, Y, test_size = 0.1,random_state = 0)
from xgboost import XGBClassifier



xgc = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                    colsample_bytree=0.75, gamma=0, learning_rate=0.01, max_delta_step=1,

                    max_depth = 4, min_child_weight=12, missing=None, n_estimators=265,

                    n_jobs=8, nthread=None, objective='binary:logistic', random_state=0,

                    reg_alpha=1, reg_lambda=0.6, scale_pos_weight=1.5, seed=None,

                    silent=True, subsample=0.8,eval_metric = 'auc')



# parameters = {

#     'max_depth': [5,6,7], #best is 5

#     'learning_rate': [0.008,0.01,0.11],

#     'n_estimators': [263,265,267],

#     'min_child_weight': [10,12,13], 

#     'max_delta_step': [1,2],

#     'subsample': [0.7,0.8,1],

#     'colsample_bytree': [0.5, 0.75,1],

#     'reg_alpha': [0.5,0.75, 1,2],

#     'reg_lambda': [0.5, 0.6,0.7],

#     'scale_pos_weight': [ 0.5, 1,1.25,1.5]

#     'random_state' : [i for i in range(10)]

# }

# gs = GridSearchCV(xgc,parameters,scoring = 'roc_auc',cv = 4,n_jobs=8)

# gs.fit(X_train,y_train)

# print(gs.best_score_)
xgc.fit(X_train,y_train)
from sklearn import metrics

from sklearn.model_selection import cross_val_score



y_pred = xgc.predict_proba(X_test)[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



score = cross_val_score(xgc,X_train,y_train,cv=5,scoring='roc_auc')

print(sum(score)/len(score),(max(score)-min(score))*100)

score
df_test = pd.read_csv('../input/test.csv')

df_test = df_test.drop('REPORTED_SATISFACTION',axis =1)

df_test = df_test.drop('REPORTED_USAGE_LEVEL',axis =1)

df_test = df_test.drop('CONSIDERING_CHANGE_OF_PLAN',axis =1)

df_test = pd.get_dummies(df_test)



df_test['INCOME_HANDSET_PRICE'] = (0.7*(np.log(df_test.INCOME))**3 + 0.005*(np.log(df_test.INCOME))**4-585) - df.HANDSET_PRICE

df_test['MONTHLY_COST'] = (df_test.HANDSET_PRICE/12) + df_test.OVERAGE+df_test.LEFTOVER+(df_test.OVER_15MINS_CALLS_PER_MONTH*18)+df_test.AVERAGE_CALL_DURATION*40



df_test.loc[df_test['INCOME']<120000,'REAL_INCOME'] = df_test.INCOME

df_test.loc[df_test['INCOME']>120000,'REAL_INCOME'] = df_test.INCOME*0.65-15000

df_test['AGE'] = (df_test.REAL_INCOME-20000-3000*df_test.COLLEGE_one)/(1700+150*df_test.COLLEGE_one) + df_test.COLLEGE_one*4 +18

df_test = df_test.drop('REAL_INCOME',axis =1)

df_test = df_test.drop('AVERAGE_CALL_DURATION',axis =1)

df_test = df_test.drop('HANDSET_PRICE',axis =1)

df_test = df_test.drop('OVER_15MINS_CALLS_PER_MONTH',axis =1)

df_test = df_test.drop('COLLEGE_one',axis =1)

df_test['HOUSE'] = np.log(df_test.HOUSE)

df_test['INCOME'] = np.log(df_test.INCOME)



test_pred = xgc.predict_proba(df_test)[:,1]

test_result = pd.DataFrame({'LEAVE':test_pred,'ID':[i for i in range(len(test_pred))]})

test_result.to_csv('testSubmit.csv',index=False)
cross_score = cross_val_score(xgc, X_train, y_train, cv=5, scoring='roc_auc')

cross_score
print(max(cross_score)-min(cross_score))
Y_test_pred = xgc.predict_proba(df_test)[:, 1]



test_result = pd.DataFrame({'LEAVE': Y_test_pred,

                            'ID': [i for i in range(Y_test_pred.size)]})



test_result.to_csv('testSubmit.csv', index=False)

xgc.get_params()
importance= pd.Series(xgc.feature_importances_,index=X_train.columns)

importance.sort_values(ascending=False)
feature_importance = xgc.feature_importances_

feat_importances = pd.Series(xgc.feature_importances_, index=df_ft.columns)

feat_importances = feat_importances.nlargest(30)

feat_importances.plot(kind='barh' , figsize=(10,10)) 