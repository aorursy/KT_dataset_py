import pandas as pd

import numpy as np

customer = pd.read_csv("../input/customer.csv")

test_file = pd.read_csv("../input/test-file.csv")

train = pd.read_csv("../input/train.csv")
train.head()
test_file.head()
customer.head()
cust = customer.copy()

cust.drop(['cust_type',"cust_approach_date"], axis=1, inplace=True)

cust.head()
cust['city'] = cust['city'].str.replace('city ' ,'')

cust['cust_group'] = cust['cust_group'].str.replace('group ' ,'')

cust['cust_attrb_1'] = cust['cust_attrb_1'].str.replace('cat ' ,'')

cust['cust_attrb_2'] = cust['cust_attrb_2'].str.replace('cat ' ,'')

cust['cust_attrb_3'] = cust['cust_attrb_3'].str.replace('cat ' ,'')

cust['cust_attrb_4'] = cust['cust_attrb_4'].str.replace('cat ' ,'')

cust['cust_attrb_5'] = cust['cust_attrb_5'].str.replace('cat ' ,'')

cust['cust_attrb_6'] = cust['cust_attrb_6'].str.replace('cat ' ,'')

cust['cust_attrb_7'] = cust['cust_attrb_7'].str.replace('cat ' ,'')

cust.head()
cust.info()
cust[["city", "cust_group", "cust_attrb_1", "cust_attrb_2", "cust_attrb_3", "cust_attrb_4", "cust_attrb_5", "cust_attrb_6", "cust_attrb_7"]] = cust[["city", "cust_group", "cust_attrb_1", "cust_attrb_2", "cust_attrb_3", "cust_attrb_4", "cust_attrb_5", "cust_attrb_6", "cust_attrb_7"]].apply(pd.to_numeric)
cust.info()
train1 = train.copy()

train1.head()
train1.drop(['product_id'], axis=1, inplace=True)

train1.head()
train1['year'] = pd.DatetimeIndex(train1['visit_date']).year

train1['month'] = pd.DatetimeIndex(train1['visit_date']).month

train1.drop(['visit_date'], axis=1, inplace=True)

train1.head()
train1['campaign_category'] = train1['campaign_category'].str.replace('type ' ,'')

train1['prod_char_1'] = train1['prod_char_1'].str.replace('cat ' ,'')

train1['prod_char_2'] = train1['prod_char_2'].str.replace('cat ' ,'')

train1['prod_char_3'] = train1['prod_char_3'].str.replace('cat ' ,'')

train1['prod_char_4'] = train1['prod_char_4'].str.replace('cat ' ,'')

train1['prod_char_5'] = train1['prod_char_5'].str.replace('cat ' ,'')

train1['prod_char_6'] = train1['prod_char_6'].str.replace('cat ' ,'')

train1['prod_char_7'] = train1['prod_char_7'].str.replace('cat ' ,'')

train1.head()
# Replacing Nan with 0's

train1.replace(np.nan, 0, inplace=True)

train1.head()
train1.info()
train1[["campaign_category", "prod_char_1", "prod_char_2", "prod_char_3", "prod_char_4", "prod_char_5", "prod_char_6", "prod_char_7"]] = train1[["campaign_category", "prod_char_1", "prod_char_2", "prod_char_3", "prod_char_4", "prod_char_5", "prod_char_6", "prod_char_7"]].apply(pd.to_numeric)

train1.info()
test = test_file.copy()

test.drop(['product_id'], axis=1, inplace=True)

test['year'] = pd.DatetimeIndex(test['visit_date']).year

test['month'] = pd.DatetimeIndex(test['visit_date']).month

test.drop(['visit_date'], axis=1, inplace=True)

test['campaign_category'] = test['campaign_category'].str.replace('type ' ,'')

test['prod_char_1'] = test['prod_char_1'].str.replace('cat ' ,'')

test['prod_char_2'] = test['prod_char_2'].str.replace('cat ' ,'')

test['prod_char_3'] = test['prod_char_3'].str.replace('cat ' ,'')

test['prod_char_4'] = test['prod_char_4'].str.replace('cat ' ,'')

test['prod_char_5'] = test['prod_char_5'].str.replace('cat ' ,'')

test['prod_char_6'] = test['prod_char_6'].str.replace('cat ' ,'')

test['prod_char_7'] = test['prod_char_7'].str.replace('cat ' ,'')

test.replace(np.nan, 0, inplace=True)

test.head()
len(train1)
train_data = pd.merge(train1, cust, on='cust_id', how='left')

train_data.drop(['cust_id'], axis=1, inplace=True)

train_data.head()
len(train_data)
train_data.info()

# there are no null items
len(test)
test_data = pd.merge(test, cust, on='cust_id', how='left')

test_data.drop(['cust_id'], axis=1, inplace=True)

test_data.head()
len(test_data)
test_data.info()
test_data[["campaign_category", "prod_char_1", "prod_char_2", "prod_char_3", "prod_char_4", "prod_char_5", "prod_char_6", "prod_char_7"]] = test_data[["campaign_category", "prod_char_1", "prod_char_2", "prod_char_3", "prod_char_4", "prod_char_5", "prod_char_6", "prod_char_7"]].apply(pd.to_numeric)
test_data.info()
### Checking the Churn Rate

churn = (sum(train_data['purchase'])/len(train_data['purchase'].index))*100

churn
from sklearn.model_selection import train_test_split
# Putting feature variable to X

X = train_data.drop(['purchase'], axis=1)

X.head()
X.shape
# Putting response variable to y

y = train_data['purchase']

y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
from sklearn.feature_selection import RFE

rfe = RFE(logreg, 15)             # running RFE with 20 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'purchase':y_train.values, 'purchase_Prob':y_train_pred})

y_train_pred_final['CustID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.purchase_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.purchase, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.purchase, y_train_pred_final.predicted))
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final['Purchase_Prob'] = y_train_pred
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Purchase_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.purchase, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.purchase, y_train_pred_final.predicted))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned

print(FP/ float(TN+FP))
# positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
# Importing matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.purchase, y_train_pred_final.Purchase_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.purchase, y_train_pred_final.Purchase_Prob)
# Finding the Optiomal Cutoff

# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Purchase_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.purchase, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
# Considering 0.45 as optiomal cutoff

y_train_pred_final['final_predicted'] = y_train_pred_final.Purchase_Prob.map( lambda x: 1 if x > 0.45 else 0)



y_train_pred_final.head()
len(y_train_pred_final)
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.purchase, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.purchase, y_train_pred_final.final_predicted )

confusion2
len(test_file)
X_test = test_data[col]

X_test.head()
len(X_test)
X_test_sm = sm.add_constant(X_test)
len(X_test_sm)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
len(y_test_pred)
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)

# Let's see the head

y_pred_1.head()
len(y_pred_1)
# Putting CustID to index

y_pred_1['cust_id'] = test_file['cust_id']

y_pred_1.head()
# Renaming the column 

y_pred_1= y_pred_1.rename(columns={ 0 : 'Purchase_Prob'})
y_pred_1['purchase'] = y_pred_1.Purchase_Prob.map(lambda x: 1 if x > 0.45 else 0)
y_pred_1.head()
y_pred_2 = y_pred_1[['cust_id', 'purchase']]

y_pred_2.head()
y_pred_2.to_csv("spardhan_LR.csv", index = False)
y_pred_1['purchase'] = y_pred_1.Purchase_Prob.map(lambda x: 1 if x > 0.4 else 0)

y_pred_3 = y_pred_1[['cust_id', 'purchase']]

y_pred_3.to_csv("spardhan_LR1.csv", index = False)
# Splitting the data into train and test

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb

import scipy.stats as st

from sklearn import model_selection





params = {  

    "n_estimators": range(5,5000,20),

    "max_depth": range(2,2000,20),

    "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002, 0.0002, 0.3, 0.03, 0.003, 0.0003, 0.4, 0.04, 0.004, 0.0004, 0.5, 0.05, 0.005, 0.0005, 0.6, 0.06, 0.006, 0.0006, 0.7, 0.07, 0.007, 0.0007, 0.8, 0.08, 0.008, 0.0008, 0.9, 0.09, 0.009, 0.0009],

    "colsample_bytree": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],

    "subsample": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],

    "gamma": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.75, 1.8, 1.9, 2],

    'reg_alpha': [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],

    "min_child_weight": range(1,30,1),

}



xgbreg = XGBClassifier(nthreads=-1) 

from sklearn.model_selection import RandomizedSearchCV



gs = RandomizedSearchCV(xgbreg, params, n_jobs=1)

model_xgboost = gs.fit(X_train_xgb, y_train_xgb)
# Model best estimators

print("Learning Rate: ", model_xgboost.best_estimator_.get_params()["learning_rate"])

print("Gamma: ", model_xgboost.best_estimator_.get_params()["gamma"])

print("Max Depth: ", model_xgboost.best_estimator_.get_params()["max_depth"])

print("Subsample: ", model_xgboost.best_estimator_.get_params()["subsample"])

print("Max Features at Split: ", model_xgboost.best_estimator_.get_params()["colsample_bytree"])

print("Alpha: ", model_xgboost.best_estimator_.get_params()["reg_alpha"])

print("Lamda: ", model_xgboost.best_estimator_.get_params()["reg_lambda"])

print("Minimum Sum of the Instance Weight Hessian to Make a Child: ",

      model_xgboost.best_estimator_.get_params()["min_child_weight"])

print("Number of Trees: ", model_xgboost.best_estimator_.get_params()["n_estimators"])
model_xgboost.best_estimator_
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.9, gamma=0.2,

              learning_rate=0.005, max_delta_step=0, max_depth=75,

              min_child_weight=2, missing=None, n_estimators=1000, n_jobs=1,

              nthread=None, nthreads=-1, objective='binary:logistic',

              random_state=0, reg_alpha=0.8, reg_lambda=1, scale_pos_weight=1,

              seed=None, silent=None, subsample=0.7, verbosity=1)

model.fit(X_train_xgb, y_train_xgb)
# make predictions for test data

y_pred = model.predict(X_test_xgb)

predictions = [round(value) for value in y_pred]
# evaluate predictions

accuracy = accuracy_score(y_test_xgb, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
X_test1 = test_data.copy()

X_test1.head()
X_test1.info()
# make predictions for test data

y_pred = model.predict(X_test1)

predictions = [round(value) for value in y_pred]
len(predictions)
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(predictions)

# Let's see the head

y_pred_1.head()
# Putting CustID to index

y_pred_1['cust_id'] = test_file['cust_id']

y_pred_1.head()
# Renaming the column 

y_pred_1= y_pred_1.rename(columns={ 0 : 'purchase'})

y_pred_1.head()
y_pred_4 = y_pred_1[['cust_id', 'purchase']]

y_pred_4.head()
y_pred_4.to_csv("spardhan_XGB2.csv", index = False)
# Splitting the data into train and test

X_train_gbm, X_test_gbm, y_train_gbm, y_test_gbm = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm

from sklearn.model_selection import GridSearchCV

# fit model no training data

model1 = GradientBoostingClassifier(random_state=10)

model1.fit(X_train_gbm, y_train_gbm)
# make predictions for test data

y_pred = model1.predict(X_test_gbm)

predictions = [round(value) for value in y_pred]
# evaluate predictions

accuracy = accuracy_score(y_test_gbm, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
#param_test1 = {'n_estimators':range(10,1000,100)}

#gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1), 

#param_grid = param_test1, scoring='roc_auc',iid=False, cv=5)

#gsearch1.fit(X_train_gbm, y_train_gbm)



params = {  

    "n_estimators": [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 250, 500, 1000],

    "max_depth": [1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45, 47, 50, 52, 55, 57, 60, 62, 65, 67, 70, 73, 75, 77, 80, 83, 85, 87, 90, 93, 95, 97, 100],

    "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002, 0.0002, 0.3, 0.03, 0.003, 0.0003, 0.4, 0.04, 0.004, 0.0004, 0.5, 0.05, 0.005, 0.0005, 0.6, 0.06, 0.006, 0.0006, 0.7, 0.07, 0.007, 0.0007, 0.8, 0.08, 0.008, 0.0008, 0.9, 0.09, 0.009, 0.0009],

    "subsample": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],

    'min_samples_split':range(10,2000,20),

    'min_samples_leaf':range(10,1000,10),

    'max_features':range(2,30,2),

}



gbmreg = GradientBoostingClassifier(random_state=10) 

from sklearn.model_selection import RandomizedSearchCV



gs = RandomizedSearchCV(gbmreg, params, n_jobs=1)

model_xgboost = gs.fit(X_train_gbm, y_train_gbm)
model_xgboost.best_estimator_
model = GradientBoostingClassifier(criterion='friedman_mse', init=None,

                           learning_rate=0.03, loss='deviance', max_depth=45,

                           max_features=14, max_leaf_nodes=None,

                           min_impurity_decrease=0.0, min_impurity_split=None,

                           min_samples_leaf=80, min_samples_split=210,

                           min_weight_fraction_leaf=0.0, n_estimators=105,

                           n_iter_no_change=None, presort='auto',

                           random_state=10, subsample=0.6, tol=0.0001,

                           validation_fraction=0.1, verbose=0,

                           warm_start=False)

model.fit(X_train_gbm, y_train_gbm)
# make predictions for test data

y_pred = model.predict(X_test_gbm)

predictions = [round(value) for value in y_pred]
# evaluate predictions

accuracy = accuracy_score(y_test_gbm, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# make predictions for test data

y_pred = model.predict(X_test1)

predictions = [round(value) for value in y_pred]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(predictions)

# Let's see the head

y_pred_1.head()
# Putting CustID to index

y_pred_1['cust_id'] = test_file['cust_id']

y_pred_1.head()
# Renaming the column 

y_pred_1= y_pred_1.rename(columns={ 0 : 'purchase'})

y_pred_1.head()
y_pred_5 = y_pred_1[['cust_id', 'purchase']]

y_pred_5.head()
y_pred_5.to_csv("spardhan_GBM1.csv", index = False)