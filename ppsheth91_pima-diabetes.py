import os
import pandas as pd
import numpy as np
os.chdir("E:\Data Science\R Programs\Csv files")
os.getcwd()
df = pd.read_csv("diabetes.csv")
df.head()
df.info()
# Pre processing tghe data #

# visualize the data #

df.info()

#1)  there are no missing values #

df.Outcome.head() 
# 1 : patient has diabetes and 0 : patieny does not have diabtes #
df.head()
# 2) There are no duplicate entrries

df = df.drop_duplicates()

len(df)

# ther are no duplocates enteroes in the data
# 3) Check for data redundancy #, same column or different columns have same data, there is none
# Data Visualization, to observe the relatiobnship between the data elements #

# # needs to be relationship between independent & dependent variable  - visualization 
# 1. relationship between categorical & mumerical - box plot
# 2. relationship between categprical & categoruical - bar chart (countplot)
# 3. relationship between numerical & mumerical - scatter plot 

## relationship between independent variables - corr plot or heat map
# The output variables is Outcome of the patient L: Diabetic or non-diabetic #
# Visialization #

import matplotlib.pyplot as plt
import seaborn as sns
df.columns
sns.boxplot(x=df['Outcome'],y=df['Pregnancies'])
plt.show()

# There ia a relation between diabetes and pregancies# 
sns.boxplot(x=df['Outcome'],y=df['Glucose'])
plt.show()

# There ia a relation between diabetes and Glucose#  
# Persons having high glucose are likely to have diabetes 
sns.boxplot(x=df['Outcome'],y=df['BloodPressure'])
plt.show()

# There ia a relation between diabetes and Glucose# 
sns.boxplot(x=df['Outcome'], y=df['SkinThickness'])
plt.show() 

sns.boxplot(x=df['Outcome'], y=df['Insulin'])
plt.show() 

sns.boxplot(x=df['Outcome'], y=df['BMI'])
plt.show() 

sns.boxplot(x=df['Outcome'], y=df['DiabetesPedigreeFunction'])
plt.show() 

sns.boxplot(x=df['Outcome'], y=df['Age'])
plt.show() 

# Fronm this it can said all independent variables are significant as there is realatiobship between dependent and independent variables
# Splitting the dependent and independent variables #

Y = df['Outcome']
Y.head()
# Independent variables #
X = df.drop(['Outcome'],axis=1)
X.head()
# Correlation plot #

X.corr()

# All absolute values of correlation coefficents are less than 0.7 yhere is no multo collinearity
X.head()
# We have multiple locations where there the observation is zero and value is missing #

a_Pregnancies = X.loc[X['Pregnancies']==0]
print(len(a_Pregnancies))

b_Glucose = X.loc[X['Glucose']==0]
print(len(b_Glucose))

c_BloodPressure = X.loc[X['BloodPressure']==0]
print(len(c_BloodPressure))

d_SkinThickness = X.loc[X['SkinThickness']==0]
print(len(d_SkinThickness))

e_Insulin = X.loc[X['Insulin']==0]
print(len(e_Insulin))

e_BMI= X.loc[X['BMI']==0]
print(len(e_BMI))

e_DiabetesPedigreeFunction= X.loc[X['DiabetesPedigreeFunction']==0]
print(len(e_DiabetesPedigreeFunction))

e_Age= X.loc[X['Age']==0]
print(len(e_Age))
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif
# We will impute the missing values with mean of that column #

from sklearn.impute import SimpleImputer

Zero_values = SimpleImputer(missing_values=0, strategy ='mean')  # by default it will impite the missing values column wise#

X = pd.DataFrame(Zero_values.fit_transform(X), columns = X.columns)
X.head()
a_Pregnancies = X.loc[X['Pregnancies']==0]
print(len(a_Pregnancies))

b_Glucose = X.loc[X['Glucose']==0]
print(len(b_Glucose))

c_BloodPressure = X.loc[X['BloodPressure']==0]
print(len(c_BloodPressure))

d_SkinThickness = X.loc[X['SkinThickness']==0]
print(len(d_SkinThickness))

e_Insulin = X.loc[X['Insulin']==0]
print(len(e_Insulin))

e_BMI= X.loc[X['BMI']==0]
print(len(e_BMI))

e_DiabetesPedigreeFunction= X.loc[X['DiabetesPedigreeFunction']==0]
print(len(e_DiabetesPedigreeFunction))

e_Age= X.loc[X['Age']==0]
print(len(e_Age))
# Feature Engineering : 
# 1) Feature scaling : Minmax, standard scaler  : To bring the variabkes to the same scale
# 2) Feature Extraction : PCA ( we will use PCA to reduce the no. of variabkes and reduce the multi collinearity)
# 3) Feature transformation : converting catrgorical inyo numerical ( dont have any categorical variables)

# Splitting the data frame into test abd train sets #

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=30)
print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(Y_train.shape)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
# Applying the Decision tree algorithm #

dtc = DecisionTreeClassifier()
mo = dtc.fit(X_train,Y_train)
mo

# Predicting the train data set #

y_train_pred = mo.predict(X_train)

conf_tr = confusion_matrix(y_train_pred,Y_train)
conf_tr
Acc = accuracy_score(Y_train,y_train_pred)
Acc
# predictions on test data set #

y_test_pred = mo.predict(X_test)

conf_te = confusion_matrix(y_test_pred,Y_test)
conf_te

Acc = accuracy_score(y_test_pred,Y_test)
print(Acc)
# Building the model with changes parameters #

dt = DecisionTreeClassifier(criterion='gini',max_leaf_nodes=6)
# training #

mo1 = dt.fit(X_train,Y_train)
mo1
# Predicting the train data set # # Acc -0.772

y_train_pred = mo1.predict(X_train)

conf_tr = confusion_matrix(y_train_pred,Y_train)
conf_tr
Acc = accuracy_score(Y_train,y_train_pred)
Acc

# predictions on test data set # Acc- 0.792

y_test_pred = mo1.predict(X_test)

conf_te = confusion_matrix(y_test_pred,Y_test)
conf_te

Acc = accuracy_score(y_test_pred,Y_test)
print(Acc)
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test_pred,Y_test)
score
# Random Forest Model #
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=15,max_features=2, min_samples_leaf=18)
rf
mo1 = rf.fit(X_train,Y_train)
y_pred_train = mo1.predict(X_train)
conf_tr = confusion_matrix(y_train_pred,Y_train)
conf_tr
Acc = accuracy_score(y_pred_train,Y_train)
print(Acc)

# train_accuracy score : 0.802
# test data set #

y_pred_test = mo1.predict(X_test)

acc = accuracy_score(y_pred_test,Y_test)

print(acc)

# test_accuracy score : 0.766
# Hyper parameter tuning along wiht usung the randome forest alforirhm #

# max_depth = 2,3,4
#max_leaf_nodes = 4,5,6,7
# min_sample_leaf = 15,18,20
# max_features =2,3,4
# n_estimators = 7,10,15,20

params = {'min_samples_leaf':[12,15,17,18,20],'n_estimators':[10,12,15,18,20,22],'max_features':[2,3,4]}

# 'max_depth':[2,3,4], 'min_samples_leaf':[15,18], 
from sklearn.model_selection import GridSearchCV

r = RandomForestClassifier()
grid = GridSearchCV(r,params,cv=10)
mo_gr = grid.fit(X_train,Y_train)

# features : 3, n_esto : 10, max_doth =2, min_samp,_leaf :15, max_leaf_noed :5
print(grid.best_score_)  # best score with best parameters #
print(grid.best_params_)
print(grid.best_estimator_)
estimator = RandomForestClassifier(max_features=2, min_samples_leaf=20, n_estimators=10)
from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator,X_train,Y_train,cv=15)
score
score.mean() 
# mean score : 0.7586
cp = mo_gr.best_estimator_
cp
# testing model of random forest # 

y_pred_test = cp.predict(X_test)

con = confusion_matrix(y_pred_test,Y_test)


acc = accuracy_score(y_pred_test,Y_test)
print(acc)

# best test accuracy : 0.7922 with parameters : 'max_features': 2, 'n_estimators': 15, 'min_samples_leaf': 18
cl = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=3, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=12, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=12, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
# CV : 10, train_accuracy : 0.779, test acc : 0.791, {'max_features': 4, 'n_estimators': 10, 'min_samples_leaf': 12}
params = {'min_samples_leaf':[9,10,12,15,17,18,20],'n_estimators':[7,10,12,15,18,20,22],'max_features':[2,3,4]}
# Random Search CV #

from sklearn.model_selection import RandomizedSearchCV

r = RandomForestClassifier()
grid = RandomizedSearchCV(estimator=r,param_distributions=params,cv=10,n_iter=20,n_jobs=-1,scoring='roc_auc')
mo_rf = grid.fit(X_train,Y_train)
print(mo_rf.best_score_)  # best score with best parameters #
print(mo_rf.best_params_)
print(mo_rf.best_estimator_)
cp = mo_rf.best_estimator_
mo = RandomForestClassifier(max_features=4, min_samples_leaf=12, n_estimators=10)
# checking the cross validated score for this model #

score = cross_val_score(mo,X_train,Y_train,cv=10)
score
score.mean()


#score : 0.76

# so if we take any random data, we can get an accuract of ~76%
# testing model of random forest # 

y_pred_test = cp.predict(X_test)

con = confusion_matrix(y_pred_test,Y_test)

acc = accuracy_score(y_pred_test,Y_test)
print(acc)

# test accuracy of 0.75# 
# We need to ave the model #

import pickle

pickle.dump(mo_rf,open('diabetes_predict.pkl','wb'))
a = pickle.load(open('diabetes_predict.pkl','rb'))
X_test.iloc[0,:]
X_test.columns
Y_test.head(20)
# Lets test a data,whether it is abloe to predict correctly # for record no: 226

d = a.predict([[8,151,78,32,210,43,0.5,36]])
d
# Lets test a data,whether it is abloe to predict correctly # for record no: 226

c = a.predict([[3.0,111,56,39,155.5,30.1,0.557,30.0]])
c
# Lets test a data,whether it is abloe to predict correctly # for record no: 226

m = a.predict([[4.494673,101.0,76.0,29.15,155.54,35.70,0.19,26.0]])
m
# Lets test a data,whether it is abloe to predict correctly # for record no: 610

n = a.predict([[3,104,54,21,158,30.90,0.292,24.0]])
n
m = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=4, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=18, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=15, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0,0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
## Hyperparameter optimization using RandomizedSearchCV
from xgboost import XGBClassifier

! pip install xgboost
import xgboost
classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=15,scoring='roc_auc',n_jobs=-1,cv=10,verbose=3)
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
model = random_search.fit(X_train,Y_train)
timer(start_time) # timing ends here for "start_time" variable
print(model.best_score_)
print(model.best_params_)
print(model.best_estimator_)
a = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              max_features=4, min_child_weight=1, min_samples_leaf=18,
              missing=np.nan, monotone_constraints='()', n_estimators=22, n_jobs=0,
              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1, tree_method='exact',
              validate_parameters=1, verbosity=None)
cp = model.best_estimator_
cp
# testing model of random forest #

y_pred_test = cp.predict(X_test)

acc = accuracy_score(y_pred_test,Y_test)

print(acc)

# test_accuracy : 0.812