# import required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix
from imblearn.metrics import sensitivity_specificity_support
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_columns",250)
pd.set_option("display.max_rows",250)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/telecorm-churn-dataset/telecom_churn_data.csv')
data.head(5)
data.sample(5)
# If we try and understand the data here we have null values for the months the user hasn't recharged
# Try to simulate this record with your real life scenario
# where you are the user so you might be interested in few services of a telecomm company
data.iloc[[87286,90003],:]
data.shape
data.info(verbose=True)
data.describe(percentiles=[0.25,0.5,0.75,0.85,0.95])
columns = ["arpu_6","arpu_7","arpu_8","arpu_9"]
plt.figure(figsize=(6,4))
for col in columns:
    plt.title(col)
    sns.distplot(data[col])
    plt.show()
# create column name list by types of columns
id_cols = ['mobile_number', 'circle_id']

date_cols = ['last_date_of_month_6',
             'last_date_of_month_7',
             'last_date_of_month_8',
             'last_date_of_month_9',
             'date_of_last_rech_6',
             'date_of_last_rech_7',
             'date_of_last_rech_8',
             'date_of_last_rech_9',
             'date_of_last_rech_data_6',
             'date_of_last_rech_data_7',
             'date_of_last_rech_data_8',
             'date_of_last_rech_data_9']

cat_cols =  ['night_pck_user_6',
             'night_pck_user_7',
             'night_pck_user_8',
             'night_pck_user_9',
             'fb_user_6',
             'fb_user_7',
             'fb_user_8',
             'fb_user_9']

num_cols = [column for column in data.columns if column not in id_cols + date_cols + cat_cols]

# print the number of columns in each list
print("# ID cols: %d\n# Date cols:%d\n# Numeric cols:%d\n# Category cols:%d" % (len(id_cols), len(date_cols), len(num_cols), len(cat_cols)))

# check if we have missed any column or not
print(len(id_cols) + len(date_cols) + len(num_cols) + len(cat_cols) == data.shape[1])
print(data.isnull().sum() / data.shape[0] * 100)
# I am imuputing null values with "0" there are many ways you can impute data
# Please go ahead and select any good method that suits your given model, please share the feedback on comments sections,
# for others to take that ahead.
# I always belive that the comment section is best place to learn something as different people come with their vision.
# impute missing values with 0
zero_impute = num_cols + cat_cols
data[zero_impute].describe()
data[zero_impute] = data[zero_impute].apply(lambda x: x.fillna(0))
print(round(100 * data[zero_impute].isnull().sum() / data.shape[0],2))
# Let's check all values
print(100 * data.isnull().sum() / data.shape[0])
# Dropping Unique column or feature having less variance
cols = ['mobile_number','circle_id','last_date_of_month_6','last_date_of_month_7','last_date_of_month_8','last_date_of_month_9']
data.drop(cols,inplace=True,axis=1)
# Let's check all values
print(100 * data.isnull().sum() / data.shape[0])
date_df = data.filter(regex='^date',axis=1)
date_columns=date_df.columns
date_df.head()
date_df = date_df.apply(pd.to_datetime)
date_df.info()
date_df['day_of_last_rech_6'] = date_df.date_of_last_rech_6.dt.day
date_df['day_of_last_rech_7'] = date_df.date_of_last_rech_7.dt.day
date_df['day_of_last_rech_8'] = date_df.date_of_last_rech_8.dt.day
date_df['day_of_last_rech_9'] = date_df.date_of_last_rech_9.dt.day

date_df['day_of_last_rech_data_6'] = date_df.date_of_last_rech_data_6.dt.day
date_df['day_of_last_rech_data_7'] = date_df.date_of_last_rech_data_7.dt.day
date_df['day_of_last_rech_data_8'] = date_df.date_of_last_rech_data_8.dt.day
date_df['day_of_last_rech_data_9'] = date_df.date_of_last_rech_data_9.dt.day

date_df.head()
date_df.drop(columns=date_columns,axis=1,inplace=True)
date_df = date_df.fillna(0)
print(date_df.isnull().sum()*100/date_df.shape[0])
date_df = date_df.astype(int)
date_df.head()
# sns.heatmap(date_df.corr(),annot=True,cmap='coolwarm')
date_df['avg_day_of_last_rech_data'] = ( date_df.day_of_last_rech_data_6 + date_df.day_of_last_rech_data_7 +
                                        date_df.day_of_last_rech_data_6 + date_df.day_of_last_rech_data_9) / 4
date_df['avg_day_of_last_rech'] = (date_df.day_of_last_rech_6 + date_df.day_of_last_rech_7 +
                                        date_df.day_of_last_rech_6 + date_df.day_of_last_rech_9) / 4

date_df.drop(columns=['day_of_last_rech_6','day_of_last_rech_7','day_of_last_rech_8','day_of_last_rech_9',
                      'day_of_last_rech_data_6','day_of_last_rech_data_7','day_of_last_rech_data_8','day_of_last_rech_data_9']
             ,axis=1,
             inplace=True)
date_df = date_df.apply(lambda x: round(x,0))
date_df = date_df.apply(lambda x: x.astype(int))
date_df.head()
# combining back
data.drop(columns=date_columns,axis=1,inplace=True)
data = pd.concat([data,date_df],axis=1)
data.head()
data.shape
print(data.isnull().sum()*100/data.shape[0])
# calculate the total data recharge amount for June and July --> number of recharges * average recharge amount
data['total_data_rech_6'] = data.total_rech_data_6 * data.av_rech_amt_data_6
data['total_data_rech_7'] = data.total_rech_data_7 * data.av_rech_amt_data_7
# calculate total recharge amount for June and July --> call recharge amount + data recharge amount
data['amt_data_6'] = data.total_rech_amt_6 + data.total_data_rech_6
data['amt_data_7'] = data.total_rech_amt_7 + data.total_data_rech_7
# calculate average recharge done by customer in June and July
data['av_amt_data_6_7'] = (data.amt_data_6 + data.amt_data_7) / 2
# look at the 70th percentile recharge amount
print("Recharge amount at 70th percentile: {0}".format(data.av_amt_data_6_7.quantile(0.7)))
# retain only those customers who have recharged their mobiles with more than or equal to 70th percentile amount
churn_filtered = data.loc[data.av_amt_data_6_7 >= data.av_amt_data_6_7.quantile(0.7), :]
churn_filtered = churn_filtered.reset_index(drop=True)
churn_filtered.shape
# delete variables created to filter high-value customers
churn_filtered = churn_filtered.drop(['total_data_rech_6', 'total_data_rech_7',
                                      'amt_data_6', 'amt_data_7', 'av_amt_data_6_7'], axis=1)
churn_filtered.shape
churn_filtered.sample(5)
# calculate total incoming and outgoing minutes of usage
churn_filtered['total_calls_mou_9'] = churn_filtered.total_ic_mou_9 + churn_filtered.total_og_mou_9
# calculate 2g and 3g data consumption
churn_filtered['total_internet_mb_9'] =  churn_filtered.vol_2g_mb_9 + churn_filtered.vol_3g_mb_9
# create churn variable: those who have not used either calls or internet in the month of September are customers who have churned

# 0 - not churn, 1 - churn
churn_filtered['churn'] = churn_filtered.apply(lambda row: 1 if (row.total_calls_mou_9 == 0 and row.total_internet_mb_9 == 0) else 0,
                                               axis=1)
# delete derived variables
churn_filtered = churn_filtered.drop(['total_calls_mou_9', 'total_internet_mb_9'], axis=1)
# change data type to category
churn_filtered.churn = churn_filtered.churn.astype("category")

# print churn ratio
print("Churn Ratio:")
print(churn_filtered.churn.value_counts()*100/churn_filtered.shape[0])
churn_filtered['arpu_diff'] = churn_filtered.arpu_8 - ((churn_filtered.arpu_6 + churn_filtered.arpu_7)/2)

churn_filtered['onnet_mou_diff'] = churn_filtered.onnet_mou_8 - ((churn_filtered.onnet_mou_6 + churn_filtered.onnet_mou_7)/2)

churn_filtered['offnet_mou_diff'] = churn_filtered.offnet_mou_8 - ((churn_filtered.offnet_mou_6 + churn_filtered.offnet_mou_7)/2)

churn_filtered['roam_ic_mou_diff'] = churn_filtered.roam_ic_mou_8 - ((churn_filtered.roam_ic_mou_6 + churn_filtered.roam_ic_mou_7)/2)

churn_filtered['roam_og_mou_diff'] = churn_filtered.roam_og_mou_8 - ((churn_filtered.roam_og_mou_6 + churn_filtered.roam_og_mou_7)/2)

churn_filtered['loc_og_mou_diff'] = churn_filtered.loc_og_mou_8 - ((churn_filtered.loc_og_mou_6 + churn_filtered.loc_og_mou_7)/2)

churn_filtered['std_og_mou_diff'] = churn_filtered.std_og_mou_8 - ((churn_filtered.std_og_mou_6 + churn_filtered.std_og_mou_7)/2)

churn_filtered['isd_og_mou_diff'] = churn_filtered.isd_og_mou_8 - ((churn_filtered.isd_og_mou_6 + churn_filtered.isd_og_mou_7)/2)

churn_filtered['spl_og_mou_diff'] = churn_filtered.spl_og_mou_8 - ((churn_filtered.spl_og_mou_6 + churn_filtered.spl_og_mou_7)/2)

churn_filtered['total_og_mou_diff'] = churn_filtered.total_og_mou_8 - ((churn_filtered.total_og_mou_6 + churn_filtered.total_og_mou_7)/2)

churn_filtered['loc_ic_mou_diff'] = churn_filtered.loc_ic_mou_8 - ((churn_filtered.loc_ic_mou_6 + churn_filtered.loc_ic_mou_7)/2)

churn_filtered['std_ic_mou_diff'] = churn_filtered.std_ic_mou_8 - ((churn_filtered.std_ic_mou_6 + churn_filtered.std_ic_mou_7)/2)

churn_filtered['isd_ic_mou_diff'] = churn_filtered.isd_ic_mou_8 - ((churn_filtered.isd_ic_mou_6 + churn_filtered.isd_ic_mou_7)/2)

churn_filtered['spl_ic_mou_diff'] = churn_filtered.spl_ic_mou_8 - ((churn_filtered.spl_ic_mou_6 + churn_filtered.spl_ic_mou_7)/2)

churn_filtered['total_ic_mou_diff'] = churn_filtered.total_ic_mou_8 - ((churn_filtered.total_ic_mou_6 + churn_filtered.total_ic_mou_7)/2)

churn_filtered['total_rech_num_diff'] = churn_filtered.total_rech_num_8 - ((churn_filtered.total_rech_num_6 + churn_filtered.total_rech_num_7)/2)

churn_filtered['total_rech_amt_diff'] = churn_filtered.total_rech_amt_8 - ((churn_filtered.total_rech_amt_6 + churn_filtered.total_rech_amt_7)/2)

churn_filtered['max_rech_amt_diff'] = churn_filtered.max_rech_amt_8 - ((churn_filtered.max_rech_amt_6 + churn_filtered.max_rech_amt_7)/2)

churn_filtered['total_rech_data_diff'] = churn_filtered.total_rech_data_8 - ((churn_filtered.total_rech_data_6 + churn_filtered.total_rech_data_7)/2)

churn_filtered['max_rech_data_diff'] = churn_filtered.max_rech_data_8 - ((churn_filtered.max_rech_data_6 + churn_filtered.max_rech_data_7)/2)

churn_filtered['av_rech_amt_data_diff'] = churn_filtered.av_rech_amt_data_8 - ((churn_filtered.av_rech_amt_data_6 + churn_filtered.av_rech_amt_data_7)/2)

churn_filtered['vol_2g_mb_diff'] = churn_filtered.vol_2g_mb_8 - ((churn_filtered.vol_2g_mb_6 + churn_filtered.vol_2g_mb_7)/2)

churn_filtered['vol_3g_mb_diff'] = churn_filtered.vol_3g_mb_8 - ((churn_filtered.vol_3g_mb_6 + churn_filtered.vol_3g_mb_7)/2)
churn_filtered = churn_filtered.filter(regex='[^9]$', axis=1)
churn_filtered.drop('sep_vbc_3g',axis=1,inplace=True)
churn_filtered.shape
cat_cols
# extract all names that end with 9
col_9_names = data.filter(regex='9$', axis=1).columns

# update num_cols and cat_cols column name list
cat_cols = [col for col in cat_cols if col not in col_9_names]
cat_cols.append('churn')
num_cols = [col for col in churn_filtered.columns if col not in cat_cols]
cat_cols
# churn_filtered
def cap_outliers(array, k=3):
    upper_limit = array.mean() + k*array.std()
    lower_limit = array.mean() - k*array.std()
    array[array<lower_limit] = lower_limit
    array[array>upper_limit] = upper_limit
    return array
# cap outliers in the numeric columns
churn_filtered[num_cols] = churn_filtered[num_cols].apply(cap_outliers, axis=0)
# change churn to numeric
churn_filtered['churn'] = pd.to_numeric(churn_filtered['churn'])
# divide data into train and test
X = churn_filtered.drop("churn", axis = 1)
y = churn_filtered.churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 4, stratify = y)
# print shapes of train and test sets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
cat_cols.remove('churn')
X_train[cat_cols].head()
# del pca
# apply pca to train data
pca = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])
pca.fit(X_train)
churn_pca = pca.fit_transform(X_train)
# extract pca model from pipeline
pca = pca.named_steps['pca']

# look at explainded variance of PCA components
print(pd.Series(np.round(pca.explained_variance_ratio_.cumsum(), 4)*100))
pca.n_components_
# plot feature variance
features = range(pca.n_components_)
cumulative_variance = np.round(np.cumsum(pca.explained_variance_ratio_)*100, decimals=4)
plt.figure(figsize=(175/20,100/20)) 
# 100 elements on y-axis; 175 elements on x-axis; 20 is normalising factor
plt.plot(cumulative_variance,'g')
100 * churn_filtered['churn'].value_counts() / churn_filtered.shape[0]
# create pipeline
PCA_VARS = 60
steps = [('scaler', StandardScaler()),
         ("pca", PCA(n_components=PCA_VARS)),
         ("logistic", LogisticRegression(class_weight='balanced'))]
pipeline = Pipeline(steps)
# fit model
pipeline.fit(X_train, y_train)

# check score on train data
pipeline.score(X_train, y_train)
# predict churn on test data
y_pred = pipeline.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: ", round(sensitivity, 2), "\n", "Specificity: ", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
print("AUC :", round(roc_auc_score(y_test, y_pred_prob),2))
# Logistic model
logistic = LogisticRegression(class_weight={0:0.1, 1: 0.9})

# pipeline steps
steps=[('scaler',StandardScaler()),('pca',PCA()),('logistic',logistic)]

# Pipeline
pca_logistic = Pipeline(steps)

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# PCA Components
features = [60, 80,100]

# hyperparameter space
params = {'pca__n_components': features, 'logistic__C': [0.1, 0.5, 1, 5, 10], 'logistic__penalty': ['l1', 'l2']}

# create gridsearch object
_logistic_ = GridSearchCV(estimator=pca_logistic,
                          cv=folds,
                          param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1,
                          return_train_score=True)
_logistic_.fit(X_train,y_train)
# cross validation results
cv_results = pd.DataFrame(_logistic_.cv_results_)
cv_results.head()
plt.figure(figsize=(5,3))
plt.plot(cv_results.groupby('param_logistic__C').mean().index,
         cv_results.groupby('param_logistic__C').mean()['mean_train_score'])
plt.plot(cv_results.groupby('param_logistic__C').mean().index,
         cv_results.groupby('param_logistic__C').mean()['mean_test_score'])
plt.ylim(0.9,0.95)
plt.show()

plt.figure(figsize=(5,3))
plt.plot(cv_results.groupby('param_pca__n_components').mean().index,
         cv_results.groupby('param_pca__n_components').mean()['mean_train_score'])
plt.plot(cv_results.groupby('param_pca__n_components').mean().index,
         cv_results.groupby('param_pca__n_components').mean()['mean_test_score'])
plt.ylim(0.9,0.95)
plt.show()
# print best hyperparameters
print("Best AUC : ", round(_logistic_.best_score_,3))
print("Best hyperparameters: ", _logistic_.best_params_)
# Logistic model
logistic = LogisticRegression(class_weight={0:0.1, 1: 0.9},C=0.1,penalty='l2')

# pipeline steps
steps=[('scaler',StandardScaler()),('pca',PCA(n_components=100)),('logistic_model',logistic)]

# Pipeline
log_model = Pipeline(steps)

log_model.fit(X_train,y_train)

# predict churn on test data
y_pred = log_model.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: ", round(sensitivity, 2), "\n", "Specificity: ", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = log_model.predict_proba(X_test)[:, 1]
print("AUC :", round(roc_auc_score(y_test, y_pred),2))
# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# PCA Components
features = [60,80]

# hyperparameter space
params = {'pca__n_components': features,
          'svm_model__kernel' : ['linear', 'poly', 'rbf'],
          'svm_model__degree' : [0,1],
          'svm_model__gamma' : ['scale', 'auto']}

# pipeline steps
steps = [('scale',StandardScaler()),('pca',PCA()),('svm_model',SVC(class_weight={0:0.1, 1: 0.9}))]

# Pipeline
pca_svm = Pipeline(steps)

# create gridsearch object
_svm_ = GridSearchCV(estimator=pca_svm, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1,return_train_score=True, verbose=1)
# fitting data
_svm_.fit(X_train,y_train)
# cross validation results
cv_results = pd.DataFrame(_svm_.cv_results_)
cv_results.head()
# print best hyperparameters
print("Best ro_auc: ", _svm_.best_score_)
print("Best hyperparameters: ", _svm_.best_params_)
# SVM model
svm_model = SVC(class_weight={0:0.1,1: 0.9},gamma='scale',kernel='rbf',degree=0)

# pipeline steps
steps=[('scaler',StandardScaler()),('pca',PCA(n_components=100)),('svm_mdl',svm_model)]

# Pipeline
_svm_model_ = Pipeline(steps)

_svm_model_.fit(X_train,y_train)

# predict churn on test data
y_pred = _svm_model_.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: ", round(sensitivity, 2), "\n", "Specificity: ", round(specificity, 2), sep='')

# check area under curve
# roc_auc_score
print("AUC :", round(roc_auc_score(y_test, y_pred),2))
# ?DecisionTreeClassifier
# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# PCA Components
features = [80,100]

# hyperparameter space
params = {'pca__n_components': features,
          'DT__criterion' : ["gini", "entropy"],
          'DT__max_depth' : [1,10,20,30,50,100],
          'DT__min_samples_split' : [10,50,100,350,500],
          'DT__min_samples_leaf':[10,50,100,250,500],
          'DT__max_features':["auto","sqrt", "log2"]}

# pipeline steps
steps = [('scale',StandardScaler()),('pca',PCA()),('DT',DecisionTreeClassifier(class_weight={0:0.1, 1: 0.9}))]

# Pipeline
pca_dt = Pipeline(steps)

# create gridsearch object
_DT_ = GridSearchCV(estimator=pca_dt, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1,return_train_score=True, verbose=1)
# fitting data
_DT_.fit(X_train,y_train)
# print best hyperparameters
print("Best ro_auc: ", _DT_.best_score_)
print("Best hyperparameters: ", _DT_.best_params_)
# DT model
dt_model = DecisionTreeClassifier(class_weight={0:0.1, 1: 0.9},
                                  criterion='entropy',
                                  max_depth= 20,
                                  max_features='sqrt',
                                  min_samples_leaf=100,
                                  min_samples_split=350)

# pipeline steps
steps=[('scaler',StandardScaler()),('pca',PCA(n_components=80)),('dt_mdl',dt_model)]

# Pipeline
DT_pipe = Pipeline(steps)

DT_pipe.fit(X_train,y_train)

# predict churn on test data
y_pred = DT_pipe.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: ", round(sensitivity, 2), "\n", "Specificity: ", round(specificity, 2), sep='')

# check area under curve
# roc_auc_score
print("AUC :", round(roc_auc_score(y_test, y_pred),2))
# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# PCA Components
features = [80,100]

# pipeline steps
steps = [('scale',StandardScaler()),('pca',PCA()),('RF',RandomForestClassifier(class_weight={0:0.1, 1: 0.9}))]

# hyperparameter space
params = {'RF__max_depth': [50,100,150,200],
              'RF__min_samples_leaf': [100,200,300,400],
              'RF__min_samples_split': [200, 500],
              'RF__n_estimators': [100,200, 300], 
              'RF__max_features': [5, 10]}

# Pipeline
pca_rt = Pipeline(steps)

# create gridsearch object
_RF_ = GridSearchCV(estimator=pca_rt, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1,return_train_score=True, verbose=1)
# pca_rt.get_params().keys()
# fitting data
_RF_.fit(X_train,y_train)
# print best hyperparameters
print("Best ro_auc: ", _DT_.best_score_)
print("Best hyperparameters: ", _DT_.best_params_)