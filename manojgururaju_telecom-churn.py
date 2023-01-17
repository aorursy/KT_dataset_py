import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

from sklearn import linear_model

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from sklearn.preprocessing import StandardScaler

from pprint import pprint

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score

import os



# hide warnings

import warnings

warnings.filterwarnings('ignore')
# Importing Housing.csv

telecom = pd.read_csv("../input/telecom-customer/Telecom_customer churn.csv")
# summary of the dataset: 99999 rows, 226 columns

telecom.info()
telecom.shape
telecom.head()
# Checking columns which have missing values



telecom.isnull().mean().sort_values(ascending=False)
# columns with more than 50% missing values



column_missing_data = telecom.loc[:,telecom.isnull().mean() >= 0.5 ]

print("Number of columns with missing data {}".format(len(column_missing_data.columns)))

column_missing_data.columns
# Droping columns with more than 50% missing values

telecom = telecom.loc[:, telecom.isnull().mean() <= .5]
telecom.shape
# Droping date columns since there will be no time series analysis



date_cols = telecom.columns[telecom.columns.str.contains(pat = 'date')]

telecom = telecom.drop(date_cols, axis = 1) 
# Dropping Mobile Number



telecom = telecom.drop('mobile_number', axis = 1) 
telecom.shape
# Checking percentage of missing values in dataset



telecom.isnull().mean().sort_values(ascending=False)
# All the column names with missing values

telecom.loc[:,telecom.isnull().mean() > 0].columns
# Plotting missing values

plt.figure(figsize=(20, 5))

sns.heatmap(telecom.isnull())
# Remove Columns which have only 1 unique Value



col_list = telecom.loc[:,telecom.apply(pd.Series.nunique) == 1]

telecom = telecom.drop(col_list, axis = 1)

telecom.shape
telecom.describe()
# Storing column names before imputing



col_name = telecom.columns

col_name
# Imputing median values using SimpleImputer

from sklearn.impute import SimpleImputer



imp_mean = SimpleImputer( strategy='median') 

imp_mean.fit(telecom)

telecom = imp_mean.transform(telecom)
telecom= pd.DataFrame(telecom)

telecom.columns = col_name

telecom.head()
plt.figure(figsize=(20, 5))

sns.heatmap(telecom.isnull())
# Renaming columns



telecom.rename(columns={'jun_vbc_3g': 'vbc_3g_6', 

                        'jul_vbc_3g': 'vbc_3g_7', 

                        'aug_vbc_3g': 'vbc_3g_8', 

                        'sep_vbc_3g': 'vbc_3g_9'}, inplace=True)
total_rech_amt_6_7 = telecom[['total_rech_amt_6','total_rech_amt_7']].sum(axis=1)
# Selecting top 30 percent subscribers for churn prediction



p70 = np.percentile(total_rech_amt_6_7, 70.0)



tele_top30 = telecom[total_rech_amt_6_7 > p70]

tele_top30.shape
tele_top30['total_usage_9'] = tele_top30[['total_ic_mou_9','total_og_mou_9','vol_2g_mb_9','vol_3g_mb_9']].sum(axis=1)
tele_top30['churn'] = tele_top30['total_usage_9'].apply(lambda x: 1 if x==0 else 0 )
tele_top30.head()
mon_9_cols = tele_top30.columns[tele_top30.columns.str.contains(pat = '_9')]

mon_9_cols
tele_top30.drop(mon_9_cols, axis=1, inplace = True)

tele_top30.shape
# Converting age on network to years from days



tele_top30['aon_yr'] = round(tele_top30['aon']/365,2)

tele_top30.drop('aon',axis=1,inplace=True)
col_list = tele_top30.columns[tele_top30.columns.str.contains('_6|_7')]

len(col_list)
unique_col_list = col_list.str[:-2].unique()

len(unique_col_list)
unique_col_list
for col in unique_col_list:

    col_new_name = col+"_6_7"

    col_6_name = col+"_6"

    col_7_name = col+"_7"

    tele_top30[col_new_name] = tele_top30[[col_6_name,col_7_name]].sum(axis=1)
tele_top30.shape
tele_top30.drop(col_list, axis=1, inplace=True)

tele_top30.shape
tele_top30.head()
tele_top30.describe()
tele_top30['churn'].describe()
# Storing churn data in new dataframe

Churn = pd.DataFrame(tele_top30['churn'])



# Dropping churn column from tele_top30 before capping operation

tele_top30 = tele_top30.drop(['churn'], axis=1)
# Derving 25th and 75th percentile



Q1=tele_top30.quantile(0.25)

Q3=tele_top30.quantile(0.75)



# Deriving Inter Quartile Range

IQR=Q3-Q1



# Derving the Upper limit and Lower limit

LL = Q1 - 3*IQR

UL = Q3 + 3*IQR 
# Capping the data using Upper Limit and Lower Limit



q = [LL,UL]

tele_top30 = tele_top30.clip(LL,UL,axis=1)

print(tele_top30.shape)
tele_top30.describe()
# Removing columns which have only one value after capping operation



col_list = tele_top30.loc[:,tele_top30.apply(pd.Series.nunique) == 1]

tele_top30 = tele_top30.drop(col_list, axis = 1)

tele_top30.shape
# Adding churn column to tele_top30



tele_top30 = pd.concat([tele_top30,Churn], axis=1)

tele_top30.shape
# Plotting the correlation matrix using seaborn heatmap



corr_mat = tele_top30.corr()

plt.figure(figsize=(20, 10))

sns.heatmap(corr_mat)
# Finding the pairs of most correlated features



abs(corr_mat).unstack().sort_values(ascending = False).drop_duplicates().head(10)
# Plotting the jointplot to check correlation



sns.jointplot(x = 'total_rech_amt_6_7', y = 'arpu_6_7', data=tele_top30, kind='reg')
# Plotting the jointplot to check correlation



sns.jointplot(x = 'total_rech_amt_8', y = 'arpu_8', data=tele_top30, kind='reg', color = [255/255,152/255,150/255])
#Finding highest correlated features with churn



corr_tgt = abs(corr_mat["churn"]).sort_values(ascending = False)

top_features = corr_tgt.loc[((corr_tgt > 0.2) & (corr_tgt != 1))]

top_features
# Plotting absolute correlation value of churn with all other varibales



plt.figure(figsize=(20,5))

corr_tgt.sort_values(ascending = False).plot(kind='bar')
# Checking the imbalance in churn feature



tele_top30['churn'].value_counts()*100.0 /len(tele_top30)
plt.figure(figsize=(3, 4))

sns.countplot('churn', data=tele_top30)

plt.title('Churn distribution')

plt.show()
Interpretable_Model_df = tele_top30
y = Interpretable_Model_df.pop('churn')

X = Interpretable_Model_df

X.shape
X_cols = X.columns
# Scaling the data using standard scaler



scaler = StandardScaler()

X = scaler.fit_transform(X)
# Creating the test train split



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)
# Balancing the dataSet using SMOTE method



from imblearn.over_sampling import SMOTE



sm = SMOTE(sampling_strategy='auto', random_state=100)

X_train_bal, y_train_bal = sm.fit_sample(X_train, y_train)
print(X_train_bal.shape)

print(y_train_bal.shape)
plt.figure(figsize=(3, 4))

sns.countplot(y_train_bal)

plt.title('Churn distribution')

plt.show()
from sklearn.feature_selection import SelectFromModel



C = [100, 10, 1, 0.5, 0.1, 0.01, 0.001]



for c in C:

    lassoclf = LogisticRegression(penalty='l1', solver='liblinear', C=c).fit(X_train_bal, y_train_bal)

    model = SelectFromModel(lassoclf, prefit=True)

    X_lasso = model.transform(X_train_bal)

    print('C Value - ',c, ' selects',X_lasso.shape[1],' no. of Features')

    
lassoclf = LogisticRegression(penalty='l1', solver='liblinear', C=.001).fit(X_train_bal, y_train_bal)

model = SelectFromModel(lassoclf, prefit=True)

X_train_lasso = model.transform(X_train_bal)

pos = model.get_support(indices=True)

selected_features = list(Interpretable_Model_df.columns[pos])

print(selected_features)
X_train_lasso = pd.DataFrame(X_train_lasso)

X_train_lasso.columns = selected_features

X_train_lasso
# Defining common code



def print_all_scores(y_test, test_prediction, y_train, train_prediction):

    print('Precision on test set:\t'+str(round(precision_score(y_test,test_prediction) *100,2))+"%")

    print('Recall on test set:\t'+str(round(recall_score(y_test,test_prediction) *100,2))+"%")

    print("Training Accuracy: "+str(round(accuracy_score(y_train,train_prediction) *100,2))+"%")

    print("Test Accuracy: "+str(round(accuracy_score(y_test,test_prediction) *100,2))+"%")
# Creating a base logistic regression model

lr = LogisticRegression(random_state=100)



# Lookin at the parameters used by our base model

print('Parameters currently in use:\n')

pprint(lr.get_params())
# fit the model

lr.fit(X_train_lasso, y_train_bal)



# Predicting values

X_test_lasso = pd.DataFrame(data=X_test).iloc[:, pos]

X_test_lasso.columns = selected_features

predictions = lr.predict(X_test_lasso)

train_pred = lr.predict(X_train_lasso)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,predictions, y_train_bal, train_pred)
# Initialising logistic Regression

log_reg = LogisticRegression(random_state = 100)



# Creating hyper parameter grid

parameter_grid = {'solver': ['newton-cg', 'lbfgs','liblinear','sag'],

                  'penalty': ['l1', 'l2', 'elasticnet', 'none'],

                  'C': [100, 10, 1.0, 0.1, 0.01]}



gs = GridSearchCV(estimator=log_reg, param_grid=parameter_grid, n_jobs=-1, cv=3, scoring='accuracy', error_score=0)
# Fitting the model

grid_result = gs.fit(X_train_lasso, y_train_bal)



# Finding the best model

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Initialising hyper tuned logistic Regression

log_reg_ht = LogisticRegression(C= 1.0, penalty= 'l2', solver= 'liblinear', random_state = 100)



# Fitting the model

log_reg_ht.fit(X_train_lasso, y_train_bal)
# Predicting the labels

train_pred = log_reg_ht.predict(X_train_lasso)

test_pred = log_reg_ht.predict(X_test_lasso)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,test_pred,y_train_bal,train_pred)
# To get the weights of all the variables

weights = pd.Series(log_reg_ht.coef_[0],

                 index=selected_features)

weights.sort_values(ascending = False).plot(kind = 'bar')
# Running the random forest with default parameters.

rfc = RandomForestClassifier(random_state = 100)



# Lookin at the parameters used by our base model

print('Parameters currently in use:\n')

pprint(rfc.get_params())
# fit the model

rfc.fit(X_train_lasso,y_train_bal)



# Making predictions

predictions = rfc. predict(X_test_lasso)

train_pred = rfc. predict(X_train_lasso)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,predictions,y_train_bal,train_pred)
max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]

max_depth.append(None)



# Create the random parameter grid

parameter_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)],

                  'max_features': ['auto', 'sqrt'],

                  'max_depth': max_depth,

                  'min_samples_split': [100, 500, 1000],

                  'min_samples_leaf': [50, 250, 500],

                  'bootstrap': [True, False]}



pprint(parameter_grid)



# Searching across different combinations for best model parameters

rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = parameter_grid, n_iter = 100, 

                               cv = 3, verbose=2, random_state=100, n_jobs = -1)
# Fit the random search model

rf_random.fit(X_train_lasso, y_train_bal)



# Finding the best parameters

rf_random.best_params_
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [15, 20, 25],

    'min_samples_leaf': range(40, 50, 60),

    'min_samples_split': range(80, 100, 120),

    'n_estimators': [800, 1000, 1200], 

    'max_features': ['sqrt'],

    'bootstrap': [False]

}



# Create a based model

rf = RandomForestClassifier(random_state = 100)



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1,verbose = 1)
# Fitting the model

grid_result = grid_search.fit(X_train_lasso, y_train_bal)



# Finding the best model

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Model with the best hyperparameters



rfc = RandomForestClassifier(bootstrap=False,

                             max_depth=20,

                             min_samples_leaf=40, 

                             min_samples_split=80,

                             max_features='sqrt',

                             n_estimators=800)



# Fit

rfc.fit(X_train_lasso, y_train_bal)
# Predict

train_pred = rfc.predict(X_train_lasso)

test_pred = rfc.predict(X_test_lasso)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,predictions,y_train_bal,train_pred)
# To get the weights of all the variables

weights = pd.Series(rfc.feature_importances_,

                 index=selected_features)

weights.sort_values(ascending = False).plot(kind = 'bar')
# fit model on training data with default hyperparameters

xgb = XGBClassifier()



# Lookin at the parameters used by our base model

print('Parameters currently in use:\n')

pprint(xgb.get_params())
# Fitting the model

xgb.fit(X_train_lasso,y_train_bal)



# Making predictions

predictions = xgb.predict(X_test_lasso)

train_pred = xgb.predict(X_train_lasso)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,predictions,y_train_bal,train_pred)
# AUC Score

print("AUC Score on test set:\t" +str(round(roc_auc_score(y_test,predictions) *100,2)))
# hyperparameter tuning with XGBoost



# specify range of hyperparameters

param_grid = {'learning_rate': [0.2, 0.6], 

             'subsample': [0.3, 0.6, 0.9]}          





# specify model

xgb_ht = XGBClassifier(max_depth=2, n_estimators=200)



# set up GridSearchCV()

gs = GridSearchCV(estimator = xgb_ht, param_grid = param_grid, scoring= 'roc_auc', 

                        cv = 3, verbose = 1, return_train_score=True, n_jobs = -1)     
# Fitting the model

grid_result = gs.fit(X_train_lasso,y_train_bal) 



# Finding the best model

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Model with the best hyperparameters

xgb_ht = XGBClassifier(max_depth=2, n_estimators=200, learning_rate = 0.6, subsample = 0.9)



# Fit

xgb_ht.fit(X_train_lasso, y_train_bal)
# Predict

train_pred = xgb_ht.predict(X_train_lasso)

test_pred = xgb_ht.predict(X_test_lasso)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,predictions,y_train_bal,train_pred)
# To get the weights of all the variables

weights = pd.Series(xgb_ht.feature_importances_,

                 index=selected_features)

weights.sort_values(ascending = False).plot(kind = 'bar')
def draw_roc( y_test_churn, y_pred_churn ):

    fpr, tpr, thresholds = metrics.roc_curve(  y_test_churn, y_pred_churn,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score(  y_test_churn, y_pred_churn )

    print("ROC score: {}".format(auc_score))

    plt.figure(figsize=(6, 6))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return fpr, tpr, thresholds
# Running pca with default parameters.

pca = PCA(random_state=100)



# Fitting the model

pca.fit(X_train_bal)
# cumulative variance

var_cumu = np.cumsum(pca.explained_variance_ratio_)



# code for Scree plot

fig = plt.figure(figsize=[12,8])

plt.vlines(x=30, ymax=1, ymin=0, colors="r", linestyles="--")

plt.hlines(y=0.95, xmax=30, xmin=0, colors="g", linestyles="--")

plt.plot(var_cumu)

plt.ylabel("Cumulative variance explained")

plt.show()
# Initializing the PCA model

pca_inc = IncrementalPCA(n_components=30)



# Fitting the model

df_train_pca_inc = pca_inc.fit_transform(X_train_bal)



# Looking at the shape

df_train_pca_inc.shape
df_train_pca_inc
# Plottong correlation



corrmat = np.corrcoef(df_train_pca_inc.transpose())

plt.figure(figsize=[15,5])

sns.heatmap(corrmat)
# Applying the transformation on test



df_test_pca_inc = pca_inc.transform(X_test)

df_test_pca_inc.shape
# Creating a base logistic regression model

lr = LogisticRegression(random_state=100)



# Lookin at the parameters used by our base model

print('Parameters currently in use:\n')

pprint(lr.get_params())
# fit the model

lr.fit(df_train_pca_inc, y_train_bal)



# Predicting values

predictions = lr.predict(df_test_pca_inc)

test_pred = lr.predict(df_train_pca_inc)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,predictions,y_train_bal,train_pred)
# Initialising logistic Regression

log_reg = LogisticRegression(random_state = 100)



# Creating hyper parameter grid

parameter_grid = {'solver': ['newton-cg', 'lbfgs','liblinear','sag'],

                  'penalty': ['l1', 'l2', 'elasticnet', 'none'],

                  'C': [100, 10, 1.0, 0.1, 0.01]}



gs = GridSearchCV(estimator=log_reg, param_grid=parameter_grid, n_jobs=-1, cv=3, scoring='accuracy', error_score=0)
# Fitting the model

grid_result = gs.fit(df_train_pca_inc, y_train_bal)



# Finding the best model

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Initialising hyper tuned logistic Regression

log_reg_ht = LogisticRegression(C= 0.1, penalty= 'l2', solver= 'newton-cg', random_state = 100)



# Fitting the model

log_reg_ht.fit(df_train_pca_inc, y_train_bal)
# Predicting the labels

train_pred = log_reg_ht.predict(df_train_pca_inc)

test_pred = log_reg_ht.predict(df_test_pca_inc)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,predictions,y_train_bal,train_pred)
# Running the random forest with default parameters.

rfc = RandomForestClassifier(random_state = 100)



# Lookin at the parameters used by our base model

print('Parameters currently in use:\n')

pprint(rfc.get_params())
# fit the model

rfc.fit(df_train_pca_inc,y_train_bal)



# Making predictions

predictions = rfc.predict(df_test_pca_inc)

train_pred = rfc.predict(df_train_pca_inc)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,predictions,y_train_bal,train_pred)
max_depth = [int(x) for x in np.linspace(10, 30, num = 3)]



# Create the random parameter grid

parameter_grid = {'n_estimators': [int(x) for x in np.linspace(start = 600, stop = 1000, num = 5)],

                  'max_features': ['auto', 'sqrt'],

                  'max_depth': max_depth,

                  'min_samples_split': [500, 1000],

                  'min_samples_leaf': [250, 500],

                  'bootstrap': [True, False]}



pprint(parameter_grid)



# Searching across different combinations for best model parameters

rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = parameter_grid, n_iter = 50, 

                               cv = 3, verbose=2, random_state=100, n_jobs = -1)
# Fit the random search model

rf_random.fit(df_train_pca_inc, y_train_bal)



# Finding the best parameters

rf_random.best_params_
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [15, 20, 25],

    'min_samples_leaf': range(200, 300, 50),

    'min_samples_split': range(400, 600, 100),

    'n_estimators': [800, 1000, 1200], 

    'max_features': ['auto'],

    'bootstrap': [False]

}



# Create a based model

rf = RandomForestClassifier(random_state = 100)



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1,verbose = 1)
# Fitting the model

grid_result = grid_search.fit(df_train_pca_inc, y_train_bal)



# Finding the best model

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Model with the best hyperparameters



rfc = RandomForestClassifier(bootstrap=False,

                             max_depth=20,

                             min_samples_leaf=200, 

                             min_samples_split=400,

                             max_features='auto',

                             n_estimators=800)



# Fit

rfc.fit(df_train_pca_inc, y_train_bal)
# Predict

train_pred = rfc.predict(df_train_pca_inc)

test_pred = rfc.predict(df_test_pca_inc)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,predictions,y_train_bal,train_pred)
# fit model on training data with default hyperparameters

xgb = XGBClassifier()



# Lookin at the parameters used by our base model

print('Parameters currently in use:\n')

pprint(xgb.get_params())
# Fitting the model

xgb.fit(df_train_pca_inc,y_train_bal)



# Making predictions

predictions = xgb.predict(df_test_pca_inc)

train_pred = xgb.predict(X_train_lasso)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,predictions,y_train_bal,train_pred)
# AUC Score

print("AUC Score on test set:\t" +str(round(roc_auc_score(y_test,predictions) *100,2)))
# hyperparameter tuning with XGBoost



# specify range of hyperparameters

param_grid = {'learning_rate': [0.2, 0.6], 

             'subsample': [0.3, 0.6, 0.9]}          





# specify model

xgb_ht = XGBClassifier(max_depth=2, n_estimators=200)



# set up GridSearchCV()

gs = GridSearchCV(estimator = xgb_ht, param_grid = param_grid, scoring= 'roc_auc', 

                        cv = 3, verbose = 1, return_train_score=True, n_jobs = -1)     
# Fitting the model

grid_result = gs.fit(df_train_pca_inc,y_train_bal) 



# Finding the best model

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Model with the best hyperparameters

xgb_ht = XGBClassifier(max_depth=2, n_estimators=200, learning_rate = 0.6, subsample = 0.9)



# Fit

xgb_ht.fit(df_train_pca_inc, y_train_bal)
# Predict

train_pred = xgb_ht.predict(df_train_pca_inc)

test_pred = xgb_ht.predict(df_test_pca_inc)
# Accuracy, precision, recall/sensitivity of the model

print_all_scores(y_test,predictions,y_train_bal,train_pred)