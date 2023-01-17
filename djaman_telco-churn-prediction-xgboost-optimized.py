#installing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import cross_validation, metrics 
from sklearn.grid_search import GridSearchCV   #Perforing grid search

#reading the csv file 
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#snippet of dataset
df.head()
df.columns
df.describe()
df.info()
#density plot of churn column
df.isnull().sum()

#dropping the column customerID as its not needed
df = df.drop(columns = ['customerID'])
df.head()
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
#One hot encoding all the columns with values Yes No.

from sklearn.preprocessing import LabelEncoder
encoded_df = df.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)
encoded_df.head()
plt.hist(encoded_df['Churn'] )
# Correlation matrix
corr = encoded_df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

encoded_df.columns
features = encoded_df [['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']]
label = encoded_df['Churn']

#train test split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.33, random_state=7)

#function to train the model

def modelfit(alg,features_train,label_train):
    
    X_train, X_test, y_train, y_test = train_test_split(features_train, label_train, test_size=0.33, random_state=7)
    #Fit the algorithm on the data
    alg.fit(X_train, y_train)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
    
    #Predict Test set:
    dtest_predictions = alg.predict(X_test)
    dtest_predprob = alg.predict_proba(X_test)[:,1]
        
    #Print model report:
    print ("Model Report")
    #print ("Training set Accuracy : %.4g" % alg.score(y_train.values, dtrain_predictions))
    print ("Training set Accuracy : %.4g" % metrics.accuracy_score(y_train.values, dtrain_predictions))
    #print ("Test set Accuracy : %.4g" % alg.score(y_test.values, dtest_predictions))
    print ("Test set Accuracy : %.4g" % metrics.accuracy_score(y_test.values, dtest_predictions))
    print ("Training set AUC Score (Train) : %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
    print ("AUC Score (Test) : %f" % metrics.roc_auc_score(y_test, dtest_predprob))               
                                                               
    #the feat imp result will be in np array, convert it to Series so we can plot it later
    feat_imp = pd.Series(alg.feature_importances_)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
#Choose all predictors except target & IDcols
xgb1 = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, features, label)
#Base Model

xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb2, features, label)

#parameters are passed as a Dictionary with parameters names (string) as keys and lists of parameter settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.
param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6] }
gsearch2 = GridSearchCV(estimator = xgb2, 
     param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    
gsearch2.fit(X_train,y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

#Update the estimator using the parameters already tunes before that is, max_Depth and min_child_weight

xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

#parameters are passed as a Dictionary with parameters names (string) as keys and lists of parameter settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.
param_test3 = {
 'gamma':[0.1,0.2,0.3,0.4,0.5] }
gsearch3 = GridSearchCV(estimator = xgb3, 
     param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    
gsearch3.fit(X_train,y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


#Update the estimator using the parameters already tunes before that is, gamma

xgb4 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
param_test4 = {
 'learning_rate':[0.1,0.2,0.3,0.4,0.5] }
gsearch4 = GridSearchCV(estimator = xgb4, 
     param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    
gsearch4.fit(X_train,y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_



#Re- train the model with tweaked paramters 
xgb5 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb5, features, label)
