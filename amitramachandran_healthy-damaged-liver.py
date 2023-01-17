# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#importing the dataset using pandas into a variable
liverdata = pd.read_csv("../input/indian_liver_patient.csv")
#peeking into the dataset
liverdata.head()
#descriptive statistics
# findings :- 583 rows & 11 columns , albumin and Globulin ratio has 4 null values
liverdata.info()
#findings :- The numeric values are not normalised. 
liverdata.describe()
# findings :- only 2 values are there in the column dataset and this seems to be the target value
set(liverdata['Dataset'])
# actions :- mapping value 2 to 0 since 0 means disease free and 1 is diseased 
li = {2:0}
liverdata.replace({'Dataset':li},inplace=True)
#findings :- from the dataset description 416 are diseased while 167 are healthy people. 
liverdata['Dataset'].value_counts()
# action :- finding the null values.
liverdata[liverdata['Albumin_and_Globulin_Ratio'].isnull()]
liverdata[liverdata['Albumin']==3.9].mean()
liverdata[liverdata['Albumin']==3.1].mean()
liverdata[liverdata['Albumin']==4.8].mean()
liverdata[liverdata['Albumin']==2.7].mean()
#action :- trying to fill the A/G ratio according to the albumin mean
values = {209:1.194,241:0.932,253:0.855,312:1.50}
liverdata['Albumin_and_Globulin_Ratio'].fillna(value=values,inplace=True)
#findings :- filled the null values with the new mean values. 
liverdata.info()
#action = to numerize the gender column 
dummy = pd.get_dummies(liverdata['Gender'])
#concated the dummy df with the original df 
liverdata = pd.concat([liverdata,dummy],axis=1)
#action = dropped the categorical column 
liverdata = liverdata.drop('Gender',axis=1)
#action = importing libraries 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#action = fitted the df to scaler object and transformed to get a numpy array of values 
clean_df = scaler.fit_transform(liverdata)
#converting the array to dataframe 
liver_new = pd.DataFrame(clean_df)
#assigning columnnames for the new dataframe 
liver_new.columns = liverdata.columns
#findings = This is our normalized (within scale 0 ,1 by default) and cleaned dataset.  
liver_new.describe()
correlate = liver_new.corr()
#findings :- we can see that the total & direct billirubin , alamine & aspartate aminotransferase, albumin & proteins and the albumin globulin ratio are all 
#showing positive correlations.
import seaborn as sns
sns.heatmap(correlate, cmap = "YlGnBu",linewidths=.5,linecolor='cyan')
# Prepating the target and training class 
X = liver_new.drop('Dataset',axis=1).values
y = liver_new['Dataset'].values
#since we are having a sparse dataset with high bias the splitting of train and test data will affect the model adversly hence using stratified k fold.
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X,y)
# using xgboost which is a boosting classifier of ensemble. This is proven to give better accuracy.
import xgboost as xgb
# To defining the parameters for the xgboost classifier 
params = {
    'min_child_weight': 10.0,
    'objective': 'binary:logistic',
    'max_depth': 10,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round' : 700
    }
# ROC accuracy is a scoring method.
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
# This piece of code is from "HyungsukKang notebook - Stratified KFold+XGBoost+EDA Tutorial(0.281)" 
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, 5))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
     # Convert our data into XGBoost format
    d_train = xgb.DMatrix(X_train, y_train)
    d_test = xgb.DMatrix(X_test, y_test)
    watchlist = [(d_train, 'train')]
    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, maximize=True, verbose_eval=100)
    print('[Fold %d/%d Prediciton:]' % (i + 1, 5))
    # Predict on our test data
    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
    

roc_auc_score(y_test,p_test)
predictions = []
#changing the values of the continuous variable in p_test to binary values
for val in p_test:
    if val >= p_test.mean():
        predictions.append(1)
    else:
        predictions.append(0)
        
    
predictions[0:10]
print("values in the target and predicted variables :",len(predictions),len(y_test))
from sklearn.metrics import accuracy_score,f1_score
#roc_auc score & f1 score are almost similar,though roc is generally used for imbalanced datasets while f1_score can also be used to know our model performance.
f1_score(y_test,predictions)
