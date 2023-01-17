# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling as pp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#data = pd.read_csv("../input/train.csv",dtype={'Year':'str','Qrtr':'str','Date':'str'})

data = pd.read_csv("../input/train.csv",dtype={'YearBuilt':'str','YrSold':'str','GarageYrBlt':'str','YearRemodAdd':'str'})

data.shape
data.info(verbose=True)
data.describe()
profile = pp.ProfileReport(data)

profile.to_file("HousingSales.html")

pp.ProfileReport(data)
missing_list = data.columns[data.isna().any()].tolist()

data.columns[data.isna().any()].tolist()
data.shape
data_org = data

data_org.shape

data.drop(['Alley','MasVnrArea','PoolQC','Fence','MiscFeature'], inplace=True, axis=1)

data.shape
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical = ['object']



for cols in list((data.select_dtypes(include=numerics)).columns.values):

    data[cols] =  data[cols].replace(np.nan,data[cols].median())

    

for cols in list((data.select_dtypes(include=categorical)).columns.values):

    data[cols] =  data[cols].replace(np.nan,"Not_Available")    
# Checking to see if all missing values have been taken care of 

data.columns[data.isna().any()].tolist()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



train_numeric = data.select_dtypes(include=numerics)

train_numeric.shape
import seaborn as sns

import matplotlib.pyplot as plt

for i in range(0,33):

    plt.figure(figsize=(45, 10))

    b = sns.boxplot(data = train_numeric.iloc[:,i:i+3])

    b.tick_params(labelsize= 30)

    i = i+3 
#varstobetreated = ['Age','avg_change_in_efficiency_6m','avg_spl_inc_amt_3M','avg_deviation_avg_training_peers_3M','avg_perc_deviation_incentive_amt_peers_3M','avg_peer_bktx_cases_ratio_3M','OD_EFFICIENCY_3M','CD_EFFICIENCY_3M','time_last_supervisor_change','incentive_amount_lag_1','incentive_amount_lag_2','incentive_amount_lag_3','incentive_amount_lag_4','incentive_amount_lag_5','incentive_amount_lag_6','peer_od_cases_ratio_lag_2','peer_od_cases_ratio_lag_3','peer_od_cases_ratio_lag_4','AVG_OB_CASES_3M','perc_resignation_under_supervisor_3M']

varstobetreated = list(train_numeric.columns)

for cols in varstobetreated:

    Q1 = data[cols].quantile(0.25)

    Q3 = data[cols].quantile(0.75)

    IQR = Q3 - Q1

    Upper_Limit = Q3 + 1.5*IQR

    Lower_Limit = Q1 - 1.5*IQR

    data[cols] = np.where(data[cols] > Upper_Limit,Upper_Limit,data[cols])

    data[cols] = np.where(data[cols] < Lower_Limit,Lower_Limit,data[cols])
# Filtering out variables

data.drop(['BsmtHalfBath', 'CentralAir', 'Condition2', 'Heating', 'RoofMatl', 'Street', 'Utilities', 'Heating', 'PoolArea', 'YearRemodAdd', 'YearBuilt', 'GarageYrBlt'],inplace=True, axis=1)

data.shape
colstokeep = ['Id','MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'LotShape',

       'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',

       'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle',

       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',

       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

       'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',

       'HeatingQC', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',

       'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional',

       'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars',

       'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal',

       'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']
def getFeatures(df , run_id, tuple_cols_toKeep, cat_ResponseVariable):

    #response = df[cat_ResponseVariable]  == 'Y'

    response = df[cat_ResponseVariable]

    if(df is None):

        df = get_data_from_SQL()

    #writeFrameToCSV(df,PROJECT2_HOME+run_id)\n",

    if(tuple_cols_toKeep is None):

        features = df

    else:

        df = df.drop(columns = cat_ResponseVariable)

        features = pd.DataFrame(df, columns = tuple_cols_toKeep)

        features = features.drop(columns = ['Id'])

    features = pd.get_dummies(features,drop_first=True)

    features[cat_ResponseVariable] = response

    features.head()

    return(features)
features = getFeatures(data,'1020',colstokeep,'SalePrice')

features_new = pd.DataFrame(data, columns = colstokeep) 

features_final = pd.concat([features,features_new['Id']], axis=1) 

features_copy = features_final

labelkey = features_final['Id']

labels = features_final['SalePrice']

feature_list = list(features_final.columns)

features_final = features_final.drop(columns = ['SalePrice','BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch', 'MiscVal', 'Electrical_Mix'])

features_final = features_final.drop(columns = ['Id'])

features_final.shape
import numpy as np

import pandas as pd

import time

from statsmodels.stats.outliers_influence import variance_inflation_factor    

from joblib import Parallel, delayed



# Defining the function that you will run later

def calculate_vif_(X, thresh=5.0):

    variables = [X.columns[i] for i in range(X.shape[1])]

    dropped=True

    while dropped:

        dropped=False

        #print(len(variables))

        vif = Parallel(n_jobs=-1,verbose=5)(delayed(variance_inflation_factor)(X[variables].values, ix) for ix in range(len(variables)))



        maxloc = vif.index(max(vif))

        if max(vif) > thresh:

            #print(time.ctime() + ' dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))

            variables.pop(maxloc)

            dropped=True



    #print('Remaining variables:')

    #print([variables])

    return X[[i for i in variables]]

X2 = calculate_vif_(features_final,5)

X2.shape



features_final = X2
# Libraries that sklearn provides:



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

import statsmodels.formula.api as sm
train_features, test_features, train_labels, test_labels, train_labelkey, test_labelkey = train_test_split(features_final, labels, labelkey, test_size = 0.15, random_state = 42)

print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Training Label Key Shape:', train_labelkey.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)

print('Testing Label Key Shape:', test_labelkey.shape)
rf = RandomForestRegressor(n_estimators = 50, random_state = 42 , max_depth= 5, min_samples_leaf=5)

rf.fit(train_features, train_labels)
predicted_vals_train = rf.predict(train_features)

predicted_vals_train
labelkey_train = np.array(train_labelkey)

labels_train = np.array(train_labels)

predicted_vals_train = rf.predict(train_features)

data_train = pd.concat([pd.DataFrame(labelkey_train), pd.DataFrame(labels_train), pd.DataFrame(predicted_vals_train)], axis=1) 

data_train.shape

data_train.columns = ['Id','Actual','Predicted']

data_train['pred_error'] = (data_train['Actual']-data_train['Predicted']).abs()

data_train['pred_error_percent'] = data_train['pred_error']/data_train['Actual']

print(data_train['pred_error_percent'].mean())
labelkey_test = np.array(test_labelkey)

labels_test = np.array(test_labels)

predicted_vals_test = rf.predict(test_features)

data_test = pd.concat([pd.DataFrame(labelkey_test), pd.DataFrame(labels_test), pd.DataFrame(predicted_vals_test)], axis=1) 

data_test.shape

data_test.columns = ['Id','Actual','Predicted']

data_test['pred_error'] = (data_test['Actual']-data_test['Predicted']).abs()

data_test['pred_error_percent'] = data_test['pred_error']/data_test['Actual']

print(data_test['pred_error_percent'].mean())
feat_importances = pd.Series(rf.feature_importances_, index= train_features.columns)

feat_importances.nlargest(25).plot(kind='bar')
importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]

a = []



print("Feature ranking:")

for f in range(train_features.shape[1]):

    print("feature  %d (%f)" % (indices[f], importances[indices[f]]))
SalePrice_CutOff = np.percentile(features_copy['SalePrice'],80)

features_copy['SalePrice_Classification'] =  np.where(features_copy['SalePrice'] >= SalePrice_CutOff,1,0)
features_copy['SalePrice_Classification'].value_counts()
labels_classification = features_copy['SalePrice_Classification']

labelkey_classification = features_copy['Id']

final_colstokeep = list(features_final.columns)

features_final_classification = pd.DataFrame(features_copy,columns=final_colstokeep)

#features_final_classification.reset_index(inplace = True)

features_final_classification.shape 
train_features_class, test_features_class, train_labels_class, test_labels_class,  train_labelkey_class, test_labelkey_class = train_test_split(features_final_classification, labels_classification, labelkey_classification,  test_size = 0.15, random_state = 42)

#train_features_class, test_features_class, train_labels_class, test_labels_class,  train_labelkey_class, test_labelkey_class = train_test_split(features_copy, labels_classification, labelkey_classification,  test_size = 0.15, random_state = 42)

print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Training Label Key Shape:', train_labelkey.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)

print('Testing Label Key Shape:', test_labelkey.shape)
xgb_class = XGBClassifier(gamma=0.01, learning_rate=0.1,max_depth=2,n_estimators=70,n_jobs=1,random_state=42)

xgb_class.fit(train_features_class, train_labels_class)
labelkey_class_train = np.array(train_labelkey_class)

labels_class_train = np.array(train_labels_class)

predicted_probs_train = xgb_class.predict_proba(train_features_class)

data_train = pd.concat([pd.DataFrame(labelkey_class_train),pd.DataFrame(labels_class_train), pd.DataFrame(predicted_probs_train,columns=['Col_0','Col_1'])], axis=1) 

data_train.shape

data_train['prob_decile'] = pd.qcut(data_train['Col_1'], 10,labels=False)

data_train.head()
labelkey_class_test = np.array(test_labelkey_class)

labels_class_test = np.array(test_labels_class)

predicted_probs_test = xgb_class.predict_proba(test_features_class)

data_test = pd.concat([pd.DataFrame(labelkey_class_test),pd.DataFrame(labels_class_test), pd.DataFrame(predicted_probs_test,columns=['Col_0','Col_1'])], axis=1) 

data_test.shape

data_test['prob_decile'] = pd.qcut(data_test['Col_1'], 10,labels=False)

data_test.head()
from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(labels_class_train, predicted_probs_train[::,1], pos_label=1)

import matplotlib.pyplot as plt

auc_train = metrics.roc_auc_score(labels_class_train, predicted_probs_train[::,1])

print(auc_train)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc_train))

plt.legend(loc=4)
fpr, tpr, thresholds = metrics.roc_curve(labels_class_test, predicted_probs_test[::,1], pos_label=1)

import matplotlib.pyplot as plt

auc_test = metrics.roc_auc_score(labels_class_test, predicted_probs_test[::,1])

print(auc_test)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc_test))

plt.legend(loc=4)
data_train.head()
data_train_copy=data_train

columns_names=['Id','TAG','Col_0','Col_1','prob_decile']

#columns_names=['Id','TAG','Col_0','Col_1','prob_decile','prediction']

data_train.columns = columns_names

data_train.tail()



Base_Considered = data_train.loc[(data_train['prob_decile'] >= 7)]

set_prob = Base_Considered['Col_1'].min()

print(set_prob)

data_train['prediction']=np.where(data_train['Col_1'] >= set_prob ,1,0)

data_train.head()



dt_pivot = pd.pivot_table(data_train, values= 'Id', index= 'prob_decile', columns= 'TAG', aggfunc= np.count_nonzero ,

               margins= True)

dt_pivot.index.name = None

dt_pivot.columns.name = None

dt_pivot.columns = ['Base', 'Responders', 'Total']



table = pd.pivot_table(data_train, values=['Id', 'Col_1'], index=['prob_decile'], aggfunc={'prob_decile': np.count_nonzero,'Col_1': [min, max]})

table['responders']=dt_pivot['Responders']

Base_Considered=table.sort_index(ascending=False)

Base_Considered['cumulative_responders'] = Base_Considered.responders.cumsum()

Base_Considered['responders_perc'] = 100*Base_Considered.cumulative_responders/Base_Considered.responders.sum()

Base_Considered



dt_pivot = pd.pivot_table(data_train, values= 'Id', index= 'TAG', columns= 'prediction', aggfunc= np.count_nonzero ,

               margins= True)

# print(dt_pivot)

confusion_matrix=dt_pivot

confusion_matrix.rename(index={0:'FALSE',1:'TRUE'}, columns={0:'FALSE',1:'TRUE'}, inplace=True)

confusion_matrix



print('Accuracy :')

print((confusion_matrix.iloc[0,0]+confusion_matrix.iloc[1,1])/(confusion_matrix.iloc[2,2]))

print('precision:')

print((confusion_matrix.iloc[1,1])/(confusion_matrix.iloc[2,1]))

print('Recall:')

print((confusion_matrix.iloc[1,1])/(confusion_matrix.iloc[1,2]))

print('F1 Score:')

a=2*((confusion_matrix.iloc[1,1])/(confusion_matrix.iloc[2,1])*(confusion_matrix.iloc[1,1])/(confusion_matrix.iloc[1,2]))

b=((confusion_matrix.iloc[1,1])/(confusion_matrix.iloc[2,1]))+((confusion_matrix.iloc[1,1])/(confusion_matrix.iloc[1,2]))

print(a/b)
feat_importances = pd.Series(xgb_class.feature_importances_, index= train_features_class.columns)

feat_importances.nlargest(25).plot(kind='bar')
importances = xgb_class.feature_importances_

indices = np.argsort(importances)[::-1]

a = []



print("Feature ranking:")

for f in range(train_features_class.shape[1]):

    print("feature  %d (%f)" % (indices[f], importances[indices[f]]))
#train_logit = pd.concat([train_labels_class,train_features_class], axis=1)

#train_cols = train_logit.columns[1:]



import statsmodels.api as sm

logit = sm.Logit(train_labels_class,train_features_class)

#logit = sm.Logit(train_logit['SalePrice_Classification'],train_logit[train_cols])



# fit the model

result = logit.fit(method = 'bfgs')
result.summary2()
predicted_probs_train = result.predict(train_features_class)

data_train = pd.concat([pd.DataFrame(train_labelkey_class),pd.DataFrame(train_labels_class), pd.DataFrame(predicted_probs_train)], axis=1)

data_train.shape

data_train.columns = ['Id','OrginalFlag','PredictedProbability']

data_train['prob_decile'] = pd.qcut(data_train['PredictedProbability'], 10,labels=False)

data_train.head()



predicted_probs_test = result.predict(test_features_class)

data_test = pd.concat([pd.DataFrame(test_labelkey_class),pd.DataFrame(test_labels_class), pd.DataFrame(predicted_probs_test)], axis=1) 

data_test.shape

data_test.columns = ['Id','OrginalFlag','PredictedProbability']

data_test['prob_decile'] = pd.qcut(data_test['PredictedProbability'], 10,labels=False)

data_test.head()
from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(train_labels_class, predicted_probs_train, pos_label=1)

import matplotlib.pyplot as plt

auc_train = metrics.roc_auc_score(train_labels_class, predicted_probs_train)

print(auc_train)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc_train))

plt.legend(loc=4)
from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(test_labels_class, predicted_probs_test, pos_label=1)

import matplotlib.pyplot as plt

auc_test = metrics.roc_auc_score(test_labels_class, predicted_probs_test)

print(auc_test)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc_test))

plt.legend(loc=4)