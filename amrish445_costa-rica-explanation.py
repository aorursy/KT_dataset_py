import pandas as pd
import numpy as np 


import matplotlib.pyplot as plt

# Image manipulation
from skimage.io import imshow, imsave

# 1.2 Image compression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# 1.3 Libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# 1.4 ML - we will classify using lightgbm
#          with stratified cross validation
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

# 1.5 OS related
import os, sys, time

# 1.6 Bayes Optimization
#  Install as: pip install bayesian-optimization   
from bayes_opt import BayesianOptimization


# Disable scientific notation

#options(scipen = 999)

Costatrain = pd.read_csv("../input/train.csv")
Costatest = pd.read_csv("../input/test.csv")

print ("Train Dataset: Rows, Columns: ", Costatrain.shape)
print ("Test Dataset: Rows, Columns: ", Costatest.shape)
Costatrain.head()

Costatest.head()
# Get the examindata

def ExamineData(x):
    """Prints various data charteristics, given x
    """
    print("Data shape:", x.shape)
    print("\nColumn:", x.columns)
    print("\nData types", x.dtypes)
    print("\nDescribe data", x.describe())
    print("\nData ", x.head(2))
    print ("\nSize of data", sys.getsizeof(x)/1000000, "MB" )    # Get size of dataframes
    print("\nAre there any NULLS", np.sum(x.isnull()))
#Costatrain.index.values
#Costatrain.info()
ExamineData(Costatrain)

#To take  target column from test data
Train_target = Costatrain['Target']
Train_target.value_counts(normalize=True)
#Missing value
naData = Costatrain.isnull().sum().values / Costatrain.shape[0] *100
df_na = pd.DataFrame(naData, index=Costatrain.columns, columns=['Count'])
df_na = df_na.sort_values(by=['Count'], ascending=False)

missing_count = df_na[df_na['Count']>0].shape[0]

print(f'We got {missing_count} rows which have missing value in train set ')
df_na.head(10)

#ax=sns.barplot("df_na", "missing_count",data=df_na.head(10))
#_ = plt.title('Fraction of NaN values, %')
Costatrain_null = Costatrain.isnull().sum()
Costatrain_null_non_zero = Costatrain_null[Costatrain_null>0] / Costatrain.shape[0]

sns.barplot(x=Costatrain_null_non_zero, y=Costatrain_null_non_zero.index)
_ = plt.title('Fraction of NaN values in Train data, %')
#Costatest

#Missing value
naData = Costatest.isnull().sum().values / Costatest.shape[0] *100
df_na = pd.DataFrame(naData, index=Costatest.columns, columns=['Count'])
df_na = df_na.sort_values(by=['Count'], ascending=False)

missing_count = df_na[df_na['Count']>0].shape[0]

print(f'We got {missing_count} rows which have missing value in test set ')
df_na.head(10)

Costatest_null = Costatest.isnull().sum()
Costatest_null_non_zero = Costatest_null[Costatest_null>0] / Costatest.shape[0]

sns.barplot(x=Costatest_null_non_zero, y=Costatest_null_non_zero.index)
_ = plt.title('Fraction of NaN values in Test data, %')
# Impute missing values for v2a1, v18q1 for test and train data
Costatrain['v2a1'] = Costatrain['v2a1'].fillna(value=Costatrain['tipovivi3'])
Costatest['v2a1'] = Costatest['v2a1'].fillna(value=Costatest['tipovivi3'])

Costatrain['v18q1'] = Costatrain['v18q1'].fillna(value=Costatrain['v18q'])
Costatest['v18q1'] = Costatest['v18q1'].fillna(value=Costatest['v18q'])
Costatrain.info()
Costatest.info()
Costatrain.head(5)
Costatest.head(5)
#feature engineering for convert 0,1 to false and true respectively.
cols = ['edjefe', 'edjefa']
Costatrain[cols] = Costatrain[cols].replace({'no': 0, 'yes':1}).astype(float)
Costatest[cols] = Costatest[cols].replace({'no': 0, 'yes':1}).astype(float)

#Costatrain[cols] =Costatrain[cols].applymap(lambda x: 1 if x == True else x)
#Costatest[cols] =Costatest[cols].applymap(lambda x: 0 if x == 'no' else x)
#Costatrain[cols].replace({'no': 0, 'yes':1}).astype(float)
#Costatrain[cols]
#Costatest[cols]

#Added new feature for roof and electricity
cols=['techozinc','techoentrepiso','techocane','techootro']


Costatrain['roof_waste_material'] = np.nan
Costatest['roof_waste_material'] = np.nan
Costatrain['electricity_other'] = np.nan
Costatest['electricity_other'] = np.nan

def fill_roof_exception(x):
    if (x['techozinc'] == 0) and (x['techoentrepiso'] == 0) and (x['techocane'] == 0) and (x['techootro'] == 0):
        return 1
    else:
        return 0
    
def fill_no_electricity(x):
    if (x['public'] == 0) and (x['planpri'] == 0) and (x['noelec'] == 0) and (x['coopele'] == 0):
        return 1
    else:
        return 0

Costatrain['roof_waste_material'] = Costatrain.apply(lambda x : fill_roof_exception(x),axis=1)
Costatest['roof_waste_material'] = Costatest.apply(lambda x : fill_roof_exception(x),axis=1)
Costatrain['electricity_other'] = Costatrain.apply(lambda x : fill_no_electricity(x),axis=1)
Costatest['electricity_other'] = Costatest.apply(lambda x : fill_no_electricity(x),axis=1)
#Costatrain['roof_waste_material']
Costatrain.head()
Costatrain['adult'] = Costatrain['hogar_adul'] - Costatrain['hogar_mayor']
Costatrain['dependency_count'] = Costatrain['hogar_nin'] + Costatrain['hogar_mayor']
Costatrain['dependency'] = Costatrain['dependency_count'] / Costatrain['adult']
Costatrain['child_percent'] = Costatrain['hogar_nin']/Costatrain['hogar_total']
Costatrain['elder_percent'] = Costatrain['hogar_mayor']/Costatrain['hogar_total']
Costatrain['adult_percent'] = Costatrain['hogar_adul']/Costatrain['hogar_total']


Costatest['adult'] = Costatest['hogar_adul'] - Costatest['hogar_mayor']

Costatest['dependency_count'] = Costatest['hogar_nin'] + Costatest['hogar_mayor']
Costatest['dependency'] = Costatest['dependency_count'] / Costatest['adult']
Costatest['child_percent'] = Costatest['hogar_nin']/Costatest['hogar_total']
Costatest['elder_percent'] = Costatest['hogar_mayor']/Costatest['hogar_total']
Costatest['adult_percent'] = Costatest['hogar_adul']/Costatest['hogar_total']
############more feature engineer 
#drop column in train and test data ,which will no use.
submission = Costatest[['Id']]
Costatrain.drop(columns=['idhogar','Id', 'tamhog', 'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)
Costatest.drop(columns=['idhogar','Id', 'tamhog', 'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)

correlation = Costatrain.corr()
correlation = correlation['Target'].sort_values(ascending=False)
print(f'The most 20 positive feature: \n{correlation.head(20)}')
print('*'*50)

print(f'The most 20 negative feature: \n{correlation.tail(20)}')
#Costatrain.dtypes
# get the labels
y = Costatrain['Target']

#y
Costatrain.drop(['Target'], inplace=True, axis=1)
#x = Costatrain.values
#x
#
# Create training and validation sets
#
#x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# Create the LightGBM data containers
#
#categorical_features = [c for c, col in enumerate(Costatrain.columns) if 'cat' in col]
#train_data = lgb.Dataset(x, label=y)
#test_data = lgb.Dataset(x_test, label=y_test)

#
# Train the model
#
#parameter value is copied from 
clf = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.89, min_child_samples = 90, num_leaves = 14, subsample = 0.96)

kfold = 5
skf = StratifiedKFold(n_splits=kfold, shuffle=True)
X=Costatrain
folds = skf.split(X,y)    # Our data is in X and y
type(folds)    # generator
#kf.get_n_splits((Costatrain,  y)
predicts = []
for train_index, test_index in folds:
    print("train_test_index")
    X_train, X_val = Costatrain.iloc[train_index], Costatrain.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
            early_stopping_rounds=400, verbose=100)
    predicts.append(clf.predict(Costatest))



indices = np.argsort(clf.feature_importances_)[::-1]
indices = indices[:75]

# Visualise these with a barplot
plt.subplots(figsize=(20, 15))
g = sns.barplot(y=Costatrain.columns[indices], x = clf.feature_importances_[indices], orient='h')
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=26)
g.tick_params(labelsize=12)
g.set_title("LightGBM feature importance");
#submission = Costatest[['Id']]
#ExamineData(Costatrain);
#Costatest.dtypes;
#Costatrain.head()
#submission['Target']=np.array(predicts_result).mean(axis=0).round().astype(int)
#predicts_result
y_pred =np.array(predicts).mean(axis=0).round().astype(int)# np.mean(predicts_result,axis=1) 
#len(y_pred)
#submission
#submission.dtypes
sub = pd.DataFrame()
sub['Id'] = submission['Id']
sub['Target'] = y_pred
sub.to_csv('submission.csv', index=False)
sub.head()
#sub = pd.read_csv("sample_submission.csv", header =0)
#submission['Target'] = y_pred
#sub
#sub.to_csv("sub.csv",index = False)
#y_pred.dtypes
#submission['Target']=y_pred
#X=Costatest[predicts_result]
#submission['Target'] =predicts_result#xg_cl.predict(X)# np.array(predicts_result).mean(axis=0).round().astype(int)
#np.array(predicts_result).mean(axis=0).round().astype(int)
#submission.to_csv('submission.csv', index = False)
