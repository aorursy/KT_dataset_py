# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import cross_val_score,StratifiedKFold





from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from rgf.sklearn import RGFClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
trainData = pd.read_csv('../input/train.csv',na_values=[-1,-1.0])

testData = pd.read_csv('../input/test.csv',na_values=[-1,-1.0])
trainData['dset'] = 'train'

testData['dset'] = 'test'

combined = pd.concat([trainData, testData], ignore_index=True)
  # ps_ind_01 

    # mergering but had good roc without merging

#combined['ps_ind_01_merged_cat'] = 1

#combined.loc[combined.ps_ind_01 < 2 ,'ps_ind_01_merged_cat'] = 0

#combined.loc[combined.ps_ind_01 > 2 ,'ps_ind_01_merged_cat'] = 2



  # ps_ind_02_cat 

    # only NA imputation

combined['ps_ind_02_cat'].fillna(4,inplace=True)



  # ps_ind_03 

    # not improved with binning. possible entropy based binning couldnt find a good package

#combined['ps_ind_03_merged_cat'] = 1

#combined.loc[combined.ps_ind_01 < 2 ,'ps_ind_03_merged_cat'] = 0

#combined.loc[combined.ps_ind_01 > 4 ,'ps_ind_03_merged_cat'] = 2



  #ps_ind_04_cat

combined['ps_ind_04_cat'].fillna(1,inplace=True)



  #ps_ind_05_cat : category 5 has lesser count, may be merged to two cat

combined['ps_ind_05_cat'].fillna(2,inplace=True)



  #ps_ind_06/06/08/09/10/11/12/13/16/17/18_bin 

    # 10,11,12,13 has very less count on one category

    

  #ps_ind_14 

    # binning possible

combined['ps_ind_14_binned'] = combined['ps_ind_14']

combined.loc[combined.ps_ind_14 > 0 ,'ps_ind_14_binned'] = 1



  #ps_ind_15 

    # better roc without any modification

#combined['ps_ind_15_binned'] = combined['ps_ind_15']

#combined.loc[combined.ps_ind_15 < 3 ,'ps_ind_15_binned'] = 0

#combined.loc[combined.ps_ind_15 > 8 ,'ps_ind_15_binned'] = 2

#combined.loc[(combined.ps_ind_15 > 2) & (combined.ps_ind_15 < 9) ,'ps_ind_15_binned'] = 1



  #ps_reg_01

combined['ps_reg_01'] = combined['ps_reg_01']*10



  #ps_reg_02

combined['ps_reg_02'] = combined['ps_reg_02']*10



  #ps_reg_01_02

combined['ps_reg_01_02'] = combined['ps_reg_01']*combined['ps_reg_02']    

  #ps_reg_03

    # NA 8%

    # outlier removing

combined['ps_reg_03'].fillna(combined['ps_reg_03'].median(skipna=True),inplace=True)

combined['ps_reg_03_mod'] = combined['ps_reg_03']

combined.loc[combined.ps_reg_03_mod < 0.25 ,'ps_reg_03_mod'] = 0.25

combined.loc[combined.ps_reg_03_mod > 2.25 ,'ps_reg_03_mod'] = 2.25

combined['ps_reg_03_mod'] = np.log(combined['ps_reg_03_mod']*100)



  #ps_car_01_cat

    # NA merged cat 9

combined['ps_car_01_cat'].fillna(9,inplace=True)

combined.loc[combined.ps_car_01_cat < 4 ,'ps_car_01_cat'] = 3



  #ps_car_02_cat

    # NA merged cat 9

combined['ps_car_02_cat'].fillna(1,inplace=True)



  #ps_car_03_cat

    # 80% NA



  #ps_car_04_cat

    # 3,4,5,6,7 has low counts

combined.loc[(combined.ps_car_04_cat < 8) & (combined.ps_car_04_cat >2) ,'ps_car_04_cat'] = 3    



  #ps_car_05_cat

    # 50% NA    

combined['ps_car_05_cat'].fillna(2,inplace=True)



  #ps_car_06_cat

    # 17 categories

    # some are having very low count

combined.loc[combined.ps_car_06_cat == 5,'ps_car_06_cat'] = 2

combined.loc[combined.ps_car_06_cat == 8,'ps_car_06_cat'] = 2

combined.loc[combined.ps_car_06_cat == 12,'ps_car_06_cat'] = 2

combined.loc[combined.ps_car_06_cat == 13,'ps_car_06_cat'] = 2

combined.loc[combined.ps_car_06_cat == 16,'ps_car_06_cat'] = 2

combined.loc[combined.ps_car_06_cat == 17,'ps_car_06_cat'] = 2



  #ps_car_07_cat

    # 2% NA

combined['ps_car_07_cat'].fillna(0,inplace=True)

combined['ps_car_07_bin'] = combined['ps_car_07_cat'] # renaming to bin



  #ps_car_08_cat

combined['ps_car_08_bin'] = combined['ps_car_08_cat'] # renaming to bin



  #ps_car_09_cat

    # NA 569/877

    # one category having low count

combined['ps_car_09_cat'].fillna(1,inplace=True)

combined.loc[combined.ps_car_09_cat == 4,'ps_car_09_cat'] = 1



  #ps_car_10_cat

    # cat 2 has very low count

combined.loc[combined.ps_car_10_cat==2,'ps_car_10_cat'] = 1

combined['ps_car_10_bin'] = combined['ps_car_10_cat']



  #ps_car_11_cat

    # too many categories



  #ps_car_11

    # better to try with category option

combined['ps_car_11'].fillna(1,inplace=True)



  #ps_car_12

    # NA changed with mean

combined['ps_car_12'].fillna(combined['ps_car_12'].median(skipna=True),inplace=True)

combined.loc[combined.ps_car_12>0.75,'ps_car_12'] = 0.75

combined.loc[combined.ps_car_12<0.2828,'ps_car_12'] = 0.2828



  #ps_car_13

combined.loc[combined.ps_car_13>2,'ps_car_13'] = 2



  #ps_car_14

combined['ps_car_14'].fillna(combined['ps_car_14'].median(skipna=True),inplace=True)

combined.loc[combined.ps_car_14<0.275,'ps_car_14'] <- 0.275

combined.loc[combined.ps_car_14>0.575,'ps_car_14'] <- 0.575



  #ps_car_15

combined['ps_car_15'] = round(combined['ps_car_15']**2,0)
trainFeatures = [

    "ps_ind_01",

    "ps_ind_02_cat", 

    "ps_ind_03",

    "ps_ind_04_cat",     

    "ps_ind_05_cat",      # cat 5 has very low count

    "ps_ind_06_bin", 

    "ps_ind_07_bin", 

    "ps_ind_08_bin", 

    "ps_ind_09_bin",    

    #"ps_ind_10_bin",     # non-zero variance 

    #"ps_ind_11_bin",     # non-zero variance 

    #"ps_ind_12_bin",     # non-zero variance 

    #"ps_ind_13_bin",     # non-zero variance

    "ps_ind_14_binned",  # one category dominates the count, # from the feature importance

    "ps_ind_15",

    "ps_ind_16_bin", 

    "ps_ind_17_bin", 

    "ps_ind_18_bin",      

    

    "ps_reg_01",

    "ps_reg_02",

    "ps_reg_01_02",

    #"ps_reg_03",

    "ps_reg_03_mod",

    

    "ps_car_01_cat", 

    "ps_car_02_cat",     

    #"ps_car_03_cat",     # 80% NA 

    "ps_car_04_cat", 

    "ps_car_05_cat",

    "ps_car_06_cat",      # saw slight drop in auc. but feature importance is good

    "ps_car_07_bin",      # rename as bin variable from cat due to two category

    "ps_car_08_bin",     # rename as bin variable from cat due to two category # from the feature importance

    "ps_car_09_cat", 

    "ps_car_10_bin",     # merge and rename as bin but accuracy dropped little # from the feature importance

    #"ps_car_11_cat",     # too many categories when introduce accuracy went down

    "ps_car_11",         

    "ps_car_12", 

    "ps_car_13",

    "ps_car_14",

    "ps_car_15",

    "ps_calc_01", 

    "ps_calc_02",

    "ps_calc_03", 

    "ps_calc_04",

    "ps_calc_05",        

    "ps_calc_06",

    "ps_calc_07", 

    "ps_calc_08",

    "ps_calc_09", 

    "ps_calc_10",

    "ps_calc_11", 

    "ps_calc_12",

    "ps_calc_13",        

    "ps_calc_14",

    "ps_calc_15_bin",    # from feature importance after introducing calc

    "ps_calc_16_bin",    # from feature importance after introducing calc

    "ps_calc_17_bin",    # from feature importance after introducing calc

    "ps_calc_18_bin",    # from feature importance after introducing calc

    "ps_calc_19_bin",    # from feature importance after introducing calc

    "ps_calc_20_bin",    # from feature importance after introducing calc

]
id_test = combined.loc[combined.dset=='test','id'].values

target_train = combined.loc[combined.dset=='train','target'].values



trainSet = combined.loc[combined.dset=='train',trainFeatures]

testSet = combined.loc[combined.dset=='test',trainFeatures]



cat_features = [a for a in trainSet.columns if a.endswith('cat')]

for column in cat_features:

    trainSet[column]=trainSet[column].astype('category')

    testSet[column]=testSet[column].astype('category')



temp = pd.get_dummies(trainSet[cat_features])

trainSet = pd.concat([trainSet,temp],axis=1)

trainSet.drop(np.asarray(cat_features),axis=1,inplace=True)



temp = pd.get_dummies(testSet[cat_features])

testSet = pd.concat([testSet,temp],axis=1)

testSet.drop(np.asarray(cat_features),axis=1,inplace=True)
# parameters

lgb_params = {}

lgb_params['n_estimators'] = 500      # n_estimators (int, optional (default=10)) – Number of boosted trees to fit.

lgb_params['learning_rate'] = 0.02    # learning_rate (float, optional (default=0.1)) – Boosting learning rate.

lgb_params['colsample_bytree'] = 0.8  # colsample_bytree (float, optional (default=1.)) – Subsample ratio of columns when constructing each tree.

lgb_params['subsample'] = 0.8         # subsample (float, optional (default=1.)) – Subsample ratio of the training instanc

lgb_params['subsample_freq'] = 2     # subsample_freq (int, optional (default=1)) – Frequence of subsample, <=0 means no enable.

lgb_params['max_bin'] = 32            # max_bin (int, optional (default=255)) – Number of bucketed bin for feature values.

lgb_params['min_child_samples'] = 20  # min_child_samples (int, optional (default=20)) – Minimum number of data need in a child(leaf).

lgb_params['random_state'] = 100

lgb_params['n_jobs'] = 2



# Model building

lgb_model = LGBMClassifier(**lgb_params)

cv_results = cross_val_score(lgb_model, trainSet, target_train, cv=StratifiedKFold(2), scoring='roc_auc',verbose=1)

print(cv_results)

# parameters

xgb_params = {}

xgb_params['objective'] = 'binary:logistic'

xgb_params['learning_rate'] = 1

xgb_params['n_estimators'] = 200

xgb_params['max_depth'] = 4

xgb_params['subsample'] = 0.9

xgb_params['colsample_bytree'] = 0.9

xgb_params['min_child_weight'] = 10

xgb_params['scale_pos_weight'] = 0.5

xgb_params['n_jobs'] = 2

xgb_params['gamma']=1

xgb_params['reg_alpha']=0

xgb_params['reg_lambda']=1

# Model building

xgb_model = XGBClassifier(**xgb_params)

cv_results = cross_val_score(xgb_model, trainSet, target_train, cv=2, scoring='roc_auc',verbose=1)

print(cv_results)
#CatBoost params initial

cat_params = {}

cat_params['iterations'] = 100

cat_params['depth'] = 8

cat_params['rsm'] = 0.95

cat_params['learning_rate'] = 0.03

cat_params['l2_leaf_reg'] = 3.5  

cat_params['border_count'] = 8

cat_params['gradient_iterations'] = 4

cat_params['n_jobs'] = 2



# Model building

catBoost_model = CatBoostClassifier(**cat_params)

cv_results = cross_val_score(catBoost_model, trainSet, target_train, cv=2, scoring='roc_auc',verbose=1)

print(cv_results)
rf_params ={}

rf_params['n_estimators'] = 1000 # The number of trees in the forest.

rf_params['min_samples_split'] = 50  #The minimum number of samples required to split an internal node:

rf_params['n_jobs'] = 2



# Model building

rf_model = RandomForestClassifier(**rf_params)

cv_results = cross_val_score(rf_model, trainSet, target_train, cv=2, scoring='roc_auc',verbose=1)

print(cv_results)
rgf_params ={}

rgf_params['max_leaf'] = 1000 

rgf_params['algorithm'] = "RGF_Sib"

rgf_params['test_interval'] = 100

rgf_params['n_jobs'] = 2



# Model building

rgf_model = RGFClassifier(**rgf_params)

cv_results = cross_val_score(rgf_model, trainSet, target_train, cv=2, scoring='roc_auc',verbose=1)

print(cv_results)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainSet, target_train, test_size = 0.2, random_state = 0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 92, kernel_initializer = 'uniform', activation = 'relu', input_dim = 92))



# Adding the second hidden layer

classifier.add(Dense(units = 92, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the third hidden layer

classifier.add(Dense(units = 92, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 32, epochs = 10)



# Part 3 - Making predictions and evaluating the model



# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)