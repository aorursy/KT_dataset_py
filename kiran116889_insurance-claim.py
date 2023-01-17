import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import datasets

from sklearn import svm

import matplotlib as plt

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
#Read the csv file

mydata=pd.read_csv('C:/Users/genus/Documents/kaggle/train/train.csv')
#check the data

mydata.head()
mydata.describe()
#check for high correlations

mydata.corr()
mydata.columns
#check for missing values

mydata.isnull().sum()

# No missing  values , we can look at feature engineering

#split using train and test

from sklearn.model_selection import train_test_split

train, test = train_test_split(mydata, test_size = 0.2)

train_target=train['target']

test_target=test['target']

train=train[['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03',

       'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin',

       'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin',

       'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15',

       'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01',

       'ps_reg_02', 'ps_reg_03', 'ps_car_01_cat', 'ps_car_02_cat',

       'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',

       'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat',

       'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14',

       'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04',

       'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',

       'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14',

       'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin',

       'ps_calc_19_bin', 'ps_calc_20_bin']]



test=test[['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03',

       'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin',

       'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin',

       'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15',

       'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01',

       'ps_reg_02', 'ps_reg_03', 'ps_car_01_cat', 'ps_car_02_cat',

       'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',

       'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat',

       'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14',

       'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04',

       'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',

       'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14',

       'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin',

       'ps_calc_19_bin', 'ps_calc_20_bin']]

train_target.describe()


#Feature selection using Random forest



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0,class_weight='balanced')

clf=clf.fit(train,train_target)

newlist={"Feature":train.columns.values,"Importance":clf.feature_importances_}
newlist=pd.DataFrame(newlist)
newlist.sort_values(by='Importance',ascending=0)
#Feature selection using gradient descent

from sklearn.ensemble import GradientBoostingClassifier

clf =GradientBoostingClassifier(n_estimators=50)

clf=clf.fit(train,train_target)

newlist={"Feature":train.columns.values,"Importance":clf.feature_importances_}



newlist=pd.DataFrame(newlist)

newlist.sort_values(by='Importance',ascending=0)
# Feature Importance with Extra Trees Classifier

from pandas import read_csv

from sklearn.ensemble import ExtraTreesClassifier



model = ExtraTreesClassifier()

clf=clf.fit(train,train_target)

newlist={"Feature":train.columns.values,"Importance":clf.feature_importances_}

newlist=pd.DataFrame(newlist)

newlist.sort_values(by='Importance',ascending=0)
# From the above analysis we will select the following top 3 variables



train=train[['ps_car_13','ps_ind_03','ps_ind_05_cat']]



test=test[['ps_car_13','ps_ind_03','ps_ind_05_cat']]



#fit model using random model classifier



from sklearn.ensemble import RandomForestClassifier

rf_fit= RandomForestClassifier(n_estimators=10)

rf_fit = clf.fit(train,train_target)

pred=rf_fit.predict(train)
#check accuracy of training data

from sklearn import metrics

metrics.confusion_matrix(pred,train_target)
#check accuracy of test data

pred_test=rf_fit.predict(test)

from sklearn.metrics import roc_auc_score

roc_auc_score(pred_test,test_target)
pred_test=rf_fit.predict(test)
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import roc_auc_score

print(precision_score(pred_test,test_target))

print(recall_score(pred_test,test_target))
#Build model using kneighbours

from sklearn.neighbors import KNeighborsClassifier 

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(train,train_target) 
pred=neigh.predict(train)

#check accuracy of training data

from sklearn import metrics

metrics.confusion_matrix(pred,train_target)
#check accuracy on test data

pred_test=neigh.predict(test)

from sklearn.metrics import roc_auc_score

roc_auc_score(pred_test,test_target)

#Build the model using decision tree

from sklearn import tree

tree = tree.DecisionTreeClassifier()

tree = clf.fit(train,train_target)
#check accuracy of training data

from sklearn import metrics

#check accuracy on train data

pred_train=tree.predict(train)

metrics.confusion_matrix(pred_train,train_target)
pred_test=tree.predict(test)
metrics.confusion_matrix(pred_test,test_target)
roc_auc_score(pred_test,test_target)
# We select random forest as the best performing model which has the highest AUC and also is an ensemble classification



scoredata=pd.read_csv('C:/Users/genus/Documents/kaggle/test/test.csv')

#select the relevant variables

scoredata=scoredata[['ps_car_13','ps_ind_03','ps_ind_05_cat']]



#final scoring of the test data provided

scoredataresultsclass=rf_fit.predict(scoredata)
print(scoredataresults)
scoredataresults=rf_fit.predict_proba(scoredata)
scoredataresults[:,1]
finaldata={'probability_0':scoredataresults[:,0],'probability_1':scoredataresults[:,1],'Predicted_class':scoredataresultsclass}
finaldata
finaldata=pd.DataFrame(finaldata)
finaldata.describe()
finaldata.to_csv('C:/Users/genus/Documents/kaggle/score.csv')