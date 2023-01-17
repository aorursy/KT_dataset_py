import numpy as np 

import pandas as pd

import seaborn as sns

from datetime import datetime

master=pd.read_csv("../input/meteorological-model-versus-real-data/vigo_model_vs_real.csv",parse_dates=True)
#change label dependent variables

threshold=5000



Y=pd.DataFrame({"datetime":master.index,

                     "visibility_o":[1 if c<=threshold else 0 for c in 

                                     master["visibility_o"]]}).set_index("datetime")

#choosing independent variables

X=master[['dir_4K', 'lhflx_4K', 'mod_4K', 'prec_4K', 'rh_4K', 'visibility_4K',

        'mslp_4K', 'temp_4K', 'cape_4K', 'cfl_4K', 'cfm_4K', 'cin_4K',"wind_gust_4K",

       'conv_prec_4K']]
#decission tree

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

import graphviz

clf = DecisionTreeClassifier(max_depth=3,criterion="gini").fit(X,Y) 

dot_data = tree.export_graphviz(clf, out_file=None, 

                     feature_names=['dir_4K', 'lhflx_4K', 'mod_4K', 'prec_4K', 'rh_4K', 'visibility_4K',

        'mslp_4K', 'temp_4K', 'cape_4K', 'cfl_4K', 'cfm_4K', 'cin_4K',"wind_gust_4K",

       'conv_prec_4K'],  

                     class_names=["no_fog","fog"],  

                     filled=True, rounded=True,  

                     special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix ,classification_report 

from sklearn.model_selection import cross_val_score,cross_validate

#we do not scale!!

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)

clf = DecisionTreeClassifier(max_depth=10,criterion="gini").fit(x_train,y_train) 

y_pred=clf.predict(x_test)

#plot results

print(classification_report(y_test.values,y_pred))

print("**** Confusion matrix ****")

print(confusion_matrix(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix ,classification_report 

from sklearn.model_selection import cross_val_score,cross_validate

#we do not scale!!

x_train, x_test, y_train, y_test = train_test_split(X.values,Y.values, test_size=0.2,)

clf =RandomForestClassifier (n_estimators=1500,bootstrap=False).fit(x_train,y_train.ravel()) 

y_pred=clf.predict(x_test)

#plot results

print(classification_report(y_test.ravel(),y_pred))

print("**** Confusion matrix ****")

print(confusion_matrix(y_test,y_pred))
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix ,classification_report 

from sklearn.model_selection import cross_val_score,cross_validate

#we do not scale!!

x_train, x_test, y_train, y_test = train_test_split(X.values,Y.values, test_size=0.2,)

clf =AdaBoostClassifier (n_estimators=1500).fit(x_train,y_train.ravel()) 

y_pred=clf.predict(x_test)

#plot results

print(classification_report(y_test.ravel(),y_pred))

print("**** Confusion matrix ****")

print(confusion_matrix(y_test,y_pred))
import pickle

pickle.dump(clf, open("vis_5000_Randomforest", 'wb'))

#LightGBM

##change label dependent variables

threshold=5000

X=master[["dir_4K", "lhflx_4K", "mod_4K", "prec_4K", "rh_4K", "visibility_4K",

 "mslp_4K", "temp_4K", "cape_4K", "cfl_4K", "cfm_4K", "cin_4K","conv_prec_4K"]]

y=pd.DataFrame({"datetime":master.index, "visibility_o":[1 if c<=threshold else 0

 for c in master["visibility_o"]]}).set_index("datetime")

#split

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



#tuning

import lightgbm as lgb

d_train = lgb.Dataset(x_train, label=y_train)

params = {}

params["learning_rate"] = 0.001

params["boosting_type"] = "gbdt"

params["objective"] = "binary"

params["metric"] = "binary_logloss"

params["sub_feature"] = 0.5

params["num_leaves"] = 1000

params["min_data"] = 2000

params["max_depth"] = 40

clf = lgb.train(params, d_train, 10000)
#Prediction dealing with skewed data

y_pred=clf.predict(x_test)**(1/3)

result=pd.DataFrame({"y_pred":y_pred, "y_test":y_test.values.reshape(1,-1)[0]})

g=pd.DataFrame({"y_pred test==1":result["y_pred"][result.y_test==1],

 "y_pred test==0":result["y_pred"][result.y_test==0]}).plot(kind="box",figsize=(15,15),grid=True)
from sklearn.metrics import confusion_matrix ,classification_report 

#select threhold_nor

threshold_nor=0.55

y_pred_nor=[0 if c<=threshold_nor else 1 for c in result.y_pred]

target_names = [">"+str(threshold)+"m","<="+str(threshold)+"m" ]

print(classification_report(y_test.values,y_pred_nor , target_names=target_names))

print("**** Confusion matrix ****")

print(confusion_matrix(y_test,y_pred_nor))
import pickle

pickle.dump(clf, open("vis_5000_lightgbm", 'wb'))
