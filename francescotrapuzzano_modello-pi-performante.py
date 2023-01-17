import pip
import sys
import random
import numpy as np
import pandas as pd
import graphviz
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import array
import random
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
def feature_engineering(dataframe, print_subs=True):
  cols = ['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']
  min_vals = [4.5, 0.1, 0, 0.5, 0.1, 2, 30, 0.98, 2.7, 0.2, 8]
  max_vals = [10, 0.6, 0.9, 30, 100, 100, 290, 1.2, 3.9, 1, 14]

  #correction phase
  vol_ac_tr = 100
  chl_tr_1 = 0.1
  den_tr = 980

  if print_subs:
    print("valori corretti (max volatile.acidity):")
    print(len(np.where(dataframe[cols[1]] > vol_ac_tr)[0]))

    print("valori corretti (min chlorides):")
    print(len(np.where(dataframe[cols[4]] < chl_tr_1)[0]))

    print("valori corretti (max density):")
    print(len(np.where(dataframe[cols[7]] > den_tr)[0]))

    print("\n")

  dataframe[cols[1]] = np.where(dataframe[cols[1]] > vol_ac_tr, dataframe[cols[1]] / 1000, dataframe[cols[1]])
  dataframe[cols[4]] = np.where(dataframe[cols[4]] < chl_tr_1, dataframe[cols[4]] * 1000, dataframe[cols[4]])
  dataframe[cols[7]] = np.where(dataframe[cols[7]] > den_tr, dataframe[cols[7]] / 1000, dataframe[cols[7]])

  #remove other errors
  for i in range(len(cols)):
    if print_subs:
      print("valori rimossi (min ", cols[i], "):")
      print(len(np.where(dataframe[cols[i]] < min_vals[i])[0]))
    
    dataframe[cols[i]] = np.where(dataframe[cols[i]] < min_vals[i], np.nan, dataframe[cols[i]])

    if print_subs:
      print("valori rimossi (max ", cols[i], "):")
      print(len(np.where(dataframe[cols[i]] > max_vals[i])[0]))
    
    dataframe[cols[i]] = np.where(dataframe[cols[i]] > max_vals[i], np.nan, dataframe[cols[i]])

  return dataframe
train = pd.read_csv('../input/mldm-classification-competition-2020/train.csv')
train=feature_engineering(train)
train["Quality"] = np.where(train["Quality"].str.contains("Good"), 1, 0)

predictor_cols = ['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']
train_X = train[predictor_cols]
train_y = train.Quality

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp = imp.fit(train_X)
train_X_imp = imp.transform(train_X)

train_X_imp = preprocessing.scale(train_X_imp)

print(train_X_imp)
print(train_y)
mdl= ExtraTreesClassifier(random_state=42, criterion='gini')

print(mdl)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
max_features = ['auto','sqrt','log2']
max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]
min_samples_split = [1,3,5,7,9]
bootstrap = [True, False]

distributions= {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'bootstrap': bootstrap}

rand_s = RandomizedSearchCV(estimator=mdl,
                           param_distributions=distributions,
                           n_iter=100,
                           cv=3, 
                           n_jobs=-1,
                           random_state=42) 

rand_s.fit(train_X_imp, train_y)

print(rand_s.best_params_)
my_model=rand_s.best_estimator_

my_model.fit(train_X_imp, train_y)
my_model.score(train_X_imp, train_y)

y_pred = my_model.predict(train_X_imp)

print("Confusion Matrix:")
print(confusion_matrix(train_y, y_pred))

print("Classification Report:")
print(classification_report(train_y, y_pred))

test = pd.read_csv('../input/mldm-classification-competition-2020/test.csv')
test=feature_engineering(test)


test_X = test[predictor_cols]
test_X_imp = imp.transform(test_X)

test_X_imp = preprocessing.scale(test_X_imp)

predicted_q = my_model.predict(test_X_imp)

print(predicted_q)
#Kaggle public: 0.81088
#Kaggle private: 0.84539
ExtraTreesClassifier(bootstrap=False,
                     ccp_alpha=0.0,
                     class_weight=None,      
                     criterion='gini',
                     max_depth=60,
                     max_features='sqrt',      
                     max_leaf_nodes=None,
                     max_samples=None,    
                     min_impurity_decrease=0.0,
                     min_impurity_split=None,  
                     min_samples_leaf=1, 
                     min_samples_split=5,   
                     min_weight_fraction_leaf=0.0, 
                     n_estimators=600,        
                     n_jobs=None,
                     oob_score=False, 
                     random_state=None,
                     verbose=0,       
                     warm_start=False)