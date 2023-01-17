import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split

data = pd.read_csv("../input/flights.csv")
data = data.sample(frac = 0.01, random_state=76492)

data = data[["MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT",
                 "ORIGIN_AIRPORT","AIR_TIME", "DEPARTURE_TIME","DISTANCE","ARRIVAL_DELAY"]]
data.dropna(inplace=True)

data["ARRIVAL_DELAY"] = (data["ARRIVAL_DELAY"]>10)*1

cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
for item in cols:
    data[item] = data[item].astype("category").cat.codes +1
 
train, test, y_train, y_test = train_test_split(data.drop(["ARRIVAL_DELAY"], axis=1), data["ARRIVAL_DELAY"],
                                                random_state=10, test_size=0.4)
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit(train["AIRLINE"].values.reshape(-1, 1))  

enc.feature_indices_
airline_onehot = enc.transform(train["AIRLINE"].values.reshape(-1, 1)).toarray()
airline_onehot_test = enc.transform(test["AIRLINE"].values.reshape(-1, 1)).toarray()
#airline_onehot_test.shape
air_oh_df = pd.DataFrame(airline_onehot, columns = ["A1","A2","A3","A4","A5","A6","A7",
                                                    "A8","A9","A10","A11","A12","A13","A14"])
air_oh_test_df = pd.DataFrame(airline_onehot_test, columns = ["A1","A2","A3","A4","A5","A6","A7",
                                                         "A8","A9","A10","A11","A12","A13","A14"])

train2 = pd.concat([train.reset_index(),air_oh_df.reset_index()],axis=1).drop(["index","MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"],axis=1)
test2  = pd.concat([test.reset_index(),air_oh_test_df.reset_index()],axis=1).drop(["index","MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"],axis=1)

test2
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

def auc(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
            metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))

# Parameter Tuning
#model = xgb.XGBClassifier()
param_dist = {"max_depth": [10,30,50],
              "min_child_weight" : [1,3,6],
              "n_estimators": [200],
              "learning_rate": [0.05,0.1,0.16]}
#grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, 
#                                   verbose=10, n_jobs=-1)
#grid_search.fit(train, y_train)

#grid_search.best_estimator_

start_time = time.time()

model = xgb.XGBClassifier(max_depth=50, min_child_weight=1,  n_estimators=200,
                          n_jobs=-1 , verbose=1, learning_rate=0.16)
model.fit(train2, y_train)

print(auc(model, train2, test2))

finish_time = time.time()
total_time = finish_time - start_time

print('Elapsed time: ', total_time)
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

def auc2(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict(train)),
                            metrics.roc_auc_score(y_test,m.predict(test)))

lg = lgb.LGBMClassifier(silent=False)
#param_dist = {"max_depth": [25,50, 75],
#              "learning_rate" : [0.01,0.05,0.1],
#              "num_leaves": [300,900,1200],
#              "n_estimators": [200]
#             }
#grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
#grid_search.fit(train,y_train)
#grid_search.best_estimator_

d_train = lgb.Dataset(train, label=y_train, free_raw_data=False)
params = {"max_depth": 50, "learning_rate" : 0.1, "num_leaves": 900,  "n_estimators": 300, 
          "boosting_type": 'goss', "max_conflict_rate": 0.3}

# Sem categoricas
model2 = lgb.train(params, d_train)
auc2(model2, train, test)

# Com categoricas
cate_features_name = ["MONTH","DAY","DAY_OF_WEEK","AIRLINE","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]

model2 = lgb.train(params, d_train, categorical_feature = cate_features_name)
auc2(model2, train, test)
import catboost as cb
cat_features_index = [0,1,2,3,4,5,6]

def auc(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
                            metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))

#params = {'depth': [4, 7, 10],
#          'learning_rate' : [0.03, 0.1, 0.15],
#         'l2_leaf_reg': [1,4,9],
#         'iterations': [300]}
#cb = cb.CatBoostClassifier()
#cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv = 3)
#cb_model.fit(train, y_train)

# Sem categoricas
clf = cb.CatBoostClassifier(eval_metric="AUC", depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
clf.fit(train,y_train)
auc(clf, train, test)

# Com categoricas
clf = cb.CatBoostClassifier(eval_metric="AUC",one_hot_max_size=31, \
                            depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
clf.fit(train,y_train, cat_features= cat_features_index)
auc(clf, train, test)
