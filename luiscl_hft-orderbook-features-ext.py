import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from datetime import datetime

from scipy.special import logsumexp



from catboost import Pool, cv, CatBoostClassifier, CatBoostRegressor

from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score

os.environ["OMP_NUM_THREADS"] = "8"
train = pd.read_csv("/kaggle/input/caltech-cs155-2020/train.csv")

test = pd.read_csv("/kaggle/input/caltech-cs155-2020/test.csv")

df = pd.concat([train,test],sort=False)
from sklearn import preprocessing

eug_df = df.copy()

eug_df['opened_position_qty '] = np.log1p(eug_df['opened_position_qty '])

eug_df['closed_position_qty'] = np.log1p(eug_df['closed_position_qty'])

eug_df['transacted_qty'] = np.log1p(eug_df['transacted_qty'])

eug_df['bid1vol'] = np.log1p(eug_df['bid1vol'])

eug_df['bid2vol'] = np.log1p(eug_df['bid2vol'])

eug_df['bid3vol'] = np.log1p(eug_df['bid3vol'])

eug_df['bid4vol'] = np.log1p(eug_df['bid4vol'])

eug_df['bid5vol'] = np.log1p(eug_df['bid5vol'])

eug_df['ask1vol'] = np.log1p(eug_df['ask1vol'])

eug_df['ask2vol'] = np.log1p(eug_df['ask2vol'])

eug_df['ask3vol'] = np.log1p(eug_df['ask3vol'])

eug_df['ask4vol'] = np.log1p(eug_df['ask4vol'])

eug_df['ask5vol'] = np.log1p(eug_df['ask5vol'])
bid_cols = ['bid1','bid2', 'bid3', 'bid4', 'bid5']

bid_vol_cols = ['bid1vol', 'bid2vol', 'bid3vol', 'bid4vol', 'bid5vol']

ask_cols = ['ask1', 'ask2', 'ask3', 'ask4', 'ask5',]

ask_vol_cols = ['ask1vol','ask2vol', 'ask3vol', 'ask4vol', 'ask5vol']



group_cols = {"bid_cols":bid_cols,"bid_vol_cols":bid_vol_cols,"ask_cols":ask_cols,"ask_vol_cols":ask_vol_cols}
#watch out here

df = eug_df

for group in group_cols.keys():

    print(group)

    df[f"{group}_max"] = df[group_cols[group]].max(axis=1)

    df[f"{group}_min"] = df[group_cols[group]].min(axis=1)

    df[f"{group}_spread"] = df[f"{group}_max"].div(df[f"{group}_min"])

    

df["last_price_div__mid"] = df["last_price"].div(df["mid"])

#Add weighted mid-price for each price level

#https://wwwf.imperial.ac.uk/~ajacquie/Gatheral60/Slides/Gatheral60%20-%20Stoikov.pdf

for i in range(1,6):

    I = df['bid'+str(i)+'vol'].div(df['bid'+str(i)+'vol'].add(df['ask'+str(i)+'vol']))

    print(type(I))

    df["weighted_mid"+str(i)] = (df['ask'+str(i)]*I).add((1-I)*(df['bid'+str(i)]))
train = df.loc[~df.y.isna()]

print(f"train shape {train.shape[0]}")

test = df.loc[df.y.isna()]

print(f"test shape {test.shape[0]}")
test.head()
X_trn = train.drop(["id","y"],axis=1)

Y_trn = train["y"]

X_tst = test.drop(['id','y'],axis=1)
def auc(model, data, labels): #function to calculate AUC score

    return (metrics.roc_auc_score(labels,model.predict_proba(data)[:,1]))
from sklearn import metrics

from sklearn.model_selection import GridSearchCV



# Parameter Tuning

model = xgboost.XGBClassifier()

param_dist = {"max_depth": [5,6],

              "gamma":[5],

              "reg_lambda":[0.5],

              "subsample" : [0.8],

              "colsample_bytree":[0.4],

              "n_estimators": [400,500],

              "learning_rate": [0.005,0.001, 0.01]}

grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, 

                                   verbose=10, n_jobs=-1)

# param_dist = {"max_depth": [1,2],

#               "gamma":[0,1,5],

#               "subsample" : [0.8,0.9],

#               "colsample_bytree":[0.4,0.7],

#               "n_estimators": [40,60,100,140],

#               "learning_rate": [0.0001, 0.001, 0.01]}
# X_trn,Y_trn = shuffle(X_trn,Y_trn)

# grid_search.fit(X_trn,Y_trn)
grid_search.best_score_
model = xgboost.XGBClassifier()
model
model.set_params(max_depth=6,reg_lambda=0.3,gamma=5,n_estimators=230,subsample=0.8,

                learning_rate=0.0065,colsample_bytree=0.7)
from sklearn.utils import shuffle

train_x, train_y = X_trn[:473000],Y_trn[:473000]

test_x, test_y = X_trn[473000:],Y_trn[473000:]

train_x, train_y = shuffle(train_x, train_y)

test_x, test_y = shuffle(test_x, test_y)
#(LR,ESt): (0.01,140) --> (0.08, 180) --> (0.08, 180)
# model.set_params(max_depth=6,reg_lambda=0.3,gamma=5,n_estimators=240,subsample=0.8,

#                 learning_rate=0.006,colsample_bytree=0.7)

# 0.6655715231113145 and 0.62905

#model.set_params(max_depth=6,reg_lambda=0.4,gamma=5,n_estimators=240,subsample=0.8,

#                learning_rate=0.005,colsample_bytree=0.7)

#0.62841 



# model.set_params(max_depth=5,reg_lambda=0.4,gamma=5,n_estimators=240,subsample=0.8,

#                 learning_rate=0.008,colsample_bytree=0.7)

#0.6589190615078132

# model.set_params(max_depth=6,reg_lambda=0.35,gamma=5,n_estimators=240,subsample=0.8,

#                 learning_rate=0.008,colsample_bytree=0.7)

#0.660641222049973

# model.set_params(max_depth=6,reg_lambda=0.35,gamma=5,n_estimators=230,subsample=0.9,

#                 learning_rate=0.008,colsample_bytree=0.7)

#0.6604407247044543

# model.set_params(max_depth=6,reg_lambda=0.35,gamma=5,n_estimators=300,subsample=0.9,

#                 learning_rate=0.004,colsample_bytree=0.7)

#0.659337715341376 and 0.62825

# model.set_params(max_depth=6,reg_lambda=0.25,gamma=5,n_estimators=350,subsample=0.9,

#                 learning_rate=0.004,colsample_bytree=0.7)

# 0.6602872467808225

#adding log1p to base:

#worsens ein tho

model.fit(train_x,train_y)
roc_auc_score(test_y,model.predict_proba(test_x)[:,1]) 
X_trn,Y_trn = shuffle(X_trn,Y_trn)

model.fit(X_trn,Y_trn)
preds = model.predict_proba(X_tst)[:,1]

dataframe = pd.DataFrame({'id':test['id'],'Predicted':preds}) #preds to dataframe

dataframe.to_csv('last.csv',index=False) #dataframe to CSV file
feature_importances = model.get_feature_importance(train_pool)

feature_names = X.columns

for score, name in sorted(zip(feature_importances, feature_names), reverse=True):

    if score > 0.2:

        print('{0}: {1:.2f}'.format(name, score))
import shap

shap.initjs()



explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(train_pool)



# visualize the training set predictions

# SHAP plots for all the data is very slow, so we'll only do it for a sample. Taking the head instead of a random sample is dangerous! 

shap.force_plot(explainer.expected_value,shap_values[0,:300], X.iloc[0,:300])
# summarize the effects of all the features

shap.summary_plot(shap_values, X)
## todo : PDP features +- from shap
test["Predicted"] = model.predict(test.drop(["id","date","y"],axis=1),prediction_type='Probability')[:,1]

test[["id","Predicted"]].to_csv("submission.csv",index=False)