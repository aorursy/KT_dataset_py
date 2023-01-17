import os

print((os.listdir('../input/')))
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier, cv

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.utils import resample

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

test_index=df_test['Unnamed: 0'] #copying test index for later
df_train.head()
def fe(x):

    x['V17']=(x.V1+x.V2+x.V3+x.V4+x.V8)/x.V7

    x['V18']=(x.V14==-1)*1-(x.V16==3)*1/x.V2

    x['V19']=((x.V7==1)*1+(x.V11==8)*1)/x.V11

    return(x)

train_x = df_train.loc[0:29999,'V1':'V16']

train_x = fe(train_x)

train_y = df_train.loc[0:29999, 'Class']

#These specific features were introduced in the dataset as they were found to have one of the best correlations with the class labels.

print(train_x.head())
train_x[['V18','V19']].groupby(['V18'],as_index=False).mean().sort_values(by='V18',ascending=False)
corr = df_train.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    

    horizontalalignment='right'

);
corr = train_x.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    

    horizontalalignment='right'

    

);

#validation set was created to understand the basic starting points for all hyperparameters, so that randomsearch can be used better.

x_train,x_dev,y_train,y_dev= train_test_split(train_x,train_y,train_size= 0.8,test_size= 0.2, random_state= 2)

xgb= XGBClassifier(n_estimators=250,subsample=0.7,reg_lambda=0.5,min_child_weight=1.3,max_depth=5,learning_rate=0.065,gamma=1.0,colsample_bytree=0.7,colsample_bynode=0.6,colsample_bylevel=0.7,eval_metric='auc')

#the classifier is now classified with the best possible hyperparameters found from random grid search for the increased number of features.
dev_set=[(x_train,y_train),(x_dev,y_dev)]

xgb.fit(x_train,y_train, eval_metric=['auc',"logloss"],eval_set= dev_set, verbose=True, early_stopping_rounds=20)

#early stopping was used to estimate the number of trees required.
results= xgb.evals_result()

epochs = len(results['validation_0']['auc'])

x_axis = range(0, epochs)

# plot log loss

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

ax.plot(x_axis, results['validation_1']['logloss'], label='Dev')

ax.legend()

plt.ylabel('Log Loss')

plt.title('XGBoost Log Loss')

plt.show()



xgb.fit(train_x,train_y)
df_test = df_test.loc[:, 'V1':'V16']

df_test=fe(df_test) #Add require features to test set

pred = xgb.predict_proba(df_test)

  
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output.csv', index=False)