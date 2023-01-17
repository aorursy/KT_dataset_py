# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import pandas_profiling as pp

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train= pd.read_csv('../input/learn-together/train.csv')

test= pd.read_csv('../input/learn-together/test.csv')
#pp.ProfileReport(train)
train.info()
test.info()
from scipy.stats import ks_2samp

from tqdm import tqdm

ks_values =[]

p_values  = []

train_columns = train.columns[1:55]

for i in tqdm(train_columns):

    ks_values.append(ks_2samp(test[i] , train[i])[0])

    p_values.append(ks_2samp(test[i] , train[i])[1])

p_values_series = pd.Series(p_values, index = train_columns) 
colpercount = pd.DataFrame()

for col in train.columns[1:54]:    

    unique_values =train[col].nunique()

    colpercount = colpercount.append({'feature' : col,'unique_value_num' : unique_values},ignore_index= True)

sns.set(rc={'figure.figsize':(16,8)})

plot1=colpercount.plot(x='feature',y='unique_value_num',kind='bar')

for p in plot1.patches[1:]:

    h = p.get_height()

    x = p.get_x()+p.get_width()/2.

    if h != 0:

        plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 

                   textcoords="offset points", ha="center", va="bottom")

plot1.set(ylabel='Count')

plot1= plot1.set(title='Number of unique values in each columns')
sns.set(rc={'figure.figsize':(16,30)})

x=1

for num, alpha in enumerate(train.columns[1:11]):

    plt.subplot(6, 4, num+x)

    #plt.hist(diffcheck(alpha)[0], alpha=0.75, label='train', color='g')

    #plt.hist(diffcheck(alpha)[1], alpha=0.25, label='test', color='r')

    #plt.legend(loc='upper right')

    train[alpha].plot(kind='hist',color='forestgreen')

    plt.title(alpha +('-train'))

    plt.subplot(6, 4, num+x+1)

    

    x=x+1

    test[alpha].plot(kind='hist',color='salmon')    

    plt.title(alpha +('-test'))

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(16,40)})

x=1

for num, alpha in enumerate(train.columns[11:54]):

    plt.subplot(22, 4, num+x)

    #plt.hist(diffcheck(alpha)[0], alpha=0.75, label='train', color='g')

    #plt.hist(diffcheck(alpha)[1], alpha=0.25, label='test', color='r')

    #plt.legend(loc='upper right')

    train[alpha].plot(kind='hist',color='forestgreen')

    plt.title(alpha +('-train'))

    plt.subplot(22, 4, num+x+1)

    

    x=x+1

    test[alpha].plot(kind='hist',color='salmon')    

    plt.title(alpha +('-test'))

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(8,4)})

plot1=sns.countplot(train.Cover_Type)

for p in plot1.patches[1:]:

    h = p.get_height()

    x = p.get_x()+p.get_width()/2.

    if h != 0:

        plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 

                   textcoords="offset points", ha="center", va="bottom")
sns.set(rc={'figure.figsize':(16,30)})

for num,alpha in enumerate(train.columns[1:11]):

    ax1= plt.subplot(6,2,num+1)

    sns.boxplot(data=train,y=alpha,x='Cover_Type',ax=ax1)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#sns.set(rc={'figure.figsize':(16,30)})

fig = plt.figure(figsize=(16,40))

for num,alpha in enumerate(train.columns[1:11]):

    ax1= plt.subplot(10,1,num+1)

    #train[alpha].plot(kind='hist')

    plot1=train.pivot(columns='Cover_Type')[alpha].plot(kind='hist',stacked=True,ax=ax1,legend=False)

    if num==1:

        handles, labels = plot1.get_legend_handles_labels()

    plot1=plot1.set_title(alpha)    

fig.legend(handles,labels,loc='upper right')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

#sns.set(rc={'figure.figsize':(16,40)})

fig = plt.figure(figsize=(16,40))

for num,alpha in enumerate(train.columns[11:55]):

    plt.subplot(15,3,num+1)

    plot1=sns.countplot(x=train[alpha], hue =train.Cover_Type)

    if num==1:

        handles, labels = plot1.get_legend_handles_labels()

    plot1.legend_.remove()

    for p in plot1.patches[1:]:

        h = p.get_height()

        x = p.get_x()+p.get_width()/2.

        if h != 0:

            plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 

                   textcoords="offset points", ha="center", va="bottom")    

fig.legend(handles,labels,loc='lower right')#, bbox_to_anchor=(0.5, -0.12), ncol=3)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(12,8)})

plot1=sns.scatterplot(data=train,x='Elevation',y='Slope',hue='Aspect')
plot2=sns.scatterplot(data=train,x='Hillshade_Noon',y='Hillshade_9am',hue='Aspect')
plot3=sns.scatterplot(data=train,x='Hillshade_Noon',y='Hillshade_3pm',hue='Aspect')
plot4=sns.scatterplot(data=train,x='Hillshade_9am',y='Hillshade_3pm',hue='Aspect')
plot5=sns.scatterplot(data=train,x='Elevation',y='Vertical_Distance_To_Hydrology',hue='Cover_Type')
plot6=sns.scatterplot(data=train,x='Horizontal_Distance_To_Hydrology',y='Horizontal_Distance_To_Roadways',hue='Cover_Type')
X=train.copy()
X['asp_cosine'] =  np.cos(np.radians(X.Aspect))

X['slope_cosine'] = X.Slope * X['asp_cosine']

X['asp_sine'] = np.sin(np.radians(X.Aspect))

X['slope_sine'] = X.Slope * X['asp_sine']

test['asp_cosine'] =  np.cos(np.radians(test.Aspect))

test['slope_cosine'] = test.Slope * test['asp_cosine']

test['asp_sine'] = np.sin(np.radians(test.Aspect))

test['slope_sine'] = test.Slope * test['asp_sine']

X['elevation_sq'] = X.Elevation **2

X['log_elevation'] = np.log(X.Elevation+1)

test['elevation_sq'] = test.Elevation **2

test['log_elevation'] = np.log(test.Elevation+1)

X['elev_slope']= X.Elevation/X.Slope

test['elev_slope']= test.Elevation/test.Slope

X['dist_hydrolgy']= (X.Vertical_Distance_To_Hydrology**2 +X.Horizontal_Distance_To_Hydrology**2)**0.5

test['dist_hydrolgy']= (test.Vertical_Distance_To_Hydrology**2+test.Horizontal_Distance_To_Hydrology**2)**0.5

X['elev_verthydrolgy'] = X.Elevation - X.Vertical_Distance_To_Hydrology

test['elev_verthydrolgy'] = test.Elevation - test.Vertical_Distance_To_Hydrology

X['shade3_to_noon']= X.Hillshade_Noon-X.Hillshade_3pm

X['shade9_to_noon']= X.Hillshade_Noon-X.Hillshade_9am

X['shade9_to_3']= X.Hillshade_9am-X.Hillshade_3pm

test['shade3_to_noon']= test.Hillshade_Noon-test.Hillshade_3pm

test['shade9_to_noon']= test.Hillshade_Noon-test.Hillshade_9am

test['shade9_to_3']= test.Hillshade_9am-test.Hillshade_3pm

X['shade_mean']= X.loc[:,['Hillshade_Noon','Hillshade_9am','Hillshade_3pm']].mean(axis=1)

X['shade_std']= X.loc[:,['Hillshade_Noon','Hillshade_9am','Hillshade_3pm']].std(axis=1)

test['shade_mean']= test.loc[:,['Hillshade_Noon','Hillshade_9am','Hillshade_3pm']].mean(axis=1)

test['shade_std']= test.loc[:,['Hillshade_Noon','Hillshade_9am','Hillshade_3pm']].std(axis=1)

X['distance_mean']= X.loc[:,['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']].mean(axis=1)

X['distance_std']= X.loc[:,['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']].std(axis=1)

test['distance_mean']= test.loc[:,['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']].mean(axis=1)

test['distance_std']= test.loc[:,['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']].std(axis=1)
y= X.Cover_Type

X.drop(['Id','Cover_Type'],axis=1,inplace=True)

test_id = test.Id

test.drop('Id',axis=1,inplace=True)

X.drop(['Elevation','Slope','Aspect'],axis=1,inplace=True)

test.drop(['Elevation','Slope','Aspect'],axis=1,inplace=True)
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

lgbmodel= LGBMClassifier(n_estimators=400,

                           metric='multi_error',

                           num_leaves=100,

                           learning_rate=0.05,

                           verbosity=1,

                           random_state=1,

                           n_jobs=-1)

lgbmodel.fit(X_train,y_train,eval_metric='multi_error')
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score

predict_test = lgbmodel.predict(X_test)

print(classification_report(y_test,predict_test))
feature_importance_df = pd.DataFrame({'feature': X.columns,'importance': lgbmodel.feature_importances_})

plt.figure(figsize=(14,25))

sns.barplot(x="importance",

             y="feature",

             data=feature_importance_df.sort_values(by="importance",

                                            ascending=False))

plt.title('LightGBM Features and importance')

plt.tight_layout()
predictions= lgbmodel.predict(test)
params1 = {

          "objective" : "multiclass",

          "num_class" : 8,

          "num_leaves" : 100,

          'n_jobs' : -1,

          "max_depth": -1,

          "learning_rate" : 0.01,

          'metric': 'multi_error',

          "bagging_fraction" : 0.9,  # subsample

          "feature_fraction" : 0.9,  # colsample_bytree

          "bagging_freq" : 5,        # subsample_freq

          "bagging_seed" : 1337,

          "verbosity" : -1 }
import lightgbm as lgb

from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold

features = [x for x in X.columns]

folds = RepeatedKFold(n_splits=3,n_repeats=2, random_state=0)

arr1=np.empty(shape=(len(X),8))

oof1 = np.zeros(len(X))

pred_arr1=np.empty(shape=(len(test),8))

predictions1 = np.zeros(len(test))

feature_importance_df1 = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X.values, y.values)):

    print("fold n°{}".format(fold_))

    trn_data = lgb.Dataset(X.iloc[trn_idx][features], label=y.iloc[trn_idx])

    val_data = lgb.Dataset(X.iloc[val_idx][features], label=y.iloc[val_idx])



    num_round = 1000

    clf1 = lgb.train(params1, trn_data, num_boost_round=num_round,valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 100)

    arr1[val_idx] = clf1.predict(X.iloc[val_idx][features], num_iteration=clf1.best_iteration)

    oof1[val_idx] = np.argmax(arr1[val_idx],axis=1)

    fold_importance_df1 = pd.DataFrame()

    fold_importance_df1["feature"] = features

    fold_importance_df1["importance"] = clf1.feature_importance()

    fold_importance_df1["fold"] = fold_ + 1

    feature_importance_df1 = pd.concat([feature_importance_df1, fold_importance_df1], axis=0)

    pred_arr1    += clf1.predict(test[features], num_iteration=clf1.best_iteration) / (3*2)

    

predictions1 = np.argmax(pred_arr1,axis=1)

print(classification_report(y,oof1))
cols = (feature_importance_df1[["feature", "importance"]]

         .groupby("feature")

         .mean()

         .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df1.loc[feature_importance_df1.feature.isin(cols)]



plt.figure(figsize=(14,25))

sns.barplot(x="importance",

             y="feature",

             data=best_features.sort_values(by="importance",

                                            ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()
import xgboost as xgb

from xgboost.sklearn import XGBClassifier

xgb1 = XGBClassifier(

 learning_rate =0.05,

 n_estimators=1000,

 max_depth=9,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'multi:softmax',

 metric='merror',

 num_class=7,

 scale_pos_weight=1,

 seed=1337)

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

xgb1.fit(X_train,y_train)
predict_test1 = xgb1.predict(X_test)

print(classification_report(y_test,predict_test1))
predictions2= xgb1.predict(test)
feature_importance_df2 = pd.DataFrame({'feature': X.columns,'importance': xgb1.feature_importances_})

plt.figure(figsize=(14,25))

sns.barplot(x="importance",

             y="feature",

             data=feature_importance_df2.sort_values(by="importance",

                                            ascending=False))

plt.title('XBoost Model  Features and importance')

plt.tight_layout()
import statistics

final_pred = np.array([])

for i in range(0,len(test)):

    if len(set([predictions[i], predictions1[i], predictions2[i]])) == len([predictions[i], predictions1[i], predictions2[i]]):

        final_pred = np.append(final_pred,predictions[i])

    else:

        final_pred = np.append(final_pred, statistics.mode([predictions[i], predictions1[i], predictions2[i]]))

final_pred = final_pred.astype(int)


submit = pd.DataFrame({'Id': test_id,

                       'Cover_Type': final_pred})

submit.to_csv('submission.csv', index=False)