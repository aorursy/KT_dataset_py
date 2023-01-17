import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import entropy
ref_df=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/train.csv')
print(ref_df.isna().sum().sum())
ref_df.head()
sample_sub=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
print(sample_sub.isna().sum().sum())
sample_sub.head()
%%time
missings=pd.DataFrame()
for seg_id in ref_df['segment_id']:
    df=pd.read_csv(f'../input/predict-volcanic-eruptions-ingv-oe/train/{seg_id}.csv')
    missings=missings.append(pd.DataFrame(df.isna().sum()).T)
    
    
missings.set_index(ref_df['segment_id'],inplace=True)
print(missings.head())
plt.figure(figsize=(15,25))
sns.heatmap(missings,cmap = 'magma_r')
%%time

train=pd.DataFrame()
for seg_id in ref_df['segment_id']:
    df=pd.read_csv(f'../input/predict-volcanic-eruptions-ingv-oe/train/{seg_id}.csv')
    #print(df.isna().sum())
    for col in df.columns:
        if df[col].isna().sum()==len(df):
            df[col].fillna(0,inplace=True)
        else:
            df[col].fillna(df[col].mean(),inplace=True)
    #print(df.isna().sum())
    #print(df.head())
    summary=pd.DataFrame()
    des=df.describe()
    
    # df describe statistics
    for i in range(10):
        col=pd.DataFrame(des.iloc[:,i][1:]).T.reset_index(drop=True)
        summary=pd.concat([summary,col],axis=1)
    
    # skew statistics
    stat=pd.DataFrame(df.skew().values.reshape(1,-1))
    summary=pd.concat([summary,stat],axis=1)
    
    # mean absolute deviation statistics
    stat=pd.DataFrame(df.mad().values.reshape(1,-1))
    summary=pd.concat([summary,stat],axis=1)
    
    # standard error statistics
    stat=pd.DataFrame(df.sem().values.reshape(1,-1))
    summary=pd.concat([summary,stat],axis=1)

    #print(summary)
    
    train=train.append(summary)
    del summary, des, df
    
print(train.shape)
train.columns=range(train.shape[1])
train.set_index(ref_df['segment_id'].values,inplace=True)
print(train.shape)
train.to_csv('train__.csv', index=False)
train.head()
%%time

test=pd.DataFrame()
for seg_id in sample_sub['segment_id']:
    df=pd.read_csv(f'../input/predict-volcanic-eruptions-ingv-oe/test/{seg_id}.csv')
    #print(df.isna().sum())
    for col in df.columns:
        if df[col].isna().sum()==len(df):
            df[col].fillna(0,inplace=True)
        else:
            df[col].fillna(df[col].mean(),inplace=True)
    #print(df.isna().sum())
    #print(df.head())
    summary=pd.DataFrame()
    des=df.describe()

    
    for i in range(10):
        col=pd.DataFrame(des.iloc[:,i][1:]).T.reset_index(drop=True)
        summary=pd.concat([summary,col],axis=1)
    
    # skew statistics
    stat=pd.DataFrame(df.skew().values.reshape(1,-1))
    summary=pd.concat([summary,stat],axis=1)
    
    # mean absolute deviation statistics
    stat=pd.DataFrame(df.mad().values.reshape(1,-1))
    summary=pd.concat([summary,stat],axis=1)
    
    # standard error statistics
    stat=pd.DataFrame(df.sem().values.reshape(1,-1))
    summary=pd.concat([summary,stat],axis=1)
    
    #print(summary)
    
    test=test.append(summary)
    del summary, des, df
    
print(test.shape)
test.columns=range(test.shape[1])
test.set_index(sample_sub['segment_id'].values,inplace=True)
print(test.shape)
test.to_csv('test__.csv', index=False)
test.head()
train_df=pd.read_csv('./train__.csv')
test_df=pd.read_csv('./test__.csv')
X=train_df.values
y=ref_df['time_to_eruption']
x_tr,x_ts,y_tr,y_ts=train_test_split(X,y,test_size=.25,random_state=40)
x_tr.shape
from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()
clf .fit(x_tr,y_tr)
pred=clf.predict(x_ts)
print('training mae: ',mae(y_tr,clf.predict(x_tr)))
print('testing mae: ',mae(y_ts,pred))
%%time
params={'max_depth':range(13,18,3), 'min_samples_split':range(2,20,3), 'min_samples_leaf':range(1,20,4)}
clf=DecisionTreeRegressor()
clfcv=RandomizedSearchCV(clf,params,n_iter=1000,n_jobs=-1,random_state=10,return_train_score=True)
clfcv.fit(x_tr,y_tr)

print('train error: ',mae(y_tr,clfcv.best_estimator_.predict(x_tr)))
print('test error: ',mae(y_ts,clfcv.best_estimator_.predict(x_ts)))
clfcv.best_estimator_
feat_im=pd.DataFrame({'features':x_tr.columns,'importance':clfcv.best_estimator_.feature_importances_})
feat_im.sort_values(by='importance',inplace=True,ascending=False)
plt.figure(figsize=(9,18))
sns.barplot(y='features',x='importance',data=feat_im)
columns=feat_im[feat_im['importance']>=.015]['features'].values
x_tr_n=x_tr[columns]
x_ts_n=x_ts[columns]
x_tr_n.shape
%%time
params={'max_depth':range(13,18),'min_samples_split':range(2,20,3), 'min_samples_leaf':range(1,20,2)}
        #'max_features':["auto", "sqrt", "log2"]}
clf=DecisionTreeRegressor()
clfcv=GridSearchCV(clf,params,n_jobs=-1,cv=5)
clfcv.fit(x_tr_n,y_tr)

print('train error: ',mae(y_tr,clfcv.best_estimator_.predict(x_tr_n)))
print('test error: ',mae(y_ts,clfcv.best_estimator_.predict(x_ts_n)))
clfcv.best_estimator_
from sklearn.ensemble import RandomForestRegressor
%%time
params={'max_depth':range(13,18),'min_samples_split':range(2,20,3), 'min_samples_leaf':range(1,20,2)}
clf=RandomForestRegressor(n_estimators=15)
clfcv=RandomizedSearchCV(clf,params,n_iter=100,n_jobs=-1,random_state=23,cv=10)
clfcv.fit(x_tr_n,y_tr)
print('train error: ',mae(y_tr,clfcv.best_estimator_.predict(x_tr_n)))
print('test error: ',mae(y_ts,clfcv.best_estimator_.predict(x_ts_n)))
clfcv.best_estimator_
# subfile
test_n=test[columns]
pred=clfcv.predict(test_n)
sub_file['time_to_eruption']=pred
sub_file.to_csv('sub.csv',index=False)
sub_file.head()
