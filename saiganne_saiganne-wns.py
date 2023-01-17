# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm_notebook

from sklearn.metrics import roc_auc_score,confusion_matrix,average_precision_score

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.model_selection import train_test_split,StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import RandomOverSampler

from lightgbm import LGBMClassifier



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        





# Any results you write to the current directory are saved as output.
train= pd.read_csv("/kaggle/input/wns-analyticswizard/train_na17sgz/train.csv")

item_data=pd.read_csv("/kaggle/input/wns-analyticswizard/train_na17sgz/item_data.csv")

view_log=pd.read_csv("/kaggle/input/wns-analyticswizard/train_na17sgz/view_log.csv")

test = pd.read_csv("/kaggle/input/wns-analyticswizard/test_aq1fgdb/test.csv")

sample = pd.read_csv("/kaggle/input/wns-analyticswizard/sample_submission_ipsblct/sample_submission.csv")

train['impression_time']=pd.to_datetime(train['impression_time'])

test['impression_time']=pd.to_datetime(test['impression_time'])

view_log['server_time']=pd.to_datetime(view_log['server_time'])

test['is_click']=np.nan

train['data']='train'

test['data']='test'

test=test[train.columns]

cd_all=pd.concat([train,test],axis=0)
print(train.dtypes)

print("-----------------------")

print(train.shape)

print(item_data.shape)

print(view_log.shape)

print(test.shape)
train.head()
#Create new data d1 like this : Sort cd_all by user_id and impression date [ remove other columns ]

#Remove duplicate observations from d1 

d1=cd_all.sort_values(['user_id','impression_time'])



d1=d1[['user_id','impression_time']]



print(d1.shape)

d1=d1.drop_duplicates()

print(d1.shape)
#Create new data d2 : sort cd_viewlog by user_id and server_date [ remove other columns ]

d2=view_log.sort_values(['user_id','server_time'])

d2=d2[['user_id','server_time']]

print(d2.shape)

d2=d2.drop_duplicates()

print(d2.shape)
#end_data={'user_id':[],'impression_time':[],'result':[]}



#for id in tqdm_notebook(d1['user_id'].unique()):

#    for date in d1.loc[d1['user_id']==id,'impression_time']:

#        d2_sub=d2[d2['user_id']==id]

#        result=(pd.to_datetime(date-d2_sub['server_time']).dt.day<10).sum()

#        end_data['user_id'].append(id)

#        end_data['impression_time'].append(date)

#        end_data['result'].append(result)



#The above for loop  generate data in end_date, i saved that csv file and loading it to save the execution time

end_data= pd.read_csv("/kaggle/input/feature/end_data.csv")





#end_data=pd.DataFrame(end_data)
print(end_data.shape)

print(train.shape)

print(item_data.shape)

print(view_log.shape)
end_data.head()
#Nearest_date={'user_id':[],'impression_time':[],'Nearest_serverdate':[]}

#for id in tqdm_notebook(d1['user_id'].unique()):   #tqdm_notebook will give display graphical progress bar

#    for date in d1.loc[d1['user_id']==id,'impression_time']:

#        d2_sub=d2[d2['user_id']==id]

#        b=51909

#        date2 = pd.to_datetime('2018-11-26 23:30:00')

#        for date1 in d2_sub['server_time']:

#            a=pd.to_datetime(date)-pd.to_datetime(date1)

#            k=a/pd.Timedelta(1, unit='d')

#            if k>0:        

#                if k>=b:

#                    b=b

#                    date2=date2

#                else:

#                    b=k

#                    date2=date1

#        Nearest_date['user_id'].append(id)

#        Nearest_date['impression_time'].append(date)

#        Nearest_date['Nearest_serverdate'].append(date2)





#pd.DataFrame(Nearest_date).to_csv("Nearest_date.csv",index=False)

Nearest_date= pd.read_csv("../input/nearest-date/Nearest_date.csv")
Nearest_date.head()
print(cd_all.dtypes)

print("===========")

print(end_data.dtypes)

print("===========")

print(Nearest_date.dtypes)
end_data['impression_time']=pd.to_datetime(end_data['impression_time']) # changing the dtype

Nearest_date['impression_time']=pd.to_datetime(Nearest_date['impression_time']) # changing the dtype

Nearest_date['Nearest_serverdate']=pd.to_datetime(Nearest_date['Nearest_serverdate']) # changing the dtype
cd_all=pd.merge(cd_all,end_data,on=['user_id','impression_time'],how='left')



cd_all=pd.merge(cd_all,Nearest_date,on=['user_id','impression_time'],how='left')



#cd_all=pd.merge(cd_all,view_log,how='left',left_on=['user_id','Nearest_serverdate'],right_on=['user_id','server_time'])

#cd_all=pd.merge(cd_all,item_data,on='item_id',how='left')
cd_all.shape
cd_all.dtypes
from math import ceil



def week_of_month(dt):

    """ Returns the week of the month for the specified date.

    """



    first_day = dt.replace(day=1)



    dom = dt.day

    adjusted_dom = dom + first_day.weekday()



    return int(ceil(adjusted_dom/7.0))
cd_all["imp_month"] = cd_all['impression_time'].dt.month

cd_all['imp_wom']=cd_all['impression_time'].apply(lambda x: week_of_month(x))

cd_all["imp_weekday"] = cd_all['impression_time'].dt.weekday
cd_all["near_month"] = cd_all['Nearest_serverdate'].dt.month

cd_all['near_wom']=cd_all['Nearest_serverdate'].apply(lambda x: week_of_month(x))

cd_all["near_weekday"] = cd_all['Nearest_serverdate'].dt.weekday
# dropping  column

for col in ['impression_time','Nearest_serverdate']:

    cd_all.drop([col],axis=1,inplace=True)

  
cd_all['os_version']=cd_all['os_version'].map({'latest':3,'intermediate':2,'old':'1'})

cd_all['os_version']=cd_all['os_version'].astype(int)
cd_all.dtypes
temp=pd.get_dummies(cd_all['imp_month'],drop_first=True)

cd_all=pd.concat([cd_all,temp],axis=1)

cd_all.drop(['imp_month'],axis=1,inplace=True)



temp=pd.get_dummies(cd_all['imp_wom'],drop_first=True)

cd_all=pd.concat([cd_all,temp],axis=1)

cd_all.drop(['imp_wom'],axis=1,inplace=True)



temp=pd.get_dummies(cd_all['imp_weekday'],drop_first=True)

cd_all=pd.concat([cd_all,temp],axis=1)

cd_all.drop(['imp_weekday'],axis=1,inplace=True)



temp=pd.get_dummies(cd_all['near_month'],drop_first=True)

cd_all=pd.concat([cd_all,temp],axis=1)

cd_all.drop(['near_month'],axis=1,inplace=True)



temp=pd.get_dummies(cd_all['near_wom'],drop_first=True)

cd_all=pd.concat([cd_all,temp],axis=1)

cd_all.drop(['near_wom'],axis=1,inplace=True)



temp=pd.get_dummies(cd_all['near_weekday'],drop_first=True)

cd_all=pd.concat([cd_all,temp],axis=1)

cd_all.drop(['near_weekday'],axis=1,inplace=True)





temp=pd.get_dummies(cd_all['os_version'],drop_first=True)

cd_all=pd.concat([cd_all,temp],axis=1)

cd_all.drop(['os_version'],axis=1,inplace=True)
cd_all['app_code']=cd_all['app_code'].astype(str)

for col in ['app_code']:

    freqs=cd_all[col].value_counts()

    k=freqs.index[freqs>=100][:-1]

    for cat in k:

        name=col+'_'+cat

        cd_all[name]=(cd_all[col]==cat).astype(int)

    del cd_all[col]

    print(col)
cd_all=cd_all.rename(columns={'result': 'count_in_log_for10days'})
cd_all.drop(['user_id'],axis=1,inplace=True)  
cd_all.shape
train_df=cd_all[cd_all['data']=='train']

del train_df['data']

test_df=cd_all[cd_all['data']=='test']

test_df.drop(['is_click','data'],axis=1,inplace=True)
print(train_df.shape)

print(test_df.shape)
train_df.drop(['impression_id'],axis=1,inplace=True)

test_df.drop(['impression_id'],axis=1,inplace=True)
#t1,t2 = train_test_split(train_df,test_size=0.30,stratify=train['is_click'],random_state=12)



#t1_x=t1.drop(['is_click'],axis=1)

#t1_y=t1['is_click']

#t2_x=t2.drop(['is_click'],axis=1)

#t2_y=t2['is_click']



#t1col=t1_x.columns

#t2col=t2_x.columns



#ros = RandomOverSampler(random_state=143)

#t1_x,t1_y = ros.fit_resample(t1_x,t1_y)



#scaler=StandardScaler()

#scaler.fit(t1_x)

#t1_x=scaler.transform(t1_x)

#scaler.fit(t2_x)

#t2_x=scaler.transform(t2_x)



#t1_x = pd.DataFrame(t1_x, columns = t1col)

#t2_x=pd.DataFrame(t2_x, columns = t2col)
train_df_x=train_df.drop(['is_click'],axis=1)

train_df_y=train_df['is_click']

test_df_x=test_df



train_dfcol=train_df_x.columns

test_dfcol=test_df_x.columns



ros = RandomOverSampler(random_state=143)

train_df_x,train_df_y = ros.fit_resample(train_df_x,train_df_y)



scaler=StandardScaler()

scaler.fit(train_df_x)

train_df_x=scaler.transform(train_df_x)

scaler.fit(test_df_x)

test_df_x=scaler.transform(test_df_x)



train_df_x = pd.DataFrame(train_df_x, columns = train_dfcol)

test_df_x=pd.DataFrame(test_df_x, columns = test_dfcol)
# params={'class_weight':['balanced'],

#        'penalty':['l2'],

#         'C':[0.05]

#       }



#glm=LogisticRegression(fit_intercept=True)

#grid_search=GridSearchCV(glm,cv=3,param_grid=params,scoring="roc_auc",verbose=True,n_jobs=-1)

#grid_search.fit(train_df_x,train_df_y)

#grid_search.best_estimator_
clf1=LogisticRegression(C=0.05, class_weight='balanced', dual=False,

                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,

                   max_iter=100, multi_class='warn', n_jobs=None, penalty='l2',

                   random_state=None, solver='warn', tol=0.0001, verbose=0,

                   warm_start=False)



clf2=LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

               importance_type='split', learning_rate=0.1, max_depth=-1,

               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

               n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,

               random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,

               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)





Algos=[clf1,clf2]
rows=train_df_x.shape[0]

rows
layer1=pd.DataFrame({'clf1':np.zeros(rows),'clf2':np.zeros(rows)})
kf=StratifiedKFold(n_splits=10)
fold=1



for train,left_out_chunk in kf.split(train_df_x,train_df_y):

    print('fold number : ', fold)

    

    for i,clf in enumerate(Algos):

        print('Algo number :',i+1)

        

        train_df_x_train=train_df_x.loc[train]

        train_df_y_train=train_df_y[train]

        train_df_x_left_out_chunk=train_df_x.loc[left_out_chunk]

        

        clf.fit(train_df_x_train,train_df_y_train)

        p=clf.predict_proba(train_df_x_left_out_chunk)[:,1]

        

        layer1.iloc[left_out_chunk,i]=p

        

    fold+=1      
rows=test_df_x.shape[0]

layer2_test=pd.DataFrame({'clf1':np.zeros(rows),'clf2':np.zeros(rows)})
#layer2_test
for i,clf in enumerate(Algos):

    print( 'Algo number',i+1)

    clf.fit(train_df_x,train_df_y)

    p=clf.predict_proba(test_df_x)[:,1]

    

    layer2_test.iloc[:,i]=p

# second layer linear model 

logr=LogisticRegression(class_weight='balanced')
logr.fit(layer1,train_df_y)
output=logr.predict_proba(layer2_test)[:,1]
output=pd.DataFrame(output)

output=output.rename(columns={0: 'is_click'})

output=pd.concat([test['impression_id'],output],axis=1)

pd.DataFrame(output).to_csv("output.csv",index=False)