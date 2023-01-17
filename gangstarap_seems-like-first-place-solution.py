# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/TTiDS20/train.csv')
test_ = pd.read_csv('/kaggle/input/TTiDS20/test_no_target.csv')
city=pd.read_csv('/kaggle/input/TTiDS20/zipcodes.csv')
data=pd.concat([data,test_],ignore_index=True)
data.fillna(-1,inplace=True)
city=city.drop_duplicates(subset ="zipcode")

'''tst=pd.read_csv('/kaggle/input/TTiDS20/test_no_target.csv')
tr=pd.read_csv('/kaggle/input/TTiDS20/train.csv')'''
#data.loc[data['registration_year']<1000,'registration_year']=2020
data=pd.merge(data,city,on=['zipcode'],how='left').drop(['Unnamed: 0_x','Unnamed: 0_y'],axis=1)
def metric(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
import matplotlib.pyplot as plt
import seaborn as sb
data['price']=np.log(data['price']+2)

data['lang_long']=(data['latitude']**2+data['longitude']**2)
from itertools import combinations
data['registration_year_mod10']= data['registration_year'] %10
data['registration_year_mod100']= data['registration_year'] %100
data['registration_year_year']=(data['registration_year']/10).astype(str).apply(lambda x:x[:3])
data['registration_year_centry']=(data['registration_year']/100).astype(str).apply(lambda x:x[:2])
'''for x in list(combinations(['registration_year_mod100','gearbox','model','fuel','brand','city'],2))+list(combinations(['registration_year_mod100','gearbox','model','fuel','brand','city'],3)):
        if len(x)==2:
            data[x[0]+'_'+x[1]]=data[x[0]].astype(str)+data[x[1]].astype(str)
        if len(x)==3:
            data[x[0]+'_'+x[1]+'_'+x[2]]=data[x[0]].astype(str)+data[x[1]].astype(str)+data[x[2]].astype(str)
'''
'''for col,typ in zip(data.dtypes.keys(),data.dtypes.values):
    if str(typ) =='object':
        print(col)
        data[col]=data[col].astype('category').cat.codes'''
from sklearn.model_selection import train_test_split 
data.drop(['lang_long','latitude','longitude', 'zipcode','city'],axis=1,inplace=True)
test=data[-test_.shape[0]:]
train,valid = train_test_split(data[:-test_.shape[0]],test_size=0.25,random_state=3)
'''def mean_encode(col,train,valid,test,target,smooth=False):
    globalmean =train[target].mean()
    if smooth:
        alpha = 100

        nrows = train.groupby(col)[target].count()
        targetmean = train.groupby(col)[target].mean()
        smooth = (targetmean*nrows + globalmean*alpha) / (nrows + alpha)
    else:
        smooth=train.groupby(col)[target].mean()
    train['mean_'+col] = train[col].map(smooth)
    train['mean_'+col].fillna(globalmean, inplace=True)

    valid['mean_'+col]=valid[col].map(smooth)
    valid['mean_'+col].fillna(globalmean, inplace=True)
    
    test['mean_'+col]=test[col].map(smooth)
    test['mean_'+col].fillna(globalmean, inplace=True)'''
data.isna().sum(),data.shape
from catboost import CatBoostRegressor
sample=pd.read_csv('/kaggle/input/TTiDS20/sample_submission.csv')
def num_folds(k):
    test=data[-test_.shape[0]:]

    train,valid = train_test_split(data[:-test_.shape[0]],test_size=0.2,random_state=k)
    X_train = train.drop('price',axis=1)
    X_valid = valid.drop('price',axis=1)
    y_train = train.price
    y_valid = valid.price
    categories = ['type','registration_year','gearbox','power','model','fuel','brand']
    categories+=['registration_year_mod10','registration_year_mod100','registration_year_year','registration_year_centry']
    
    model = CatBoostRegressor(
        iterations=400,
        random_seed=63,
        learning_rate=0.1,
        eval_metric='MAPE',
        use_best_model=True,
        max_depth=10
    )
    model.fit(
        X_train, y_train,
        cat_features=categories,
        eval_set=(X_valid, y_valid),
         logging_level='Silent',
    plot=False
    )
    print('METriC_'+str(k),metric(np.exp(y_valid)+2,np.exp(model.predict(X_valid))+2))
    print(sorted(zip(model.feature_importances_,model.feature_names_)))
    if metric(np.exp(y_valid)+2,np.exp(model.predict(X_valid))+2)<23.5:
        print('Added fold',k)
        sample['fold_'+str(k)]=np.exp(model.predict(test[X_train.columns]))-2
    
for i in range(5):
    num_folds(i)
sample['Predicted']=sample[sample.columns[2:]].mean(axis=1)
#sample=pd.read_csv('/kaggle/input/TTiDS20/sample_submission.csv')
sample.head()
'''#sample=pd.read_csv('/kaggle/input/TTiDS20/sample_submission.csv')
sample['Predicted']=sample['Predicted']/10
sample.head()'''
sample.head()
sample[['Id','Predicted']].to_csv('baseline1.csv',index=False)
from IPython.display import FileLink
FileLink(r'baseline1.csv')
sorted(zip(model.feature_importances_,model.feature_names_))


