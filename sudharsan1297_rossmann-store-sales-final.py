# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')

test=pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')

store=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
print(train.shape)

print(test.shape)

print(store.shape)
train.head()
store.head()
train.info()
train.describe()
train['StateHoliday'].value_counts()
train.describe()[['Sales','Customers']]
train.describe()[['Sales','Customers']].loc['mean']
train.describe()[['Sales','Customers']].loc['min']
train.describe()[['Sales','Customers']].loc['max']
train['Store'].value_counts().head(20)
train['Store'].value_counts().tail(20)
train['DayOfWeek'].value_counts()
train['Open'].value_counts()
train['Promo'].value_counts()
train['Date']=pd.to_datetime(train['Date'],format='%Y-%m-%d')
train.isna().sum()
test.isna().sum()
store.isna().sum()
store1=train[train['Store']==1]
store1.head()
store1.shape
store1.resample('1d',on='Date')['Sales'].sum().plot.line(figsize=(15,5))
store1[store1['Sales']==0]
test_store1=test[test['Store']==1]

test_store1['Date']=pd.to_datetime(test_store1['Date'],format='%Y-%m-%d')
test_store1['Date'].min(),test_store1['Date'].max()
test_store1['Open'].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt

plt.hist(store1['Sales'])
store[store['Store']==1].T
store[~store['Promo2SinceYear'].isna()].iloc[0]
store=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
store['Promo2SinceWeek']=store['Promo2SinceWeek'].fillna(0)
store['Promo2SinceYear']=store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])
store['PromoInterval']=store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])
store['CompetitionDistance']=store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())

store['CompetitionOpenSinceMonth']=store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])

store['CompetitionOpenSinceYear']=store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])
store.isna().sum()
df=train.merge(store,on='Store',how='left')
print(train.shape)

print(df.shape)
df.isna().sum()
df.info()
df['day']=df['Date'].dt.day

df['year']=df['Date'].dt.year

df['month']=df['Date'].dt.month
#df['Date'].dt.strftime('%a')
#Dummies: StateHoliday,StoreType,Assortment,PromoInterval



df['StateHoliday']=df['StateHoliday'].apply(lambda x:'0' if x==0 or x=='0' else x)
df['StateHoliday']=df['StateHoliday'].map({'0':0,'a':1,'b':2,'c':3})

df['StateHoliday']=df['StateHoliday'].astype(int)
df['StoreType'].value_counts()
df['StoreType']=df['StoreType'].map({'a':0,'b':1,'c':2,'d':3})
df['Assortment']=df['Assortment'].map({'a':0,'b':1,'c':2})
df['Assortment']=df['Assortment'].astype(int)
df['PromoInterval'].value_counts()
df['PromoInterval']=df['PromoInterval'].map({'Jan,Apr,Jul,Oct':0,'Feb,May,Aug,Nov':1,'Mar,Jun,Sept,Dec':2})
df['PromoInterval']=df['PromoInterval'].astype(int)
df=df.drop('Date',1)
df.dtypes
y=np.log1p(df['Sales'])
X=df.drop(['Sales','Customers'],1)
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val=train_test_split(X,y,random_state=1,test_size=0.2)
y.plot.hist()
from sklearn.tree import DecisionTreeRegressor



dt=DecisionTreeRegressor(max_depth=11,random_state=1).fit(X_train,y_train)
y_pred_val=dt.predict(X_val)
from sklearn.metrics import r2_score,mean_squared_error



print(r2_score(y_val,y_pred_val))

print(np.sqrt(mean_squared_error(y_val,y_pred_val)))
y_val_exp=np.exp(y_val)-1

y_pred_val_exp=np.exp(y_pred_val)-1

np.sqrt(mean_squared_error(y_val_exp,y_pred_val_exp))
r2_score(y_val_exp,y_pred_val_exp)
!pip install pydotplus
def draw_tree(model, columns):

    import pydotplus

    from sklearn.externals.six import StringIO

    from IPython.display import Image

    import os

    from sklearn import tree

    

    graphviz_path = 'C:\Program Files (x86)\Graphviz2.38/bin/'

    os.environ["PATH"] += os.pathsep + graphviz_path



    dot_data = StringIO()

    tree.export_graphviz(model,

                         out_file=dot_data,

                         feature_names=columns)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

    return Image(graph.create_png())
draw_tree(dt,X_train.columns)
print(dt.feature_importances_)

plt.barh(y=X_train.columns,width=dt.feature_importances_)

plt.show()
test.head()
avg_cust=df.groupby(['Store'])[['Customers']].mean().astype(int)

test1=test.merge(avg_cust,on='Store',how='left')
test.shape,test1.shape
test_merged=test1.merge(store,on='Store',how='left')
test1.shape,test_merged.shape
test_merged.head()
test_merged['Open'].fillna(1,inplace=True)
test_merged.isna().sum()
test_merged['Date']=pd.to_datetime(test_merged['Date'],format='%Y-%m-%d')

test_merged['day']=test_merged['Date'].dt.day

test_merged['month']=test_merged['Date'].dt.month

test_merged['year']=test_merged['Date'].dt.year

test_merged=test_merged.drop('Date',1)
test_merged['StateHoliday']=test_merged['StateHoliday'].apply(lambda x:'0' if x==0 or x=='0' else x)
test_merged['StateHoliday'].value_counts()
test_merged['StoreType'].value_counts()
test_merged['Assortment'].value_counts()
test_merged['PromoInterval'].value_counts()
test_merged['StateHoliday']=test_merged['StateHoliday'].map({'0':0,'a':1})

test_merged['StateHoliday']=test_merged['StateHoliday'].astype(int)



test_merged['StoreType']=test_merged['StoreType'].map({'a':0,'b':1,'c':2,'d':3})

test_merged['StoreType']=test_merged['StoreType'].astype(int)



test_merged['Assortment']=test_merged['Assortment'].map({'a':0,'b':1,'c':2})

test_merged['Assortment']=test_merged['Assortment'].astype(int)



test_merged['PromoInterval']=test_merged['PromoInterval'].map({'Jan,Apr,Jul,Oct':0,'Feb,May,Aug,Nov':1,'Mar,Jun,Sept,Dec':2})

test_merged['PromoInterval']=test_merged['PromoInterval'].astype(int)
test_merged1=test_merged.drop('Id',1)
test_merged1.head()
X_train.head()
test_merged1.shape
y_pred=dt.predict(test_merged1[X_train.columns])
y_pred
y_pred_exp=np.exp(y_pred)-1
submission_pred=pd.DataFrame(test_merged['Id'],columns=['Id'])
submission_pred['Sales']=y_pred_exp
submission_pred['Id']=np.arange(1,len(submission_pred)+1)
submission_pred
submission_pred.to_csv('Submission.csv',index=False)
# Credit: kaggle.com

def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w



def rmspe(y, yhat):

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe



rmse_val=np.sqrt(mean_squared_error(y_val_exp,y_pred_val_exp))

rmspe_val=rmspe(y_val_exp,y_pred_val_exp)

print(rmse_val,rmspe_val)
from sklearn.model_selection import GridSearchCV

params={'max_depth':list(range(5,20))}

base_model=DecisionTreeRegressor()

cv_model=GridSearchCV(base_model,param_grid=params,return_train_score=True).fit(X_train,y_train)
df_cv_results=pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)
df_cv_results.set_index('param_max_depth')['mean_test_score'].plot.line()

df_cv_results.set_index('param_max_depth')['mean_train_score'].plot.line()

plt.show()
df_cv_results=pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)[['param_max_depth','mean_test_score','mean_train_score']]

df_cv_results
def get_rmspe_score(model,input_values,y_actual):

    y_predicted=model.predict(input_values)

    y_actual=np.exp(y_actual)-1

    y_predicted=np.exp(y_predicted)-1

    score=rmspe(y_actual,y_predicted)

    return score



params={'max_depth':list(range(5,8))}

base_model=DecisionTreeRegressor()

cv_model=GridSearchCV(base_model,param_grid=params,return_train_score=True,scoring=get_rmspe_score).fit(X_train,y_train)

pd.DataFrame(cv_model.cv_results_)[['params','mean_test_score','mean_train_score']]
from sklearn.ensemble import AdaBoostRegressor



model_ada=AdaBoostRegressor(n_estimators=5).fit(X_train,y_train)
model_ada.estimators_[0]
def draw_tree(model, columns):

    import pydotplus

    from sklearn.externals.six import StringIO

    from IPython.display import Image

    import os

    from sklearn import tree

    

    graphviz_path = 'C:\Program Files (x86)\Graphviz2.38/bin/'

    os.environ["PATH"] += os.pathsep + graphviz_path



    dot_data = StringIO()

    tree.export_graphviz(model,

                         out_file=dot_data,

                         feature_names=columns)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

    return Image(graph.create_png())
features=X_train.columns

draw_tree(model_ada.estimators_[0],features)
import xgboost as xgb
dtrain=xgb.DMatrix(X_train,y_train)

dvalidate=xgb.DMatrix(X_val,y_val)



param={'max_depth':5,'eta':1,'ojective':'reg:linear'}

model_xg=xgb.train(param,dtrain,200)

pred_y=model_xg.predict(dvalidate)





val_y_inv=np.exp(y_val)-1

pred_y_inv=np.exp(pred_y)-1

rmspe_val=rmspe(val_y_inv,pred_y_inv)

print(rmspe_val)
test_merged=test_merged.drop(['Id','Customers'],1)
y_pred_xg=model_xg.predict(xgb.DMatrix(test_merged[X_train.columns]))
y_pred_xg_exp=np.exp(y_pred_xg)-1
y_pred_xg_exp
submission_predicted1 = pd.DataFrame({'Id': test['Id'], 'Sales': y_pred_xg_exp})

testop0=(test[test['Open']==0]['Open']).index

Sales1=[]

for i in range(41088):

    if i in testop0:

        Sales1.append(0)

    else:

        Sales1.append(submission_predicted1['Sales'][i])

submission_predicted1['Sales']=Sales1

print(submission_predicted1.head())

submission_predicted1.to_csv('submission.csv', index=False)