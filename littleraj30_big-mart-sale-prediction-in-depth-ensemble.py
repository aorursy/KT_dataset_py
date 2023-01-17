import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/Train.csv')
df.head()
df.describe()
df.hist(figsize=(15,12))
df.info()
corr_matrix=df.corr()

corr_matrix['Item_Outlet_Sales']
df.Item_Identifier.value_counts()
df.Item_Fat_Content.value_counts()
df.Item_Fat_Content=df.Item_Fat_Content.replace('LF','Low Fat')
df.Item_Fat_Content=df.Item_Fat_Content.replace('reg','Regular')

df.Item_Fat_Content=df.Item_Fat_Content.replace('low fat','Low Fat')
df.Item_Fat_Content.value_counts()
df.Item_Identifier=df.Item_Identifier.astype('category')

df.Item_Fat_Content=df.Item_Fat_Content.astype('category')

df.Item_Type=df.Item_Type.astype('category')

df.Outlet_Identifier=df.Outlet_Identifier.astype('category')

df.Outlet_Establishment_Year=df.Outlet_Establishment_Year.astype('int64')



df.Outlet_Type=df.Outlet_Type.astype('category')

df.Outlet_Location_Type=df.Outlet_Location_Type.astype('category')

df.Outlet_Size=df.Outlet_Size.astype('category')
fig,axes=plt.subplots(1,1,figsize=(12,8))

sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=df)
df.describe()
fig,axes=plt.subplots(1,1,figsize=(10,8))

sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=df)

plt.plot([69,69],[0,5000])

plt.plot([137,137],[0,5000])

plt.plot([203,203],[0,9000])
df.Item_MRP=pd.cut(df.Item_MRP,bins=[25,69,137,203,270],labels=['a','b','c','d'],right=True)
df.head()
fig,axes=plt.subplots(3,1,figsize=(15,12))

sns.scatterplot(x='Item_Visibility',y='Item_Outlet_Sales',hue='Item_MRP',ax=axes[0],data=df)

sns.boxplot(x='Item_Type',y='Item_Outlet_Sales',ax=axes[1],data=df)

sns.boxplot(x='Outlet_Identifier',y='Item_Outlet_Sales',ax=axes[2],data=df)
fig,axes=plt.subplots(2,2,figsize=(15,12))

sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',ax=axes[0,0],data=df)

sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',ax=axes[0,1],data=df)

sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',ax=axes[1,0],data=df)

sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',ax=axes[1,1],data=df)
attributes=['Item_MRP','Outlet_Type','Outlet_Location_Type','Outlet_Size','Outlet_Establishment_Year','Outlet_Identifier','Item_Type','Item_Outlet_Sales']
fig,axes=plt.subplots(2,2,figsize=(15,12))

sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[0,0],data=df)

sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[0,1],data=df)

sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[1,0],data=df)

sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[1,1],data=df)
data=df[attributes]
data.info()
fig,axes=plt.subplots(1,1,figsize=(8,6))

sns.boxplot(y='Item_Outlet_Sales',hue='Outlet_Type',x='Outlet_Location_Type',data=data)
data[data.Outlet_Size.isnull()]
data.groupby('Outlet_Type').get_group('Grocery Store')['Outlet_Location_Type'].value_counts()
data.groupby('Outlet_Type').get_group('Grocery Store')
data.groupby(['Outlet_Location_Type','Outlet_Type'])['Outlet_Size'].value_counts()
(data.Outlet_Identifier=='OUT010').value_counts()
data.groupby('Outlet_Size').Outlet_Identifier.value_counts()
def func(x):

    if x.Outlet_Identifier == 'OUT010' :

        x.Outlet_Size == 'High'

    elif x.Outlet_Identifier == 'OUT045' :

        x.Outlet_Size == 'Medium'

    elif x.Outlet_Identifier == 'OUT017' :

        x.Outlet_Size == 'Medium'

    elif x.Outlet_Identifier == 'OUT013' :

        x.Outlet_Size == 'High'

    elif x.Outlet_Identifier == 'OUT046' :

        x.Outlet_Size == 'Small'

    elif x.Outlet_Identifier == 'OUT035' :

        x.Outlet_Size == 'Small'

    elif x.Outlet_Identifier == 'OUT019' :

        x.Outlet_Size == 'Small'

    elif x.Outlet_Identifier == 'OUT027' :

        x.Outlet_Size == 'Medium'

    elif x.Outlet_Identifier == 'OUT049' :

        x.Outlet_Size == 'Medium'

    elif x.Outlet_Identifier == 'OUT018' :

        x.Outlet_Size == 'Medium'

    return(x)
data.Outlet_Size=data.apply(func,axis=1)
data.head()
sns.boxplot(x='Item_MRP',y='Item_Outlet_Sales',data=data)
data[data.Item_MRP=='b'].Item_Outlet_Sales.max()
data[data.Item_Outlet_Sales==7158.6816]
data=data.drop(index=7796)

data.groupby('Item_MRP').get_group('b')['Item_Outlet_Sales'].max()
sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',data=data)
sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',data=data)
data[data.Outlet_Location_Type=='Tier 1'].Item_Outlet_Sales.max()
data[data['Item_Outlet_Sales']==9779.9362]
data=data.drop(index=4289)
sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',data=data)
sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',data=data)
data.Outlet_Establishment_Year=data.Outlet_Establishment_Year.astype('category')

data_label=data.Item_Outlet_Sales

data_dummy=pd.get_dummies(data.iloc[:,0:6])
data_dummy['Item_Outlet_Sales']=data_label
data_dummy.shape
from sklearn.model_selection import train_test_split
train,test = train_test_split(data_dummy,test_size=0.20,random_state=2019)
train.shape , test.shape
train_label=train['Item_Outlet_Sales']

test_label=test['Item_Outlet_Sales']

del train['Item_Outlet_Sales']

del test['Item_Outlet_Sales']
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(train,train_label)
from sklearn.metrics import mean_squared_error
predict_lr=lr.predict(test)
mse=mean_squared_error(test_label,predict_lr)
lr_score=np.sqrt(mse)
lr_score
from sklearn.model_selection import cross_val_score
score=cross_val_score(lr,train,train_label,cv=10,scoring='neg_mean_squared_error')
lr_score_cross=np.sqrt(-score)
np.mean(lr_score_cross),np.std(lr_score_cross)
from sklearn.linear_model import Ridge

r=Ridge(alpha=0.05,solver='cholesky')

r.fit(train,train_label)

predict_r=r.predict(test)

mse=mean_squared_error(test_label,predict_r)

r_score=np.sqrt(mse)

r_score
r=Ridge(alpha=0.05,solver='cholesky')

score=cross_val_score(r,train,train_label,cv=10,scoring='neg_mean_squared_error')

r_score_cross=np.sqrt(-score)

np.mean(r_score_cross),np.std(r_score_cross)
from sklearn.linear_model import Lasso

l=Lasso(alpha=0.01)

l.fit(train,train_label)

predict_l=l.predict(test)

mse=mean_squared_error(test_label,predict_l)

l_score=np.sqrt(mse)

l_score
l=Lasso(alpha=0.01)

score=cross_val_score(l,train,train_label,cv=10,scoring='neg_mean_squared_error')

l_score_cross=np.sqrt(-score)

np.mean(l_score_cross),np.std(l_score_cross)
from sklearn.linear_model import ElasticNet

en=ElasticNet(alpha=0.01,l1_ratio=0.5)

en.fit(train,train_label)

predict_r=en.predict(test)

mse=mean_squared_error(test_label,predict_r)

en_score=np.sqrt(mse)

en_score
en=ElasticNet(alpha=0.01,l1_ratio=0.5)

score=cross_val_score(en,train,train_label,cv=10,scoring='neg_mean_squared_error')

en_score_cross=np.sqrt(-score)

np.mean(en_score_cross),np.std(en_score_cross)
from sklearn.linear_model import SGDRegressor

sgd=SGDRegressor(penalty='l2',n_iter=100,alpha=0.05)

sgd.fit(train,train_label)

predict_r=sgd.predict(test)

mse=mean_squared_error(test_label,predict_r)

sgd_score=np.sqrt(mse)

sgd_score
sgd=SGDRegressor(penalty='l2',n_iter=100,alpha=0.05)

score=cross_val_score(sgd,train,train_label,cv=10,scoring='neg_mean_squared_error')

sgd_score_cross=np.sqrt(-score)

np.mean(sgd_score_cross),np.std(sgd_score_cross)
from sklearn.svm import SVR

svm=SVR(epsilon=15,kernel='linear')

svm.fit(train,train_label)

predict_r=svm.predict(test)

mse=mean_squared_error(test_label,predict_r)

svm_score=np.sqrt(mse)

svm_score
svm=SVR(epsilon=15,kernel='linear')

score=cross_val_score(svm,train,train_label,cv=10,scoring='neg_mean_squared_error')

svm_score_cross=np.sqrt(-score)

np.mean(svm_score_cross),np.std(svm_score_cross)
from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor()

dtr.fit(train,train_label)

predict_r=dtr.predict(test)

mse=mean_squared_error(test_label,predict_r)

dtr_score=np.sqrt(mse)

dtr_score
dtr=DecisionTreeRegressor()

score=cross_val_score(dtr,train,train_label,cv=10,scoring='neg_mean_squared_error')

dtr_score_cross=np.sqrt(-score)

np.mean(dtr_score_cross),np.std(dtr_score_cross)
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()

rf.fit(train,train_label)

predict_r=rf.predict(test)

mse=mean_squared_error(test_label,predict_r)

rf_score=np.sqrt(mse)

rf_score
rf=RandomForestRegressor()

score=cross_val_score(rf,train,train_label,cv=10,scoring='neg_mean_squared_error')

rf_score_cross=np.sqrt(-score)

np.mean(rf_score_cross),np.std(rf_score_cross)
from sklearn.ensemble import BaggingRegressor
br=BaggingRegressor(max_samples=70)
br.fit(train,train_label)
score=br.predict(test)
br_score=mean_squared_error(test_label,score)
br_score=np.sqrt(br_score)

br_score
br=BaggingRegressor()

score=cross_val_score(br,train,train_label,cv=10,scoring='neg_mean_squared_error')

br_score_cross=np.sqrt(-score)

np.mean(br_score_cross),np.std(br_score_cross)
from sklearn.ensemble import AdaBoostRegressor

ada=AdaBoostRegressor()

ada.fit(train,train_label)

g=ada.predict(test)

ada_score=mean_squared_error(test_label,g)

ada_score=np.sqrt(ada_score)

ada_score
ada=AdaBoostRegressor()

score=cross_val_score(ada,train,train_label,cv=10,scoring='neg_mean_squared_error')

ada_score_cross=np.sqrt(-score)

np.mean(ada_score_cross),np.std(ada_score_cross)
from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor()

gbr.fit(train,train_label)

p=gbr.predict(test)

gb_score=mean_squared_error(test_label,p)

gb_score=np.sqrt(gb_score)

gb_score
gb=GradientBoostingRegressor()

score=cross_val_score(gb,train,train_label,cv=10,scoring='neg_mean_squared_error')

gb_score_cross=np.sqrt(-score)

np.mean(gb_score_cross),np.std(gb_score_cross)
name=['Linear Regression','Linear Regression CV','Ridge Regression','Ridge Regression CV','Lasso Regression',

     'Lasso Regression CV','Elastic Net Regression','Elastic Net Regression CV','SGD Regression','SGD Regression CV',

     'SVM','SVM CV','Decision Tree','Decision Tree Regression','Random Forest','Random Forest CV','Ada Boost','Ada Boost CV',

     'Bagging','Bagging CV','Gradient Boost','Gradient Boost CV']
go=pd.DataFrame({'RMSE':[lr_score,lr_score_cross,r_score,r_score_cross,l_score,l_score_cross,en_score,en_score_cross,

                     sgd_score,sgd_score_cross,svm_score,svm_score_cross,dtr_score,dtr_score_cross,rf_score,rf_score_cross,

                     ada_score,ada_score_cross,br_score,br_score_cross,gb_score,gb_score_cross]},index=name)
go['RMSE']=go.applymap(lambda x: x.mean())
go.RMSE.sort_values()
fig=plt.figure(figsize=(10,6))

plt.scatter(np.arange(1,100,10),predict_r[0:100:10],color='blue')

plt.scatter(np.arange(1,100,10),p[0:100:10],color='yellow')

plt.scatter(np.arange(1,100,10),test_label[0:100:10],color='black')

plt.legend(['Random_Forest','Gradient Boosting','Real Value'])

from sklearn.model_selection import GridSearchCV
gb=GradientBoostingRegressor(max_depth=7,n_estimators=200,learning_rate=0.01)

param=[{'min_samples_split':[5,9,13],'max_leaf_nodes':[3,5,7,9],'max_features':[8,10,15,18]}]

gs=GridSearchCV(gb,param,cv=5,scoring='neg_mean_squared_error')

gs.fit(train,train_label)
gs.best_estimator_
gb=gs.best_estimator_
total=pd.concat([train,test],axis=0,ignore_index=True)
total_label=pd.concat([train_label,test_label],axis=0,ignore_index=True)
total_label.shape,total.shape
gb.fit(total,total_label)
test=pd.read_csv('../input/Test.csv')
test.shape
attributes=['Item_MRP',

 'Outlet_Type',

 'Outlet_Size',

 'Outlet_Location_Type',

 'Outlet_Establishment_Year',

 'Outlet_Identifier',

 'Item_Type']
test=test[attributes]
test.shape
test.info()
test.Item_MRP=pd.cut(test.Item_MRP,bins=[25,75,140,205,270],labels=['a','b','c','d'],right=True)

test.Item_Type=test.Item_Type.astype('category')

test.Outlet_Size=test.Outlet_Size.astype('category')

test.Outlet_Identifier=test.Outlet_Identifier.astype('category')

test.Outlet_Establishment_Year=test.Outlet_Establishment_Year.astype('int64')

test.Outlet_Type=test.Outlet_Type.astype('category')

test.Outlet_Location_Type=test.Outlet_Location_Type.astype('category')
test.info()
test.Outlet_Establishment_Year=test.Outlet_Establishment_Year.astype('category')
test.info()
def func(x):

    if x.Outlet_Identifier == 'OUT010' :

        x.Outlet_Size == 'High'

    elif x.Outlet_Identifier == 'OUT045' :

        x.Outlet_Size == 'Medium'

    elif x.Outlet_Identifier == 'OUT017' :

        x.Outlet_Size == 'Medium'

    elif x.Outlet_Identifier == 'OUT013' :

        x.Outlet_Size == 'High'

    elif x.Outlet_Identifier == 'OUT046' :

        x.Outlet_Size == 'Small'

    elif x.Outlet_Identifier == 'OUT035' :

        x.Outlet_Size == 'Small'

    elif x.Outlet_Identifier == 'OUT019' :

        x.Outlet_Size == 'Small'

    elif x.Outlet_Identifier == 'OUT027' :

        x.Outlet_Size == 'Medium'

    elif x.Outlet_Identifier == 'OUT049' :

        x.Outlet_Size == 'Medium'

    elif x.Outlet_Identifier == 'OUT018' :

        x.Outlet_Size == 'Medium'

    return(x)
test.Outlet_Size=test.apply(func,axis=1)
test_dummy=pd.get_dummies(test.iloc[:,0:6])
test_dummy.head()
predict=gb.predict(test_dummy)
predict.shape
sample=pd.read_csv('../input/Submission.csv')
sample.head()
del sample['Item_Outlet_Sales']
df=pd.DataFrame({'Item_Outlet_Sales':predict})

corr_ans=pd.concat([sample,df],axis=1)

del corr_ans['Unnamed: 0']

corr_ans
corr_ans.to_csv('correct.csv',index=None)