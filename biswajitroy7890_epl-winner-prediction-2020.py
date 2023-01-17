#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


# In[5]:


pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',1400000)
np.set_printoptions(suppress=True)


# In[6]:


dframe =pd.read_csv('C:/Users/user/Desktop/IVY WORK BOOK/PYTHON/Python Datasets/Regression Datasets/epl2020.csv')


# In[7]:


dframe


# In[8]:


dframe=dframe.drop(labels=['Unnamed: 0','matchDay'], axis=1)


# In[9]:


dframe.info()


# In[10]:


dframe.isnull().sum().any()


# In[11]:


dframe.nunique()


# In[10]:


dframe.hist(figsize=(150,200))


# In[11]:


dframe.plot.scatter(x='npxGD',y='tot_points',figsize=(15,10))


# In[12]:


dframe=dframe.drop(labels=['result','date','matchtime','pts'], axis=1)


# In[13]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dframe['h_a']=le.fit_transform(dframe['h_a'])
dframe['teamId']=le.fit_transform(dframe['teamId'])


# In[61]:


len(le.classes_)


# In[14]:


dframe.head()


# In[15]:


dframe.groupby('teamId')['tot_points'].sum().plot(kind='bar',figsize=(15,10))


# In[16]:


dframe.groupby('teamId')['tot_points'].sum().sort_values(ascending=False)


# In[17]:


Arr=np.array(dframe)


# In[18]:


mu = np.mean(Arr)
sigma = np.std(Arr)
print(mu)
print(sigma)
x =np.random.normal(mu, sigma, size=200)
fig, ax = plt.subplots()
ax.hist(x, 20)
ax.set_title('Historgram')
ax.set_xlabel('bin range')
ax.set_ylabel('frequency')


fig.tight_layout()
plt.show()


# In[19]:


dframe['tot_points'].values


# In[20]:


dframe.nunique()


# In[21]:


dframe.head(5)


# In[22]:


dframe['npxGA'].isin(dframe['npxG']).all()


# In[33]:


cat_cols=['h_a','deep','deep_allowed','scored','missed','wins','draws','loses','teamId','round','HS.x',
           'HST.x','HF.x','HC.x','HY.x','HR.x','AS.x','AST.x','AF.x','AC.x','AY.x','AR.x']

Target=['tot_points']

cont_cols=['xG','xGA','npxG','npxGA','xpts','npxGD','ppda_cal','allowed_ppda','tot_goal','tot_con','B365H.x',
            'B365D.x','B365A.x','HtrgPerc','AtrgPerc']


# In[23]:


dframe.shape


# In[24]:


variables=dframe.columns


# In[25]:


from sklearn.covariance import EllipticEnvelope


# In[26]:


X=dframe[variables].values
elp=EllipticEnvelope(random_state=10)
outlier_X=elp.fit_predict(X)


# In[27]:


dframe['outliers']=outlier_X


# In[28]:


dframe.groupby('outliers').size().value_counts()


# In[29]:


con=dframe['outliers']==-1
delete=dframe[con].index
dframe=dframe.drop(delete)


# In[30]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[34]:


y_train=dframe[Target]
X_train=dframe.drop(labels='tot_points', axis=1)


# In[35]:


X_train.head()


# In[36]:


y_train.head()


# In[37]:


feature_sel_model = SelectFromModel(Lasso(alpha=0.05, random_state=5)) 
feature_sel_model.fit(X_train, y_train)


# In[38]:


selected_feat = X_train.columns[feature_sel_model.get_support()]


# In[39]:


print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))


# In[40]:


selected_feat


# In[41]:


X=X_train[selected_feat].values
y=y_train[Target].values


# In[42]:


print(X[0:10])
print(y[0:10])


# In[43]:


from sklearn.preprocessing import StandardScaler

Predictorscalar=StandardScaler()
Targetscalar=StandardScaler()
X=Predictorscalar.fit_transform(X)
y=Targetscalar.fit_transform(y)


# In[52]:


dframe2=pd.DataFrame(X,columns=selected_feat)
dframe2['target']=y


# In[53]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics


# In[54]:


kf = KFold(n_splits=8)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[55]:


metrics.SCORERS.keys()


# In[48]:


dframe2=pd.to_pickle(dframe2,'C:/Users/user/Desktop/IVY WORK BOOK/PYTHON/pickle files/dframe2.pkl')


# In[57]:


from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[59]:


from catboost import CatBoostRegressor


# In[60]:


params={
    "iterations": [100,200,300],
    "learning_rate" : [0.05,0.01,0.007,0.02],
    "loss_function":['MAE','RMSE'],
    "bagging_temperature":[8,10,5],
     "max_depth":[2,3,5]
     }


# In[61]:


clf=CatBoostRegressor()


# In[62]:


Grid_search=GridSearchCV(clf,param_grid=params,scoring='neg_mean_absolute_error',n_jobs=-1,cv=7,verbose=3)


# In[63]:


predictmodel=Grid_search.fit(X_train, y_train)
predictions=predictmodel.predict(X_test)
print("R2 score",metrics.r2_score(y_test,predictions))
error=metrics.mean_squared_error(y_test,predictions)
print("mean Absolute error",metrics.mean_absolute_error(y_test,predictions))
print("Accuracy",(100-error))


# In[65]:


cross_val=cross_val_score(Grid_search,X,y,cv=2)
cross_val.mean()
print(cross_val)


# In[74]:


plt.hist(y_test,bins=10,align='right')


# In[75]:


plt.hist(predictions,bins=10,align='right')


# In[95]:


plt.scatter(y_test,predictions,c=predictions,s=50)
plt.show()
plt.tight_layout()


# In[1]:


from pytorch_tabnet.tab_model import TabNetRegressor


# In[60]:


dframe2.nunique()


# In[162]:


dframe2.info()


# In[292]:


if "Set" not in dframe2.columns:
    dframe2["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(dframe2.shape[0],))

train_indices = dframe2[dframe2.Set=="train"].index
valid_indices = dframe2[dframe2.Set=="valid"].index
test_indices = dframe2[dframe2.Set=="test"].index    
    


# In[293]:


print(train_indices)
print(valid_indices)
print(test_indices)


# In[294]:


categorical_dims={}


# In[295]:


unused_feat='set'
TARGET='target'

features = [ col for col in dframe2.columns if col not in unused_feat]

cat_idxs = [ i for i, f in enumerate(features) if f in cat_cols]


cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in cat_cols]


# In[296]:


cat_cols=[['teamId']]
cat_dims=[20]
cat_idxs=['teamId']


# In[297]:


Tabnetmodel=TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim)


# In[298]:


feautres=['npxGA', 'deep', 'deep_allowed', 'scored', 'missed', 'wins', 'loses',
       'teamId', 'ppda_cal', 'allowed_ppda', 'round', 'tot_goal', 'tot_con',
       'HS.x', 'HST.x', 'HF.x', 'HY.x', 'AS.x', 'AST.x', 'AC.x', 'AY.x',
       'B365H.x', 'B365A.x']


# In[299]:


X_train = dframe2[feautres].values[train_indices]
y_train = dframe2[TARGET].values[train_indices].reshape(-1,1)

X_valid = dframe2[feautres].values[valid_indices]
y_valid = dframe2[TARGET].values[valid_indices].reshape(-1,1)

X_test = dframe2[feautres].values[test_indices]
y_test = dframe2[TARGET].values[test_indices].reshape(-1,1)


# In[300]:


print(X_train)
print(y_train)
print(X_valid)
print(y_valid)
print(X_test)
print(y_test)


# In[301]:


predictmodel=Tabnetmodel.fit(X_train, y_train,X_valid, y_valid,max_epochs=500,patience=50,batch_size=1024,virtual_batch_size=128,
num_workers=0,drop_last=False)


# In[302]:


predictions=Tabnetmodel.predict(X_test)


# In[303]:


Tabnetmodel.feature_importances_


# In[304]:


Tabnetmodel.history['valid']['loss']


# In[305]:


print('Best Valid score',Tabnetmodel.best_cost)


# In[306]:


print("Accuracy",100-metrics.mean_absolute_error(predictions,y_test))


# In[307]:


X=Predictorscalar.inverse_transform(X_test).astype('float64')


# In[308]:


EPL=pd.DataFrame(X, columns=feautres)


# In[310]:


Y=Targetscalar.inverse_transform(y_test)


# In[311]:


EPL['Total_Point']=Y


# In[312]:


preds=Targetscalar.inverse_transform(predictions)


# In[313]:


EPL['predicted_Point']=preds


# In[317]:


EPL.groupby('teamId')['predicted_Point'].sum().plot(kind='bar',figsize=(15,10))


# In[318]:


EPL.groupby('teamId')['predicted_Point'].sum().sort_values(ascending=False)


# In[319]:


dframe.groupby('teamId')['tot_points'].sum().sort_values(ascending=False)


# In[324]:


EPL['Diiferance']=EPL['Total_Point']-EPL['predicted_Point']
EPL['Diiferance'].hist(figsize=(15,10))


# In[ ]: