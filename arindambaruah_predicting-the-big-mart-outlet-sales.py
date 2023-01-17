import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from scipy import stats

from scipy.stats import norm, skew
df_train=pd.read_csv('../input/big-mart-sales-prediction/Train.csv')

df_test=pd.read_csv('../input/big-mart-sales-prediction/Test.csv')
df_train.head()
df_train.dtypes
df_train.isna().any()
plt.figure(figsize=(10,8))

sns.heatmap(df_train.isna(),cmap='gnuplot')
df_train['Item_Weight'].describe()
plt.figure(figsize=(10,8))

sns.distplot(df_train['Item_Weight'].dropna(),color='green')

plt.title('Weight distribution of the items \n Median weight: {0:.2f}'.format(df_train['Item_Weight'].dropna().median()),size=25)

plt.axvline(df_train['Item_Weight'].dropna().median(),color='red',label='Median weight')

plt.legend()
fig1=plt.figure(figsize=(10,8))

ax1=fig1.add_subplot(121)

sns.boxplot(df_train['Item_Weight'],ax=ax1,orient='v',color='indianred')

ax1.set_title('Boxplot of weights',size=15)



ax2=fig1.add_subplot(122)

sns.violinplot(df_train['Item_Weight'],ax=ax2,orient='v',color='green')

ax2.set_title('Violinplot of weights',size=15)
df_train['Item_Fat_Content'].unique()
df_train['Item_Fat_Content']=df_train['Item_Fat_Content'].replace('low fat','Low Fat')

df_train['Item_Fat_Content']=df_train['Item_Fat_Content'].replace('LF','Low Fat')

df_train['Item_Fat_Content']=df_train['Item_Fat_Content'].replace('reg','Regular')

df_train['Item_Fat_Content'].unique()

df_train['Count']=1

df_fat=df_train.groupby('Item_Fat_Content')['Count'].sum().reset_index()



fig2=px.pie(df_fat,values='Count',names='Item_Fat_Content',hole=0.4)



fig2.update_layout(title='Fat content',title_x=0.48,

                  annotations=[dict(text='Fat',font_size=15, showarrow=False,height=800,width=900)])

fig2.update_traces(textfont_size=15,textinfo='percent+label')

fig2.show()
df_train['Item_Visibility'].describe()
plt.figure(figsize=(10,8))

sns.distplot(df_train['Item_Visibility'])

plt.title('Item visibility distribution \n Median:{0:.2f}'.format(df_train['Item_Visibility'].median()),size=25)

plt.axvline(df_train['Item_Visibility'].median(),color='black',label='Median')

plt.legend()
sns.set()

fig3=plt.figure(figsize=(10,5))

ax1=fig3.add_subplot(121)

sns.boxplot(df_train['Item_Visibility'],orient='v',ax=ax1,color='green')

ax2=fig3.add_subplot(122)

stats.probplot(df_train['Item_Visibility'],plot=ax2)
df_train[df_train['Item_Visibility']>0.2].shape[0]
df_train=df_train[df_train['Item_Visibility']<0.2]

stats.probplot(df_train['Item_Visibility'],plot=plt)
plt.figure(figsize=(10,8))

sns.distplot(df_train['Item_Visibility'],fit=norm,color='red')

plt.title('Distribution deviation from normal distribution',size=25)

plt.ylabel('Frequency',size=15)

plt.xlabel('Item visbility',size=15)

mu=df_train['Item_Visibility'].mean()

sigma=df_train['Item_Visibility'].std()

plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma=$ {1:.2f} )'.format(mu, sigma)])


df_type=df_train.groupby('Item_Type')['Count'].sum().reset_index()

fig4=px.sunburst(df_train,path=['Item_Type','Item_Fat_Content'],names='Item_Type',color_continuous_scale='RdBu')

fig4.update_layout(title='Item types',title_x=0.2,title_y=0.8,

                  annotations=[dict(showarrow=True,height=1000,width=900)],margin=dict(l=20, r=20, t=20, b=20))

fig4.show()



fig5=px.pie(df_type,values='Count',names='Item_Type')

fig5.update_layout(title='Item distribution',title_x=0.1,title_y=0.8)

fig5.update_traces(textfont_size=15,textinfo='percent')

fig5.show()
plt.figure(figsize=(15,10))

sns.distplot(df_train['Item_MRP'],color='orange')

plt.title('Item MRP distribution \n Median:{0:.2f} Rs'.format(df_train['Item_MRP'].median()),size=25)

plt.axvline(df_train['Item_MRP'].median(),color='black',label='Median')

plt.legend()
labels=df_train['Item_Type'].unique()

fig6=plt.figure(figsize=(10,10))

ax1=fig6.add_subplot(211)

sns.boxplot(x='Item_Type',y='Item_MRP',data=df_train,ax=ax1)

ax1.set_xticklabels(labels, rotation=75,size=9)



ax2=fig6.add_subplot(212)

sns.boxplot(x='Item_Fat_Content',y='Item_MRP',data=df_train,ax=ax2)



fig6.tight_layout(pad=3) #For spacing between subplots
df_outlets=df_train.groupby('Outlet_Identifier')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)
sns.catplot('Outlet_Identifier','Count',data=df_outlets,aspect=2,height=8,kind='bar',palette='gnuplot')

plt.xticks(size=15)

plt.ylabel('Number of items sold',size=15)

plt.xlabel('Outlet number',size=20)

plt.title('Outlet performance',size=25)

plt.yticks(np.arange(0,1100,100))
plt.figure(figsize=(10,8))

sns.boxplot('Outlet_Establishment_Year','Item_MRP',data=df_train)
plt.figure(figsize=(10,8))

sns.boxplot('Outlet_Establishment_Year','Item_Outlet_Sales',data=df_train)

plt.title('Outlet sales',size=25)
df_train['Outlet_Size'].isna().value_counts()
df_size=df_train.groupby('Outlet_Size')['Count'].sum().reset_index()

fig7=px.pie(df_size,values='Count',names='Outlet_Size',hole=0.4)

fig7.update_layout(title='Store sizes',title_x=0.5,annotations=[dict(text='Fat',font_size=15, showarrow=False,height=800,width=900)])

fig7.update_traces(textfont_size=15,textinfo='percent+label')

fig7.show()
df_size_sales=df_train.groupby('Outlet_Size')[['Item_MRP','Item_Outlet_Sales']].mean().reset_index()
df_size_sales
fig8=plt.figure(figsize=(15,10))

ax1=fig8.add_subplot(121)

sns.barplot('Outlet_Size','Item_MRP',data=df_size_sales,ax=ax1)



ax2=fig8.add_subplot(122)

sns.barplot('Outlet_Size','Item_Outlet_Sales',data=df_size_sales,ax=ax2,palette='rocket')



ax1.set_title('Average price of items sold',size=20)

ax2.set_title('Average sales of store',size=20)



df_train.head()
fig9=px.sunburst(df_train,path=['Outlet_Type','Outlet_Location_Type'],color_continuous_scale='RdBu')

fig9.update_layout(title='Store type with location type',title_x=0.5)

fig9.show()
plt.figure(figsize=(10,8))

sns.boxplot(y='Item_Outlet_Sales',hue='Outlet_Type',x='Outlet_Location_Type',data=df_train,palette='terrain')
df_train.drop('Count',axis=1,inplace=True)
corrs=df_train.dropna().corr()

plt.figure(figsize=(10,8))

sns.heatmap(corrs,annot=True,fmt='.2%')
unn_cols=['Item_Weight','Outlet_Size','Item_Identifier','Outlet_Identifier']



for cols in unn_cols:

    df_train.drop(cols,axis=1,inplace=True)
df_train['Item_Fat_Content'].replace('Low Fat',1,inplace=True)

df_train['Item_Fat_Content'].replace('Regular',0,inplace=True)
df_dummies_type=pd.get_dummies(df_train['Item_Type'])
df_train=df_train.merge(df_dummies_type,on=df_train.index)
df_train.drop('key_0',axis=1,inplace=True)

df_train.drop('Item_Type',axis=1,inplace=True)

df_train['Outlet_Location_Type'].replace('Tier 1',1,inplace=True)

df_train['Outlet_Location_Type'].replace('Tier 2',2,inplace=True)

df_train['Outlet_Location_Type'].replace('Tier 3',3,inplace=True)
df_dummies_outlet=pd.get_dummies(df_train['Outlet_Type'])

df_train=df_train.merge(df_dummies_outlet,on=df_train.index)
df_train.drop('key_0',axis=1,inplace=True)

df_train.drop('Outlet_Type',axis=1,inplace=True)
targets=df_train['Item_Outlet_Sales']

df_train.drop('Item_Outlet_Sales',axis=1,inplace=True)

df_train.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
X_train,X_test,y_train,y_test=train_test_split(df_train,targets,shuffle=True,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
reg_lin=LinearRegression()

reg_lin.fit(X_train,y_train)
reg_lin.score(X_train,y_train)
y_preds_lin=reg_lin.predict(X_test)
rmse_lin=np.sqrt(mean_squared_error(y_preds_lin,y_test))

print('RMSE for Linear Regression:{0:.2f}'.format(rmse_lin))
reg_lin_df=pd.DataFrame()

reg_lin_df['Target']=y_test

reg_lin_df['Predictions']=y_preds_lin



sns.lmplot('Target','Predictions',data=reg_lin_df,height=6,aspect=2,line_kws={'color':'black'},scatter_kws={'alpha':0.4})

plt.title('Linear Regression \n RMSE: {0:.2f}'.format(rmse_lin),size=25)
from sklearn.linear_model import RidgeCV
reg_rid=RidgeCV(cv=10)

reg_rid.fit(X_train,y_train)
reg_rid.score(X_train,y_train)
y_preds_rid=reg_rid.predict(X_test)

rmse_rid=np.sqrt(mean_squared_error(y_preds_rid,y_test))

print('RMSE for Ridge Regression:{0:.2f}'.format(rmse_rid))
reg_rid_df=pd.DataFrame()

reg_rid_df['Target']=y_test

reg_rid_df['Predictions']=y_preds_rid



sns.lmplot('Target','Predictions',data=reg_rid_df,height=6,aspect=2,line_kws={'color':'orange'},scatter_kws={'alpha':0.4,'color':'green'})

plt.title('Ridge Regression \n RMSE: {0:.2f}'.format(rmse_rid),size=25)
from sklearn.linear_model import Lasso
reg_las=Lasso()

reg_las.fit(X_train,y_train)
reg_las.score(X_train,y_train)
y_preds_las=reg_las.predict(X_test)

rmse_las=np.sqrt(mean_squared_error(y_preds_las,y_test))

print('RMSE for Lasso Regression:{0:.2f}'.format(rmse_las))
reg_las_df=pd.DataFrame()

reg_las_df['Target']=y_test

reg_las_df['Predictions']=y_preds_las



sns.lmplot('Target','Predictions',data=reg_las_df,height=6,aspect=2,line_kws={'color':'blue'},scatter_kws={'alpha':0.4,'color':'red'})

plt.title('Lasso Regression \n RMSE: {0:.2f}'.format(rmse_las),size=25)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
rfr=RandomForestRegressor(random_state=0)

param_grid={'n_estimators':[3,4,5,7,9,10,12], 'max_depth':[5,7,9,10,12]}

grid=GridSearchCV(rfr,param_grid,scoring='r2',cv=10)
grid_result=grid.fit(X_train,y_train)

grid_result.best_params_
grid_result.score(X_train,y_train)
y_preds_rfr=grid_result.predict(X_test)

rmse_rfr=np.sqrt(mean_squared_error(y_preds_rfr,y_test))

print('RMSE for Random Forest Regression:{0:.2f}'.format(rmse_rfr))
rfr_df=pd.DataFrame()

rfr_df['Target']=y_test

rfr_df['Predictions']=y_preds_rfr



sns.lmplot('Target','Predictions',data=rfr_df,height=6,aspect=2,line_kws={'color':'green'},scatter_kws={'alpha':0.4,'color':'blue'})

plt.title('Random Forest Regression \n RMSE: {0:.2f}'.format(rmse_rfr),size=25)
from sklearn.ensemble import GradientBoostingRegressor
gbdt=GradientBoostingRegressor(random_state=0)

gbdt.fit(X_train,y_train)

gbdt.score(X_train,y_train)
y_preds_gbdt=gbdt.predict(X_test)

rmse_gbdt=np.sqrt(mean_squared_error(y_preds_gbdt,y_test))

print('RMSE for Random Forest Regression:{0:.2f}'.format(rmse_gbdt))
gbdt_df=pd.DataFrame()

gbdt_df['Target']=y_test

gbdt_df['Predictions']=y_preds_gbdt



sns.lmplot('Target','Predictions',data=gbdt_df,height=6,aspect=2,line_kws={'color':'red'},scatter_kws={'alpha':0.4,'color':'black'})

plt.title('GBDT Regression \n RMSE: {0:.2f}'.format(rmse_gbdt),size=25)
from sklearn.ensemble import AdaBoostRegressor
ada=AdaBoostRegressor(random_state=0)

ada.fit(X_train,y_train)
y_preds_ada=ada.predict(X_test)

rmse_ada=np.sqrt(mean_squared_error(y_preds_ada,y_test))

print('RMSE for AdaBoost Regression:{0:.2f}'.format(rmse_ada))
ada_df=pd.DataFrame()

ada_df['Target']=y_test

ada_df['Predictions']=y_preds_ada



sns.lmplot('Target','Predictions',data=ada_df,height=6,aspect=2,line_kws={'color':'blue'},scatter_kws={'alpha':0.4,'color':'black'})

plt.title('AdaBoost Regression \n RMSE: {0:.2f}'.format(rmse_ada),size=25)
import xgboost as xgb
xgb_reg=xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
xgb_reg.fit(X_train,y_train)

xgb_reg.score(X_train,y_train)
y_preds_xgb=xgb_reg.predict(X_test)

rmse_xgb=np.sqrt(mean_squared_error(y_preds_xgb,y_test))

print('RMSE for XGBoost Regression:{0:.2f}'.format(rmse_xgb))
ada_df=pd.DataFrame()

ada_df['Target']=y_test

ada_df['Predictions']=y_preds_xgb



sns.lmplot('Target','Predictions',data=ada_df,height=6,aspect=2,line_kws={'color':'green'},scatter_kws={'alpha':0.4,'color':'black'})

plt.title('XGBoost Regression \n RMSE: {0:.2f}'.format(rmse_xgb),size=25)
from lightgbm import LGBMRegressor
lgb=LGBMRegressor(random_state=0)

lgb.fit(X_train,y_train)

lgb.score(X_train,y_train)
y_preds_lgb=lgb.predict(X_test)

rmse_lgb=np.sqrt(mean_squared_error(y_preds_lgb,y_test))

print('RMSE for LGBM Regression:{0:.2f}'.format(rmse_lgb))
lgb_df=pd.DataFrame()

lgb_df['Target']=y_test

lgb_df['Predictions']=y_preds_lgb



sns.lmplot('Target','Predictions',data=ada_df,height=6,aspect=2,line_kws={'color':'green'},scatter_kws={'alpha':0.4,'color':'red'})

plt.title('LGBM Regression \n RMSE: {0:.2f}'.format(rmse_lgb),size=25)
df_identifiers=pd.DataFrame(df_test['Item_Identifier'])

df_identifiers['Outlet_Identifier']=df_test['Outlet_Identifier']


unn_cols=['Item_Weight','Outlet_Size','Item_Identifier','Outlet_Identifier']



for cols in unn_cols:

    df_test.drop(cols,axis=1,inplace=True)
df_test.head()
df_train.head()
df_test['Item_Fat_Content']=df_test['Item_Fat_Content'].replace('low fat','Low Fat')

df_test['Item_Fat_Content']=df_test['Item_Fat_Content'].replace('LF','Low Fat')

df_test['Item_Fat_Content']=df_test['Item_Fat_Content'].replace('reg','Regular')



df_test['Item_Fat_Content'].replace('Low Fat',1,inplace=True)

df_test['Item_Fat_Content'].replace('Regular',0,inplace=True)
df_dummies_type=pd.get_dummies(df_test['Item_Type'])

df_test=df_test.merge(df_dummies_type,on=df_test.index)

df_test.drop('key_0',axis=1,inplace=True)

df_test.drop('Item_Type',axis=1,inplace=True)                   

                    
df_test['Outlet_Location_Type'].replace('Tier 1',1,inplace=True)

df_test['Outlet_Location_Type'].replace('Tier 2',2,inplace=True)

df_test['Outlet_Location_Type'].replace('Tier 3',3,inplace=True)
df_dummies_outlet=pd.get_dummies(df_test['Outlet_Type'])

df_test=df_test.merge(df_dummies_outlet,on=df_test.index)
df_test.drop('key_0',axis=1,inplace=True)

df_test.drop('Outlet_Type',axis=1,inplace=True)
y_preds_rfr=grid_result.predict(df_test)

df_rfr_submission=df_identifiers

df_rfr_submission['Item_Outlet_Sales']=y_preds_rfr

df_rfr_submission.head()
df_rfr_submission.to_csv('RFR_submission.csv',index=False)