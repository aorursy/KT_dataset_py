import warnings
warnings.filterwarnings('ignore')

import joblib
import xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline


from math import sqrt
from numpy import concatenate

from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import os
print()
df1 = pd.read_csv('../input/demographics.csv')
df1['Year'] = '2017'
print(len(df1))
df1.head(2)
df2 = pd.read_csv('../input/event_calendar.csv')
print(len(df2))
df2['YearMonth'] = df2['YearMonth'] .astype(str)
df2['YearMonth'] = df2['YearMonth'].apply(lambda x: x[0:4]+'-'+x[4:6])
df2['YearMonth']  = pd.to_datetime(df2['YearMonth'])
df2.head(2)
df3 = pd.read_csv('../input/historical_volume.csv')
print(len(df3))
df3['YearMonth'] = df3['YearMonth'] .astype(str)
df3['YearMonth'] = df3['YearMonth'].apply(lambda x: x[0:4]+'-'+x[4:6])
df3['YearMonth']  = pd.to_datetime(df3['YearMonth'])
df3 = df3.sort_values('YearMonth', ascending=True).reset_index(drop=True)
df3.head(2)
df4 = pd.read_csv('../input/industry_soda_sales.csv')
print(len(df4))
df4['YearMonth'] = df4['YearMonth'] .astype(str)
df4['YearMonth'] = df4['YearMonth'].apply(lambda x: x[0:4]+'-'+x[4:6])
df4['YearMonth']  = pd.to_datetime(df4['YearMonth'])
df4.head(2)
df5 = pd.read_csv('../input/industry_volume.csv')
print(len(df5))
df5['YearMonth'] = df5['YearMonth'] .astype(str)
df5['YearMonth'] = df5['YearMonth'].apply(lambda x: x[0:4]+'-'+x[4:6])
df5['YearMonth']  = pd.to_datetime(df5['YearMonth'])
df5.head(2)
df6 = pd.read_csv('../input/price_sales_promotion.csv')
print(len(df6))
df6['YearMonth'] = df6['YearMonth'] .astype(str)
df6['YearMonth'] = df6['YearMonth'].apply(lambda x: x[0:4]+'-'+x[4:6])
df6['YearMonth']  = pd.to_datetime(df6['YearMonth'])
df6 = df6.sort_values('YearMonth', ascending=True).reset_index(drop=True)
df6.head(2)
df7 = pd.read_csv('../input/weather.csv')
print(len(df7))
df7['YearMonth'] = df7['YearMonth'] .astype(str)
df7['YearMonth'] = df7['YearMonth'].apply(lambda x: x[0:4]+'-'+x[4:6])
df7['YearMonth']  = pd.to_datetime(df7['YearMonth'])
df7.head(2)
dfa = df4.merge(df5, on='YearMonth', how='inner')
dfa = dfa.merge(df2, on='YearMonth', how='inner')
print(len(dfa))
dfa.head(2)
df = df3.merge(df6, on=['Agency','SKU','YearMonth'], how='inner')
df = df.merge(df7, on=['Agency','YearMonth'], how='inner')

df['Year'] = df['YearMonth'].dt.year
df['Year'] = df['Year'].astype(str)

print(len(df))
df = df.merge(df1, on=['Agency','Year'], how ='left')

print(len(df))
df['SKU'] = df['SKU'].apply(lambda x: x[4:])
df['SKU'] = df['SKU'].astype(int)

df['Agency'] = df['Agency'].apply(lambda x: x[7:])
df['Agency'] = df['Agency'].astype(int)

df = df.merge(dfa, on='YearMonth', how='left')
df = df.drop('Year', axis=1)
print(len(df))

df.to_csv('../input/train.csv',index=False)
df.head(2)
# df7['Agency'] = df7['Agency'].apply(lambda x: x[7:])
# df7['Agency'] = df7['Agency'].astype(int)
print(len(df7))
df7.head(3)
df7 = df7.groupby('Agency')['Avg_Max_Temp'].agg(['mean','median']).reset_index()
print(len(df7))
df7.head(3)
agen6 = df7[(df7['mean']>=29.007939)&(df7['mean'] <29.007940)].reset_index(drop=True)
agen6
agen14 = df7[(df7['mean']>= 25.085280)&(df7['mean'] <25.085282)].reset_index(drop=True)
agen14
df1 = df1.drop('Year',axis=1)
df1 = df1.sort_values('Agency',ascending=True).reset_index(drop=True)
df1.head(2)
agen_614 = df1[(df1['Agency']=='Agency_06')|(df1['Agency']=='Agency_14')]
agen_614
df1_temp = df1[(df1['Avg_Population_2017'] >= 1800000)]
df1_temp = df1_temp[(df1_temp['Avg_Population_2017'] < 2500000)]
df1_temp.head(2)
df1_temp = df1_temp[df1_temp['Avg_Yearly_Household_Income_2017']> 185000]
df1_temp = df1_temp[df1_temp['Agency'] != 'Agency_06']
df1_temp = df1_temp[df1_temp['Agency'] != 'Agency_14'].reset_index(drop=True)
df1_temp.head(2)
fig, ax = plt.subplots(figsize=(15,10), sharex=True, sharey=True)
g = sns.scatterplot(x='Avg_Population_2017',y='Avg_Yearly_Household_Income_2017', hue='Agency', data=df1, ax=ax)
g = sns.scatterplot(x='Avg_Population_2017',y='Avg_Yearly_Household_Income_2017', marker='X', s=400 , data=agen_614, ax=ax)
g = sns.scatterplot(x='Avg_Population_2017',y='Avg_Yearly_Household_Income_2017', marker='X', s=150 , data=df1_temp, ax=ax)


ax.legend( df1['Agency'].values, loc='upper lef', ncol=2, borderaxespad=0,frameon=False, bbox_to_anchor= (1.01, 1.0))
plt.savefig('../fig/sku reco analysis 1.png',bbox_inches='tight')
agen6 = df3[df3['Agency'].isin(['Agency_55','Agency_60','Agency_5','Agency_40','Agency_8','Agency_50'])]
agen6 = agen6.sort_values('Agency', ascending=True).reset_index(drop=True)
agen6 = agen6.drop('YearMonth',axis=1)
print(len(agen6))
agen6.head()
agen6 = agen6.groupby(['Agency','SKU'])['Volume'].agg(['mean','median']).reset_index()
agen6 = agen6.drop('Agency',axis=1)
agen6.head()
# agen6['SKU'].value_counts()
agen6 = agen6.groupby('SKU')['mean','median'].mean().reset_index()
agen6 = agen6.sort_values('mean', ascending=False).reset_index(drop=True)
agen6.head(6)
dagen14 = df3[df3['Agency'].isin(['Agency_57','Agency_56','Agency_12','Agency_13','Agency_15',
                                    'Agency_16','Agency_17','Agency_20','Agency_38','Agency_39',
                                    'Agency_58','Agency_59','Agency_60'])]
print(len(dagen14))
dagen14.head()
dagen14 = dagen14.sort_values('Agency', ascending=True).reset_index(drop=True)
dagen14 = dagen14.drop('YearMonth',axis=1)
dagen14.head()
dagen14 = dagen14.groupby(['Agency','SKU'])['Volume'].agg(['mean','median']).reset_index()
dagen14.head()
dagen14 = dagen14.drop('Agency',axis=1)
dagen14.head()
# dagen14['SKU'].value_counts()
dagen14 = dagen14.groupby('SKU')['mean','median'].mean().reset_index()
dagen14 = dagen14.sort_values('mean', ascending=False).reset_index(drop=True)
dagen14.head(6)
df = pd.read_csv('../input/train.csv')
df['Volume'] = df['Volume'].round(2)
df['Price'] = df['Price'].round(2)
df['Sales'] = df['Sales'].round(2)
df['Promotions'] = df['Promotions'].round(2)
df['Avg_Max_Temp'] = df['Avg_Max_Temp'].round(2)
df.head()
df = df[['Agency','SKU','Volume']].copy()
df.head()
df.isnull().sum()
X = df.iloc[:,0:2]
X.head(2)
y = df.iloc[:,2:3]
y.head(2)
sc_x = StandardScaler()

X = sc_x.fit_transform(X.astype(float))
y = y.values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
arr_gb_val_r2 = []
arr_gb_val_mse = []

arr_gb_test_r2 = []
arr_gb_test_mse = []

for train, test in kfold.split(X_train):
    clf_gb = GradientBoostingRegressor()
    clf_gb.fit(X[train],y[train])

    Y_pred_val = clf_gb.predict(X[test])
    val_mse_gb = mean_squared_error(y[test],Y_pred_val)
    val_r2_gb = r2_score(y[test], Y_pred_val) 

    print('r2 Val : ',val_r2_gb)
    print('mse Val : ',val_mse_gb)
    arr_gb_val_r2.append(val_r2_gb)
    arr_gb_val_mse.append(val_mse_gb)
    
    Y_pred_test = clf_gb.predict(X_test)
    test_mse_gb = mean_squared_error(Y_test,Y_pred_test)
    test_r2_gb = r2_score(Y_test, Y_pred_test) 

    print('r2 Test : ',test_r2_gb)
    print('mse Val : ',test_mse_gb)
    arr_gb_test_r2.append(test_r2_gb)
    arr_gb_test_mse.append(test_mse_gb)


print(np.mean(arr_gb_val_r2))
print(np.mean(arr_gb_val_mse))
print(np.mean(arr_gb_test_r2))
print(np.mean(arr_gb_test_mse))
arr_rf_val_r2 = []
arr_rf_val_mse = []

arr_rf_test_r2 = []
arr_rf_test_mse = []

for train, test in kfold.split(X_train):
    clf_rf = RandomForestRegressor(n_estimators=500, random_state=0,max_depth=2)
    clf_rf.fit(X[train],y[train])

    Y_pred_val = clf_rf.predict(X[test])
    val_mse_rf = mean_squared_error(y[test],Y_pred_val)
    val_r2_rf = r2_score(y[test], Y_pred_val) 

    print('r2 Val : ',val_r2_rf)
    print('mse Val : ',val_mse_rf)
    arr_rf_val_r2.append(val_r2_rf)
    arr_rf_val_mse.append(val_mse_rf)
    
    Y_pred_test = clf_rf.predict(X_test)
    test_mse_rf = mean_squared_error(Y_test,Y_pred_test)
    test_r2_rf = r2_score(Y_test, Y_pred_test) 

    print('r2 Test : ',test_r2_rf)
    print('mse Val : ',test_mse_rf)
    arr_rf_test_r2.append(test_r2_rf)
    arr_rf_test_mse.append(test_mse_rf)


print(np.mean(arr_rf_val_r2))
print(np.mean(arr_rf_val_mse))
print(np.mean(arr_rf_test_r2))
print(np.mean(arr_rf_test_mse))
arr_svm_val_r2 = []
arr_svm_val_mse = []

arr_svm_test_r2 = []
arr_svm_test_mse = []

for train, test in kfold.split(X_train):
    clf_svm = SVR(kernel='rbf', C=1e3, gamma=0.1)
    clf_svm.fit(X[train],y[train])

    Y_pred_val = clf_svm.predict(X[test])
    val_mse_svm = mean_squared_error(y[test],Y_pred_val)
    val_r2_svm = r2_score(y[test], Y_pred_val) 

    print('r2 Val : ',val_r2_svm)
    print('mse Val : ',val_mse_svm)
    arr_svm_val_r2.append(val_r2_svm)
    arr_svm_val_mse.append(val_mse_svm)
    
    Y_pred_test = clf_svm.predict(X_test)
    test_mse_svm = mean_squared_error(Y_test,Y_pred_test)
    test_r2_svm = r2_score(Y_test, Y_pred_test) 

    print('r2 Test : ',test_r2_svm)
    print('mse Val : ',test_mse_svm)
    arr_svm_test_r2.append(test_r2_svm)
    arr_svm_test_mse.append(test_mse_svm)


print(np.mean(arr_svm_val_r2))
print(np.mean(arr_svm_val_mse))
print(np.mean(arr_svm_test_r2))
print(np.mean(arr_svm_test_mse))
arr_knn_val_r2 = []
arr_knn_val_mse = []

arr_knn_test_r2 = []
arr_knn_test_mse = []

for train, test in kfold.split(X_train):
    clf_knn = KNeighborsRegressor(n_neighbors=2)
    clf_knn.fit(X[train],y[train])

    Y_pred_val = clf_knn.predict(X[test])
    val_mse_knn = mean_squared_error(y[test],Y_pred_val)
    val_r2_knn = r2_score(y[test], Y_pred_val) 

    print('r2 Val : ',val_r2_knn)
    print('mse Val : ',val_mse_knn)
    arr_knn_val_r2.append(val_r2_knn)
    arr_knn_val_mse.append(val_mse_knn)
    
    Y_pred_test = clf_knn.predict(X_test)
    test_mse_knn = mean_squared_error(Y_test,Y_pred_test)
    test_r2_knn = r2_score(Y_test, Y_pred_test) 

    print('r2 Test : ',test_r2_knn)
    print('mse Val : ',test_mse_knn)
    arr_knn_test_r2.append(test_r2_knn)
    arr_knn_test_mse.append(test_mse_knn)


print(np.mean(arr_knn_val_r2))
print(np.mean(arr_knn_val_mse))
print(np.mean(arr_knn_test_r2))
print(np.mean(arr_knn_test_mse))
arr_xgb_val_r2 = []
arr_xgb_val_mse = []

arr_xgb_test_r2 = []
arr_xgb_test_mse = []

i = 1
for train, test in kfold.split(X_train):
    clf_xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)
    clf_xgb.fit(X[train],y[train])
    #save model
    joblib.dump(clf_xgb, '../model/xgb_'+str(i)+'.dat') 
    i = i+1
    
    Y_pred_val = clf_xgb.predict(X[test])
    val_mse_xgb = mean_squared_error(y[test],Y_pred_val)
    val_r2_xgb = r2_score(y[test], Y_pred_val) 

    print('r2 Val : ',val_r2_xgb)
    print('mse Val : ',val_mse_xgb)
    arr_xgb_val_r2.append(val_r2_xgb)
    arr_xgb_val_mse.append(val_mse_xgb)
    
    Y_pred_test = clf_xgb.predict(X_test)
    test_mse_xgb = mean_squared_error(Y_test,Y_pred_test)
    test_r2_xgb = r2_score(Y_test, Y_pred_test) 

    print('r2 Test : ',test_r2_xgb)
    print('mse Val : ',test_mse_xgb)
    arr_xgb_test_r2.append(test_r2_xgb)
    arr_xgb_test_mse.append(test_mse_xgb)


print(np.mean(arr_xgb_val_r2))
print(np.mean(arr_xgb_val_mse))
print(np.mean(arr_xgb_test_r2))
print(np.mean(arr_xgb_test_mse))
arr_lr_val_r2 = []
arr_lr_val_mse = []

arr_lr_test_r2 = []
arr_lr_test_mse = []

for train, test in kfold.split(X_train):
    clf_lr = LinearRegression()
    clf_lr.fit(X[train],y[train])

    Y_pred_val = clf_lr.predict(X[test])
    val_mse_lr = mean_squared_error(y[test],Y_pred_val)
    val_r2_lr = r2_score(y[test], Y_pred_val) 

    print('r2 Val : ',val_r2_lr)
    print('mse Val : ',val_mse_lr)
    arr_lr_val_r2.append(val_r2_lr)
    arr_lr_val_mse.append(val_mse_lr)
    
    Y_pred_test = clf_lr.predict(X_test)
    test_mse_lr = mean_squared_error(Y_test,Y_pred_test)
    test_r2_lr = r2_score(Y_test, Y_pred_test) 

    print('r2 Test : ',test_r2_lr)
    print('mse Val : ',test_mse_lr)
    arr_lr_test_r2.append(test_r2_lr)
    arr_lr_test_mse.append(test_mse_lr)


print(np.mean(arr_lr_val_r2))
print(np.mean(arr_lr_val_mse))
print(np.mean(arr_lr_test_r2))
print(np.mean(arr_lr_test_mse))
result_modelling = pd.DataFrame({
    'model': ['GradientBoostingRegressor', 'RandomForestRegressor', 'Support Vector Regression', 
            'KNeighborsRegressor', 'XGBRegressor', 'Linear Regression'
             ],
    'val mse': [np.mean(arr_gb_val_mse), np.mean(arr_rf_val_mse), np.mean(arr_svm_val_mse),
                np.mean(arr_knn_val_mse), np.mean(arr_xgb_val_mse), np.mean(arr_lr_val_mse)
        
    ],
    
    'val r2': [np.mean(arr_gb_val_r2), np.mean(arr_rf_val_r2), np.mean(arr_svm_val_r2),
               np.mean(arr_knn_val_r2), np.mean(arr_xgb_val_r2), np.mean(arr_lr_val_r2)
        
    ],
    'test mse': [np.mean(arr_gb_test_mse), np.mean(arr_rf_test_mse), np.mean(arr_svm_test_mse),
                 np.mean(arr_knn_test_mse), np.mean(arr_xgb_test_mse), np.mean(arr_lr_test_mse)
        
    ],
    
    'test r2': [np.mean(arr_gb_test_r2), np.mean(arr_rf_test_r2), np.mean(arr_svm_test_r2),
                np.mean(arr_knn_test_r2), np.mean(arr_xgb_test_r2), np.mean(arr_lr_test_r2)
        
    ],
})
result_modelling['val mse'] = np.round(result_modelling['val mse'], decimals = 3)
result_modelling['val r2'] = np.round(result_modelling['val r2'], decimals = 3)
result_modelling['test mse'] = np.round(result_modelling['test mse'], decimals = 3)
result_modelling['test r2'] = np.round(result_modelling['test r2'], decimals = 3)
result_modelling = result_modelling.sort_values(by='val r2', ascending=False).reset_index(drop=True)
result_modelling
result_modelling.to_csv('../result/result_modelling.csv',index=False)
val = result_modelling['val r2']
test = result_modelling['test r2']

bars = result_modelling['model']
barwidth = 0.3

total_pos = np.arange(len(bars))
val_pos = [x + barwidth for x in total_pos]
test_pos = [x + 2*barwidth for x in total_pos]

fig,ax =plt.subplots(figsize=(15,10))
plt.bar(val_pos,val,width=barwidth, color = '#800000', alpha=0.9)
plt.bar(test_pos,test,width=barwidth, color = '#008000', alpha=0.9)

plt.bar(0.3,0,width=barwidth, color = '#800000',label='Val ')
plt.bar(0.6,0,width=barwidth, color = '#008000', label='Test ')
# plt.bar(0.3,0.9450,width=barwidth, color = '#800000')
# plt.bar(0.6,0.946977,width=barwidth, color = '#008000')

title = 'Model Analysis (Cross Validation and Test)'
txpos = 4 #title x coordinate
typos = 1.1 #title y coordinate
ax.text(txpos,typos,title,horizontalalignment='center',color='#800000',fontsize=20,fontweight='bold')


# insight = '''

# '''
# ixpos = 0.1 #insight x coordinate
# iypos = 1.05 #insight y coordinate
# ax.text(ixpos,iypos,insight,horizontalalignment='left',color='grey',fontsize=16,fontweight='normal')

plt.xticks(val_pos, bars,rotation=45)
ax.legend(loc='upper left', bbox_to_anchor= (1.01, 1.0), ncol=1, borderaxespad=0,frameon=False)
ax.set_ylim(0,1.2)




plt.savefig('../fig/result_modelling.png',bbox_inches='tight')
df_test = pd.read_csv('../input/volume_forecast.csv')
df_test.head(2)
df_test.isnull().sum()
df_test['SKU'] = df_test['SKU'].apply(lambda x: x[4:])
df_test['SKU'] = df_test['SKU'].astype(int)

df_test['Agency'] = df_test['Agency'].apply(lambda x: x[7:])
df_test['Agency'] = df_test['Agency'].astype(int)
df_test.head(2)
df_test = df_test.drop('Volume', axis=1)
X_test = sc_x.fit_transform(df_test.astype(float))
model_xgb = joblib.load('../model/xgb_2.dat')
y_pred = model_xgb.predict(X_test)
y_pd = pd.DataFrame({'Volume':y_pred})
print('finish predict')
y_pd.head()
df_test = pd.read_csv('../input/volume_forecast.csv') 
df_test = df_test.drop('Volume', axis=1)
df_test.head()
frames = [df_test, y_pd]
result = pd.concat(frames, axis=1)
result.to_csv('../result/volume_forecast.csv',index=False)
result.head()
df = pd.read_csv('../input/train.csv')
df['Volume'] = df['Volume'].round(2)
df['Price'] = df['Price'].round(2)
df['Sales'] = df['Sales'].round(2)
df['Promotions'] = df['Promotions'].round(2)
df['Avg_Max_Temp'] = df['Avg_Max_Temp'].round(2)
df.head()
df.isnull().sum()
# df = df.drop('YearMonth', axis=1)
# df_temp = df
cols = list(df)

corr_ =df[cols].corr()
plt.figure(figsize=(16,10))
sns.heatmap(corr_, annot=True, fmt = ".2f", cmap = "BuPu")
plt.savefig('../fig/Data Numeric Corr.png',bbox_inches='tight')
df = df.drop('Soda_Volume',axis=1)
df = df.drop('Good Friday',axis=1)
df = df.drop('Sales',axis=1)
df = df.drop('Revolution Day Memorial',axis=1)
df = df.drop('Independence Day',axis=1)
df = df.drop('Beer Capital',axis=1)
df = df.drop('New Year',axis=1)
df = df.drop('Avg_Max_Temp',axis=1)
df = df.drop('FIFA U-17 World Cup',axis=1)
df = df.drop('Football Gold Cup',axis=1)
df = df.drop('Avg_Population_2017',axis=1)
df = df.drop('Avg_Yearly_Household_Income_2017',axis=1)
df.isnull().sum()
corr_ =df[list(df)].corr()
plt.figure(figsize=(16,10))
sns.heatmap(corr_, annot=True, fmt = ".2f", cmap = "BuPu")
plt.savefig('../fig/Data Numeric Corr 2.png',bbox_inches='tight')
df = df[['YearMonth','Volume','Price','Promotions']].copy()
df.head()
df = df.groupby('YearMonth')['Volume','Price','Promotions'].agg(['sum','mean','std']).reset_index()
df.head()
df.columns = ['YearMonth','v_sum','v_mean','v_std','p_sum','p_mean','p_std','pr_sum','pr_mean','pr_std']
len(df)
df = df.sort_values('YearMonth', ascending=True).reset_index(drop=True)
df['YearMonth'] = pd.to_datetime(df['YearMonth'])
df = df.merge(df2, on='YearMonth', how='inner')
df.head()
df = df.drop('Good Friday',axis=1)
df = df.drop('Revolution Day Memorial',axis=1)
df = df.drop('Independence Day',axis=1)
df = df.drop('Beer Capital',axis=1)
df = df.drop('New Year',axis=1)
df = df.drop('FIFA U-17 World Cup',axis=1)
df = df.drop('Football Gold Cup',axis=1)
data = df
data= data.drop('YearMonth', axis=1)
data.head(2)
data.info()
len(list(data))
fig,ax = plt.subplots(14,1,figsize=(20,15))
for i,column in enumerate([col for col in data.columns if col != 'wnd_dir']):
    data[column].plot(ax=ax[i])
    ax[i].set_title(column)
plt.savefig('../fig/Dataset1.png',bbox_inches='tight')
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
values = data.values
print(values.shape)
values = values.astype('float32')
series_to_supervised(values,1,1).head()
reframed = series_to_supervised(values,1,1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[15,16,17,18,19,20,21,22,23,24,25,26,27]], axis=1, inplace=True)
print(len(list(reframed)))
reframed.head()
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled,1,1)
reframed.drop(reframed.columns[[15,16,17,18,19,20,21,22,23,24,25,26,27]], axis=1, inplace=True)
reframed.head()
values = reframed.values
n_train = 49
train = values[:n_train]
test = values[n_train:]
trainX,trainY = train[:,:-1],train[:,-1]
testX,testY = test[:,:-1],test[:,-1]

print(trainX.shape,trainY.shape,testX.shape,testY.shape)

trainX = trainX.reshape(trainX.shape[0],1,trainX.shape[1])
testX = testX.reshape(testX.shape[0],1,testX.shape[1])


print(trainX.shape)
print(testX.shape)
len(testX)
stop_noimprovement = EarlyStopping(patience=10)
model = Sequential()
model.add(LSTM(50,input_shape=(trainX.shape[1],trainX.shape[2]),dropout=0.2))
model.add(Dense(1))
model.compile(loss="mae",optimizer="adam")

history= model.fit(trainX,trainY,validation_data=(testX,testY),epochs=100,verbose=2,callbacks=[stop_noimprovement],shuffle=False)

model.summary()
fig,ax =plt.subplots(figsize=(15,10))

plt.plot(history.history['loss'],label='train loss')
plt.plot(history.history['val_loss'],label='val loss')
plt.legend()
# plt.savefig('../fig/trai val forecasting.png',bbox_inches='tight')
len(testX)
predicted = model.predict(testX)
testXRe = testX.reshape(testX.shape[0],testX.shape[2])
predicted = np.concatenate((predicted,testXRe[:,1:]),axis=1)
print('predicted.shape : ',predicted.shape)
predicted = model.predict(testX)

testXRe = testX.reshape(testX.shape[0],testX.shape[2])
predicted = np.concatenate((predicted,testXRe[:,1:]),axis=1)
print('predicted.shape : ',predicted.shape)

predicted = scaler.inverse_transform(predicted)
testY = testY.reshape(len(testY),1)
print('testY.shape : ',testY.shape)

testY = np.concatenate((testY,testXRe[:,1:]),axis=1)
testY = scaler.inverse_transform(testY)
pd.DataFrame(testY)
np.sqrt(mean_squared_error(testY[:,0],predicted[:,0]))
result = pd.concat([pd.Series(predicted[:,0]),pd.Series(testY[:,0])],axis=1)
result.columns = ['thetahat','theta']
result['diff'] = result['thetahat'] - result['theta']
result = pd.concat([pd.Series(predicted[:,0]),pd.Series(testY[:,0])],axis=1)
result.columns = ['thetahat','theta']
result['diff'] = result['thetahat'] - result['theta']
result.head()
df_new = df[['YearMonth','Volume','Price','Promotions']].copy()
df_new.head()
df_new = df_new.groupby('YearMonth')['Volume'].agg(['sum']).reset_index()
df_new.head()
df_new['YearMonth'] = pd.to_datetime(df_new['YearMonth'])
df_new = df_new.sort_values(by='YearMonth').reset_index(drop=True)
df_new = df_new.drop('YearMonth', axis=1)
df_new.head()
fig,ax =plt.subplots(figsize=(15,10))
plt.plot(df_new)
plt.xlabel('month')
plt.ylabel('sum of volume')
plt.savefig('../fig/forecasting unvariate 1.png',bbox_inches='tight')
train = df_new[0:50]
test = df_new[50:]
# load the trainset
train = train.values
train = train.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)

# load the testset
test = test.values
test = test.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
test = scaler.fit_transform(test)
print(test.shape)
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
look_back = 1
testX, testY = create_dataset(test, look_back)


trainX, trainY = create_dataset(train, look_back)
trainX.shape
trainY.shape
testX.shape
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

history= model.fit(trainX,trainY,validation_data=(testX,testY),epochs=100,verbose=2,callbacks=[stop_noimprovement],shuffle=False)

model.summary()
fig,ax =plt.subplots(figsize=(15,10))

plt.plot(history.history['loss'],label='train loss')
plt.plot(history.history['val_loss'],label='val loss')
plt.legend()
plt.savefig('../fig/unvariate 1.png',bbox_inches='tight')
testPredict = model.predict(testX)
print(testPredict.shape)
testPredict = scaler.inverse_transform(testPredict)
testPredict
len(test)
test = df_new[55:]
test
test.values
# load the testset
test = test.values
test = test.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
test = scaler.fit_transform(test)
test = np.expand_dims(test, axis=1)
test.shape
data= test
# data = np.expand_dims(data, axis=1)

print('data n',data)
print('\n')


predict = model.predict(data)
print('predict data n ',predict)
print('\n')


future = predict
future = np.expand_dims(future, axis=1)

future_predict = model.predict(future)
print('predict data n + 1 ',future_predict)
future_predict = scaler.inverse_transform(future_predict)
future_predict