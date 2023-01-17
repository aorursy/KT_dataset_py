import pandas as pd

import numpy as np



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostClassifier,cv,Pool



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%matplotlib inline
train_orig = pd.read_csv('../input/agriculture-prediction-av/train (1).csv')

test_orig = pd.read_csv('../input/agriculture-prediction-av/test (1).csv')

train_orig.head()
train_orig.shape,test_orig.shape
train_orig.isna().sum(),test_orig.isna().sum()
train_orig.info(),test_orig.info()
#Imputing the values of crop_damage as 99 in the test data

test_orig['Crop_Damage'] = 99
df = pd.merge(train_orig,test_orig,on=[x for x in train_orig.columns],how='outer')

df['ID'] = df['ID'].apply(lambda x: int(x.strip('F')))

df.sort_values('ID',inplace=True)
plt.figure(figsize=(15,6))

plt.subplot(1,3,1)

plt.plot(df['ID'][0:2000],df['Estimated_Insects_Count'][0:2000])

plt.title('Insects Count')

plt.subplot(1,3,2)

plt.plot(df['ID'][0:1500],df['Number_Weeks_Used'][0:1500])

plt.title('Weeks Used')

plt.subplot(1,3,3)

plt.plot(df['ID'][0:5000],df['Number_Weeks_Quit'][0:5000])

plt.title('Weeks Quit')
#While performing EDA we also found that the number doses week and estimated insects count columns were not Normally distributed

#so performing basic operations to convert them to normal distribution

df['Number_Weeks_Used'].fillna(value=int(df['Number_Weeks_Used'].mean()),inplace=True)



df['sqrt_no_doses_week'] = np.sqrt(df['Number_Doses_Week'])



df['sqrt_insects'] = np.sqrt(df['Estimated_Insects_Count'])

#Though this doesn't produce effect much to the Decision Tree Models, it provided some significant features
def feature_extract(data):

  for i in ['Crop_Type','Soil_Type','Pesticide_Use_Category','Season']:

    group = data[['Estimated_Insects_Count','Number_Doses_Week','Number_Weeks_Used','Number_Weeks_Quit',i]].groupby([i]).agg(['mean','min','max','std'])

    group.columns = ['_'.join(x) + f'_{i}' for x in group.columns.ravel()]

    data = pd.merge(data,group,on=i,how='left')

  return data
df = feature_extract(df)
#Average insects per crop per soil

df['avg_insects_per_crop_soil'] = df[['Estimated_Insects_Count','Crop_Type','Soil_Type']].groupby(['Soil_Type','Crop_Type'])['Estimated_Insects_Count'].transform('mean')



#Average insects per soil per crop

df['avg_insects_per_soil_crop'] = df[['Estimated_Insects_Count','Crop_Type','Soil_Type']].groupby(['Crop_Type','Soil_Type'])['Estimated_Insects_Count'].transform('mean')



#Average insects per pesticide per crop per soil

df['avg_insects_per_pest_crop_soil'] = df[['Estimated_Insects_Count','Crop_Type','Soil_Type','Pesticide_Use_Category']].groupby(['Soil_Type','Crop_Type','Pesticide_Use_Category'])['Estimated_Insects_Count'].transform('mean')



#Average insects per pesticide per soil per crop

df['avg_insects_per_pest_soil_crop'] = df[['Estimated_Insects_Count','Crop_Type','Soil_Type','Pesticide_Use_Category']].groupby(['Crop_Type','Soil_Type','Pesticide_Use_Category'])['Estimated_Insects_Count'].transform('mean')



#Average insects per season per crop per soil

df['avg_insects_per_season_crop_soil'] = df[['Estimated_Insects_Count','Crop_Type','Soil_Type','Season']].groupby(['Soil_Type','Crop_Type','Season'])['Estimated_Insects_Count'].transform('mean')



#Average insects per season per soil per crop

df['avg_insects_per_season_crop_soil'] = df[['Estimated_Insects_Count','Crop_Type','Soil_Type','Season']].groupby(['Crop_Type','Soil_Type','Season'])['Estimated_Insects_Count'].transform('mean')
#Extracting rolling mean features from Estimated Insects Count

df['rolmean_insects_weeks_used'] = df[['Estimated_Insects_Count','Number_Weeks_Used']].groupby(['Number_Weeks_Used'])['Estimated_Insects_Count'].transform(lambda x: x.rolling(5).mean())



df['rolmean_insects_weeks_quit'] = df[['Estimated_Insects_Count','Number_Weeks_Quit']].groupby(['Number_Weeks_Quit'])['Estimated_Insects_Count'].transform(lambda x: x.rolling(window=1).mean())



#Rolling mean specifically on insects count,number of weeks used, number weeks quit and number of doses week

df['rolmean_insects'] = df['Estimated_Insects_Count'].transform(lambda x: x.rolling(window=5).mean())



df['rolmean_weeks_used'] = df['Number_Weeks_Used'].transform(lambda x: x.rolling(window=5).mean())



df['rolmean_weeks_quit'] = df['Number_Weeks_Quit'].transform(lambda x: x.rolling(window=5).mean())



df['rolmean_doses_week'] = df['Number_Doses_Week'].transform(lambda x: x.rolling(window=5).mean())



df['rolmean_season'] = df['Season'].transform(lambda x: x.rolling(window=5).mean())



df.fillna(0,inplace=True)
X = df[df['Crop_Damage'] != 99]

X_valid = df[df['Crop_Damage'] == 99]
#XGBboost Model

xgb1 = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.8, gamma=0,

              learning_rate=0.1, max_delta_step=0, max_depth=5,

              min_child_weight=1, missing=None, n_estimators=686, n_jobs=-1,

              nthread=4, num_class=3, objective='multi:softprob',

              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

              seed=27, silent=None, subsample=0.8, verbosity=1)



model_xgb = xgb1.fit(X.drop(['ID','Crop_Damage'],axis=1),X['Crop_Damage'],eval_metric='merror')

y_pred_xgb1 = model_xgb.predict(X_valid.drop(['ID','Crop_Damage'],axis=1))



predicted_res_xgb1 = pd.DataFrame(y_pred_xgb1,columns=['Crop_Damage'])

test_res_xgb1 = pd.concat([test_orig[['ID']],predicted_res_xgb1],axis=1)

test_res_xgb1 = test_res_xgb1[['ID','Crop_Damage']]

test_res_xgb1.set_index(['ID'],inplace=True)

test_res_xgb1.to_csv('test_xgb1.csv')