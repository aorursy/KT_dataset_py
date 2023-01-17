import pandas as pd

import numpy as np



import xgboost as xgb

import lightgbm as lgb

import catboost as cb



from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder



import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/cross-sell-prediction/train.csv')

test = pd.read_csv('../input/cross-sell-prediction/test.csv')
df = pd.merge(train,test,on=[x for x in train.columns if x not in ['Response']],how='outer')

df.head()
#Check the data distribution in the dataframe

df.describe()
sns.set_style('whitegrid')

sns.boxplot(x=df['Annual_Premium'])
#Set an upper limit of 1,50,000 to the premium paid by the customer

df.loc[df['Annual_Premium']>150000,'Annual_Premium'] = 150000
#Normalising the data of Age, Annual Premium and Vintage columns 

df['log_age'] = np.log(df['Age'])

df['sqrt_premium'] = np.sqrt(df['Annual_Premium'])

df['log_vintage'] = np.log(df['Vintage'])
#Calculating mean and std of premium paid and vintage on Previously Insured per Sales Channel used

group = df.groupby(['Policy_Sales_Channel','Previously_Insured'])['Annual_Premium'].agg(['mean','std'])

group.columns = [x + '_channel_insured_premium' for x in group.columns.ravel()]

df = pd.merge(df,group,on=['Policy_Sales_Channel','Previously_Insured'],how='left')



group = df.groupby(['Policy_Sales_Channel','Previously_Insured'])['Vintage'].agg(['mean','std'])

group.columns = [x + '_channel_insured_vintage' for x in group.columns.ravel()]

df = pd.merge(df,group,on=['Policy_Sales_Channel','Previously_Insured'],how='left')
#Calculating mean and std of premium paid and vintage on Previously Insured per Region

group = df.groupby(['Region_Code','Previously_Insured'])['Annual_Premium'].agg(['mean','std'])

group.columns = [x +'_region_insured_premium' for x in group.columns.ravel()]

df = pd.merge(df,group,on=['Region_Code','Previously_Insured'],how='left')



group = df.groupby(['Region_Code','Previously_Insured'])['Vintage'].agg(['mean','std'])

group.columns = [x +'_region_insured_vintage' for x in group.columns.ravel()]

df = pd.merge(df,group,on=['Region_Code','Previously_Insured'],how='left')
#Calculating mean and std of premium paid and vintage on Vehicle Damage per Region

group = df.groupby(['Region_Code','Vehicle_Damage'])['Annual_Premium'].agg(['mean','std'])

group.columns = [x +'_region_damage_premium' for x in group.columns.ravel()]

df = pd.merge(df,group,on=['Region_Code','Vehicle_Damage'],how='left')



group = df.groupby(['Region_Code','Vehicle_Damage'])['Vintage'].agg(['mean','std'])

group.columns = [x +'_region_damage_vintage' for x in group.columns.ravel()]

df = pd.merge(df,group,on=['Region_Code','Vehicle_Damage'],how='left')
#Calculating mean and std of premium paid and vintage on the basis of Vehicle Damage

group = df.groupby(['Vehicle_Damage'])['Annual_Premium'].agg(['mean','std'])

group.columns = [x + '_damage_premium' for x in group.columns.ravel()]

df = pd.merge(df,group,on=['Vehicle_Damage'],how='left')



group = df.groupby(['Vehicle_Damage'])['Vintage'].agg(['mean','std'])

group.columns = [x + '_damage_vintage' for x in group.columns.ravel()]

df = pd.merge(df,group,on=['Vehicle_Damage'],how='left')
#Calculating mean and std of premium paid and vintage on Vehicle Damage per customer Age

group = df.groupby(['Age','Vehicle_Damage'])['Vintage'].agg(['mean','std'])

group.columns = [x +'_age_damage_vintage' for x in group.columns.ravel()]

df = pd.merge(df,group,on=['Age','Vehicle_Damage'],how='left')



group = df.groupby(['Age','Vehicle_Damage'])['Annual_Premium'].agg(['mean','std'])

group.columns = [x +'_age_damage_premium' for x in group.columns.ravel()]

df = pd.merge(df,group,on=['Age','Vehicle_Damage'],how='left')
#Counting the number of customers for Previously Insured column agewise

group = df.groupby(['Age']).agg(cnt_age_insured = ('Previously_Insured','count'))

df = pd.merge(df,group,on=['Age'],how='left')



group = df.groupby(['Region_Code']).agg(cnt_region_insured = ('Previously_Insured','count'))

df = pd.merge(df,group,on=['Region_Code'],how='left')
le = LabelEncoder()

for i in ['Gender','Vehicle_Age','Vehicle_Damage']:

  df[i] = le.fit_transform(df[i])
#Converting float dtypes column to int for the catboost model to accept it

df['Region_Code'] = df['Region_Code'].astype(int)

df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype(int)
X = df[df['Response'].notnull()]

X_valid = df[df['Response'].isnull()]
cat_col = ['Gender','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Policy_Sales_Channel']



#cv_data = cb.Pool(X.drop(columns=['id','Response']),label=X['Response'],cat_features=cat_col)



params = {'iterations':2000,

         'learning_rate':0.05,

         'thread_count':4,

         'eval_metric':'AUC',

          'loss_function':'Logloss'}



#cv_res = cb.cv(dtrain=cv_data,early_stopping_rounds=100,nfold=5,params=params,plot=True)
#The cross validation gave an AUC score of 0.8589179 for the model of 384 trees with a learning rate of 0.05 as the best one

cat_col = ['Gender','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Policy_Sales_Channel']



cbc_new = cb.CatBoostClassifier(iterations=384,learning_rate=0.05,thread_count=4,cat_features=cat_col,eval_metric='AUC')

cbc_new.fit(X.drop(columns=['id','Response']),X['Response'])



y_pred_cbc_new = cbc_new.predict_proba(X_valid.drop(columns=['id','Response']))[:,1]

test_res = test['id']

test_res = pd.concat([test_res,pd.DataFrame(y_pred_cbc_new,columns=['Response'])],axis=1)

test_res.set_index('id',inplace=True)

test_res.to_csv('sub_cbc_final.csv')
#Plotting Feature Importance

score_dict = {}

feature_importances = cbc_new.get_feature_importance()

feature_names = X.drop(columns=['id','Response']).columns



for score, name in sorted(zip(feature_importances, feature_names), reverse=True):

  score_dict.update({name:score})



score_list = score_dict.items()

x,y = zip(*score_list)



plt.figure(figsize=(8,12))

plt.title('Feature Importance')

plt.barh(x,y)

plt.xlabel('Feature Score')

plt.ylabel('Features')