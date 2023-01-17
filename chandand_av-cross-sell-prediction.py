import warnings

warnings.filterwarnings('ignore')

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold,StratifiedKFold,train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier

import os

import eli5

from eli5.sklearn import PermutationImportance

from catboost import CatBoostClassifier



from keras.models import Sequential,load_model

from keras.layers import Dense, Dropout

from sklearn.preprocessing import StandardScaler

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

train_df=pd.read_csv('../input/av-janatahack-crosssell-prediction/train.csv')

test_df=pd.read_csv('../input/av-janatahack-crosssell-prediction/test.csv')

submission_df=pd.read_csv('../input/av-janatahack-crosssell-prediction/sample.csv')
train_df.head(5)
sns.distplot(train_df['Annual_Premium'])
train_df['Annual_Premium']=np.log(train_df['Annual_Premium'])

sns.distplot(train_df['Annual_Premium'])
plt.figure(figsize=(10,10))

sns.heatmap(train_df.corr(),annot=True)
sns.barplot(train_df['Response'],train_df['Response'].value_counts())
train_df=train_df.drop_duplicates(subset=[ele for ele in list(train_df.columns) if ele not in ['id']])

combine_set=pd.concat([train_df,test_df])

le=LabelEncoder()

combine_set['Gender']=le.fit_transform(combine_set['Gender'])

combine_set['Vehicle_Damage']=le.fit_transform(combine_set['Vehicle_Damage'])

combine_set['Vehicle_Age']=combine_set['Vehicle_Age'].map({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})

# df=pd.get_dummies(combine_set['Vehicle_Age'],drop_first=True)

# combine_set=pd.concat([combine_set,df],axis=1)



# fe=combine_set.groupby('Vehicle_Age').size()/len(combine_set)

# combine_set['Vehicle_Age']=combine_set['Vehicle_Age'].apply(lambda x: fe[x])

combine_set['Customer_term_in_year']=combine_set['Vintage']/365

combine_set['Total_premium_Channelwise']=combine_set.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('sum')

combine_set['Mean_premium_Channelwise']=combine_set.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('mean')

combine_set['Maximum_premium_Channelwise']=combine_set.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('max')

combine_set['Min_premium_Channelwise']=combine_set.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('min')

combine_set['Total_premium_regionwise']=combine_set.groupby(['Region_Code'])['Annual_Premium'].transform('sum')

combine_set['Mean_premium_regionwise']=combine_set.groupby(['Region_Code'])['Annual_Premium'].transform('mean')

combine_set['Max_premium_regionwise']=combine_set.groupby(['Region_Code'])['Annual_Premium'].transform('max')

combine_set['Min_premium_regionwise']=combine_set.groupby(['Region_Code'])['Annual_Premium'].transform('min')

combine_set['Age_groups_region_wise']=combine_set.groupby(['Region_Code'])['Age'].transform('nunique')

combine_set['regionwise_channels']=combine_set.groupby(['Policy_Sales_Channel'])['Region_Code'].transform('nunique')

combine_set['Channelwise_regions']=combine_set.groupby(['Region_Code'])['Policy_Sales_Channel'].transform('nunique')

combine_set['Unique_customers_based_Vinatge']=combine_set.groupby(['Region_Code','Policy_Sales_Channel'])['Vintage'].transform('nunique')

combine_set['Region_wise_Vehicle_Age_premium']=combine_set.groupby(['Region_Code','Vehicle_Age'])['Annual_Premium'].transform('sum')

combine_set['Region_wise_Vehicle_Age_premium_mean']=combine_set.groupby(['Region_Code','Vehicle_Age'])['Annual_Premium'].transform('mean')

combine_set['Region_wise_Vehicle_Age_premium_max']=combine_set.groupby(['Region_Code','Vehicle_Age'])['Annual_Premium'].transform('max')

combine_set['Channel_wise_Vehicle_Age_premium']=combine_set.groupby(['Policy_Sales_Channel', 'Vehicle_Age'])['Annual_Premium'].transform('sum')

combine_set['Channel_wise_Vehicle_Age_premium_mean']=combine_set.groupby(['Policy_Sales_Channel', 'Vehicle_Age'])['Annual_Premium'].transform('mean')

combine_set['Channel_wise_Vehicle_Age_premium_max']=combine_set.groupby(['Policy_Sales_Channel', 'Vehicle_Age'])['Annual_Premium'].transform('max')



#Rank Features

combine_set['Rank_regionwise_premium']=combine_set.groupby(['Region_Code'])['Annual_Premium'].rank(method='first',ascending=True)

combine_set['Rank_mean_regionwise_premium']=combine_set.groupby(['Region_Code'])['Annual_Premium'].rank(method='average',ascending=True)

combine_set['Rank_max_regionwise_premium']=combine_set.groupby(['Region_Code'])['Annual_Premium'].rank(method='max',ascending=True)

combine_set['Rank_min_regionwise_premium']=combine_set.groupby(['Region_Code'])['Annual_Premium'].rank(method='min',ascending=True)

combine_set['Rank_regionwise_diff']=combine_set['Rank_max_regionwise_premium']- combine_set['Rank_min_regionwise_premium']

combine_set['Rank_channelwise_premium']=combine_set.groupby(['Policy_Sales_Channel'])['Annual_Premium'].rank(method='first',ascending=True)

combine_set['Rank_mean_channelwise_premium']=combine_set.groupby(['Policy_Sales_Channel'])['Annual_Premium'].rank(method='average',ascending=True)

combine_set['Rank_max_channelwise_premium']=combine_set.groupby(['Policy_Sales_Channel'])['Annual_Premium'].rank(method='max',ascending=True)

combine_set['Rank_min_channelwise_premium']=combine_set.groupby(['Policy_Sales_Channel'])['Annual_Premium'].rank(method='min',ascending=True)

combine_set['Rank_channelwise_diff']=combine_set['Rank_max_channelwise_premium']- combine_set['Rank_min_channelwise_premium']

combine_set['Rank_Channel_wise_Vehicle_Age_Premium']=combine_set.groupby(['Policy_Sales_Channel','Vehicle_Age'])['Annual_Premium'].rank(method='first',ascending=True)

combine_set['Rank_Region_wise_Vehicle_Age_premium']=combine_set.groupby(['Region_Code','Vehicle_Age'])['Annual_Premium'].rank(method='first',ascending=True)

combine_set['Rank_Age_wise_premium']=combine_set.groupby(['Age'])['Annual_Premium'].rank(method='first',ascending=True)



combine_set.head(5)
cat_cols = train_df.select_dtypes(include = 'object')

num_cols = train_df.select_dtypes(include=['int64','float64'])

combine=pd.concat([train_df,test_df])

combine['Vintage'] = combine['Vintage']/365

combine['Vehicle_Age']=combine['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})

combine['Vehicle_Damage']=combine['Vehicle_Damage'].replace({'Yes':1,'No':0})

combine['Gender']=combine['Gender'].replace({'Male':1,'Female':0})

combine['IsPreviouslyInsuredandVehicleDamaged'] = np.where((combine['Previously_Insured']==0) & (combine['Vehicle_Damage']==1),1,0)

combine['IsVehicleDamagedandDrivingLicense'] = np.where((combine['Vehicle_Damage']==1) & (combine['Driving_License']==1),1,0)

combine['TotalAmountPaidTillDate'] = combine['Annual_Premium']*combine['Vintage']

combine['PremiumperRegion'] = combine.groupby('Region_Code')['Annual_Premium'].transform('mean')

combine['PremiumperPolicy_Sales_Channel'] = combine.groupby('Policy_Sales_Channel')['Annual_Premium'].transform('mean')

combine['AvgVehicleAgePerRegion'] = combine.groupby('Policy_Sales_Channel')['Annual_Premium'].transform('mean')

combine['AvgCustomerAgeRegionWise'] = combine.groupby('Region_Code')['Age'].transform('mean')

combine['AvgCustomerAgeSaleChannelWise'] = combine.groupby('Policy_Sales_Channel')['Age'].transform('mean')

combine['SaleChannelsPerRegion'] = combine.groupby('Region_Code')['Policy_Sales_Channel'].transform('nunique')

combine['RegionwisePreviouslyInsured'] = combine.groupby('Region_Code')['Previously_Insured'].transform('count')

combine['RegionwiseVintage'] = combine.groupby('Region_Code')['Vintage'].transform('mean').astype('int')

combine['SaleChannelwiseVintage'] = combine.groupby('Policy_Sales_Channel')['Vintage'].transform('mean').astype('int')

combine['AvgRegionGenderWisePremium'] = combine.groupby(['Region_Code','Gender'])['Annual_Premium'].transform('mean')

combine['NoPeoplePrevInsuredRegionGenderWise'] = combine.groupby(['Region_Code','Gender'])['Previously_Insured'].transform('count')

combine['NoPeoplePrevInsuredSalesChannelGenderWise'] = combine.groupby(['Policy_Sales_Channel','Gender'])['Previously_Insured'].transform('count')

combine['NoPeoplePrevInsuredSalesChannelRegionWise'] = combine.groupby(['Region_Code','Policy_Sales_Channel'])['Previously_Insured'].transform('count')

combine['AvgCustomerDurationRegionGenderWise'] = combine.groupby(['Region_Code','Gender'])['Vintage'].transform('mean')

combine['InsuranceLicense'] = combine['Driving_License'].astype('str') + '' + combine['Previously_Insured'].astype('str')

combine['InsuranceGender'] = combine['Gender'].astype('str') + '' + combine['Previously_Insured'].astype('str')

combine['Region_Code']=combine['Region_Code'].astype(int)

combine['Policy_Sales_Channel']=combine['Policy_Sales_Channel'].astype(int)

combine.head(5)

train_df=combine_set[combine_set['Response'].isnull()==False]

test_df=combine_set[combine_set['Response'].isnull()==True]

X=train_df.drop(['id','Response'],axis=1)

y=train_df['Response'] 

X_main_test=test_df.drop(['id','Response'],axis=1)

cat_col=['Gender','Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel', 'InsuranceLicense','InsuranceGender']

train = combine[combine['Response'].isnull()!= True]

test = combine[combine['Response'].isnull()== True]

test=test.drop(['id','Response'],axis=1)

X_cat = train.drop(['id',"Response"], axis=1)

Y = train["Response"]
# #Check for Permutation Importance of Features

# perm = PermutationImportance(lg,random_state=294).fit(X_val, y_val)

# eli5.show_weights(perm,feature_names=X_val.columns.tolist())

#Kfold

kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=294)

pred_score=0

preds=0



for i, (train_idx,val_idx) in enumerate(kf.split(X,y)):

    X_train,y_train=X.iloc[train_idx,:],y.iloc[train_idx]

    X_val,y_val=X.iloc[val_idx,:],y.iloc[val_idx]

    

    print('\nFold: {}\n'.format(i+1))

    

    lg=LGBMClassifier(boosting_type='gbdt',n_estimators=500,depth=10,learning_rate=0.04,objective='binary',metric='auc',

                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=294,n_jobs=-1)      

   



    lg.fit(X_train,y_train)

    print(roc_auc_score(y_val,lg.predict_proba(X_val)[:,1]))

    

    pred_score+=roc_auc_score(y_val,lg.predict_proba(X_val)[:,1])

    

    preds+=lg.predict_proba(X_main_test)[:,1]

    

print('mean_score: {}'.format(pred_score/10))



preds_lg=preds/10



    
kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=294)

predictions=[]

test_roc_score=[]



    

for i,(train_idx,val_idx) in enumerate(kf.split(X_cat,Y)):    



    X_train, y_train = X_cat.iloc[train_idx,:], Y.iloc[train_idx]



    X_val, y_val = X_cat.iloc[val_idx, :], Y.iloc[val_idx]

    



    print('\nFold: {}\n'.format(i+1))



    classifier = CatBoostClassifier(learning_rate = 0.055,random_state=42,scale_pos_weight=7, custom_metric=['AUC'])



    classifier.fit(X_train,y_train,cat_features=cat_col,eval_set=(X_val, y_val),early_stopping_rounds=30,verbose=100)

    

    testpred = classifier.predict_proba(X_val)[:,1]

    test_roc_score.append(roc_auc_score(y_val, testpred))

    print("Test ROC AUC : %.4f"%(roc_auc_score(y_val, testpred)))

    predictions.append(classifier.predict_proba(test)[:,1])



print("Mean test score:",np.mean(test_roc_score))

preds_cb=np.mean(predictions,axis=0)
#Submission File

submission_df['Response']=preds_lg*0.6+preds_cb*0.4

submission_df.to_csv('main_test.csv',index=False)

submission_df.head(5)

# np.array(lg.predict_proba(X_main_test)[:,1])