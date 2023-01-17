path = '../input/mse-pysupport-hackathon/simulation.csv'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(path)

import warnings
warnings.filterwarnings("ignore")

df.head()
# URL for Holdout data - https://www.kaggle.com/pranjalrawat/msepysupportholdout
path1 = '../input/mse-pysupport-hackathon/holdout.csv'
import pandas as pd
holdout = pd.read_csv(path1)
holdout.head()
##generating percentage of missing values in each variable(simulation)
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
missing_value.sort_values(by=['percent_missing'],ascending=False,inplace=True)
missing_value.head(30)
percent_missing_hold= holdout.isnull().sum() * 100 / len(holdout)
missing_value_hold= pd.DataFrame({'column_name': holdout.columns,
                                 'percent_missing': percent_missing_hold})
missing_value_hold.sort_values(by=['percent_missing'],ascending=False,inplace=True)
missing_value_hold.head(30)
df.drop(['Renovation','AtticFinish','OtherImprovements'],axis=1,inplace=True)
holdout.drop(['Renovation','AtticFinish','OtherImprovements'],axis=1,inplace=True)
print(df.shape,holdout.shape)
df_apt = df[df['PropertyDesc'] == 'Residential condominium']
sim_condos = df[df['Apartments'].isnull()]
print(sim_condos.shape,df_apt.shape)
holdout_condos=holdout[holdout['Apartments'].isnull()]
holdout_condos.shape
df_initial=df.copy()
hold_initial=holdout.copy()
df=df[df['Apartments'].notnull()]
holdout=holdout[holdout['Apartments'].notnull()]
df_apt = df[df['PropertyDesc'] != 'Residential condominium']
print(df_apt.shape,df.shape)
df['ArmsLengthTransaction'].value_counts(normalize=True)
ax = sns.barplot(y="SalePrice", x="ArmsLengthTransaction", data=df)
plt.rcParams["figure.figsize"] = (8,6)
family_trans_sim=df[df['ArmsLengthTransaction']==0]
family_trans_sim.shape
df=df[df['ArmsLengthTransaction']==1]
categ_cols=['MostRecentSale','PropertyAddress','OHareNoise','Floodplain','RoadProximity','TypeOfResidence','SiteDesirability','Basement','BasementFinish','CentralHeating',
           'WallMaterial','RoofMaterial','ConstructionQuality','Porch','RepairCondition','Garage2Area']
for i in categ_cols:
    df[i]=df[i].fillna(df[i].mode()[0])
    holdout[i]=holdout[i].fillna(df[i].mode()[0])
    family_trans_sim[i]=family_trans_sim[i].fillna(family_trans_sim[i].mode()[0])

percent_missing = df.isnull().sum() * 100 / len(df)
missing_value = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
missing_value.sort_values(by=['percent_missing'],ascending=False,inplace=True)
missing_value.head(10)
cont_cols=['Rooms','Bedrooms']

for i in cont_cols:
    df[i]=df[i].fillna(df[i].median())
    holdout[i]=holdout[i].fillna(df[i].median())
    family_trans_sim[i]=family_trans_sim[i].fillna(family_trans_sim[i].median())
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
missing_value.sort_values(by=['percent_missing'],ascending=False,inplace=True)
missing_value.head(10)
df['year'] = pd.DatetimeIndex(df['SaleDate']).year
df['month']=pd.DatetimeIndex(df['SaleDate']).month
holdout['year'] = pd.DatetimeIndex(holdout['SaleDate']).year
holdout['month']=pd.DatetimeIndex(holdout['SaleDate']).month
family_trans_sim['year'] = pd.DatetimeIndex(family_trans_sim['SaleDate']).year
family_trans_sim['month']=pd.DatetimeIndex(family_trans_sim['SaleDate']).month
sim_condos['year'] = pd.DatetimeIndex(sim_condos['SaleDate']).year
sim_condos['month'] = pd.DatetimeIndex(sim_condos['SaleDate']).month
holdout_condos['month']=pd.DatetimeIndex(holdout_condos['SaleDate']).month
holdout_condos['year']=pd.DatetimeIndex(holdout_condos['SaleDate']).year
pred_cols=['Rooms','Bedrooms','EstimateLand','EstimateBuilding','LandSquareFeet','BuildingSquareFeet','NumberOfCommercialUnits','FullBaths','HalfBaths','Fireplaces','Garage1Size','Garage2Size']
y=df['SalePrice']
X=df[pred_cols]


from sklearn.ensemble import RandomForestRegressor 
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)  
regressor.fit(X, y)   

family_trans_y=regressor.predict(family_trans_sim[pred_cols])
family_trans_sim['SalePrice']=family_trans_y
sns.distplot(np.log(family_trans_sim['SalePrice']))
plt.title('Distribution of Imputed Family Transaction values')
#Concat imputed values with original dataframe
df_final=pd.concat([df,family_trans_sim])
clean_df=df_final.copy()
clean_df.shape
df_final.shape
##Modifying features
df_final['GarageSize']=df_final['Garage1Size']+df_final['Garage2Size']
holdout['GarageSize']=holdout['Garage1Size']+holdout['Garage2Size']
df_final['TotalEstimate']=df_final['EstimateLand']+df_final['EstimateBuilding']
holdout['TotalEstimate']=holdout['EstimateLand']+holdout['EstimateBuilding']
df_final['area'] = df_final['Pin'].astype(str).str[0:2]
holdout['area'] = holdout['Pin'].astype(str).str[0:2]
df_final['subarea'] = df_final['Pin'].astype(str).str[2:4]
holdout['subarea'] = holdout['Pin'].astype(str).str[2:4]
df_final['lat_head']=df_final['Latitude'].astype(str).str[0:5]
holdout['lat_head']=holdout['Latitude'].astype(str).str[0:5]
df_final['ratio_lb']=df_final['BuildingSquareFeet']/df_final['LandSquareFeet']
holdout['ratio_lb']=holdout['BuildingSquareFeet']/holdout['LandSquareFeet']
df_final['ln_le']=np.log(df_final['EstimateLand'])
holdout['ln_le']=np.log(holdout['EstimateLand'])
df_final['sqr_bdrms']=df_final['Bedrooms']**2
holdout['sqr_bdrms']=holdout['Bedrooms']**2
df_final['ln_te']=np.log(df_final['TotalEstimate'])
holdout['ln_te']=np.log(holdout['TotalEstimate'])
df_final.corr().sort_values(by=['SalePrice'],ascending=False).head(20)
final_categ=['MostRecentSale','TownshipLocation','TownshipName','RoadProximity','TypeOfResidence','SiteDesirability','Basement',
            'BasementFinish','CentralHeating','OtherHeating','CentralAir','AtticType','WallMaterial','RoofMaterial','CathedralCeiling','DesignPlan','ConstructionQuality',
            'Porch','RepairCondition','GarageIndicator','Garage1Attachment','Garage1Area','Garage2Attachment','year','month','lat_head','area','NeighborhoodCode','subarea']
for i in final_categ:
    print(i,df_final[i].nunique(),holdout[i].nunique())
for i in final_categ:
    df_final_categ=pd.DataFrame(pd.get_dummies(df_final[final_categ],drop_first=True))
    holdout_final_categ=pd.DataFrame(pd.get_dummies(holdout[final_categ],drop_first=True))
print(df_final_categ.shape,holdout_final_categ.shape)
cont_vars=['TotalEstimate','EstimateBuilding','Garage1Size','GarageSize','Rooms','Bedrooms','BuildingSquareFeet','FullBaths','HalfBaths','Fireplaces','NumberOfCommercialUnits','MultiCode',
          'Apartments','ratio_lb','ln_le']
final_df=pd.concat([df_final_categ,df_final[cont_vars]],axis=1)
final_holdout=pd.concat([holdout_final_categ,holdout[cont_vars]],axis=1)                   
final_df.shape,final_holdout.shape
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


X=final_df
y=df_final['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

import xgboost
model=xgboost.XGBRegressor()
model.fit(X_train,y_train)
y_pred_xg=(model.predict(X_test))
y_pred_nc=model.predict(final_holdout)
pred_nc = pd.DataFrame(np.c_[holdout.Id.astype(str), (y_pred_nc)], columns = ['Id', 'SalePrice'])
print(sim_condos.shape,holdout_condos.shape)
for i in sim_condos.columns:
    if sim_condos[i].isna().sum()>0.75*len(sim_condos):
        sim_condos.drop([i],axis=1,inplace=True)
        holdout_condos.drop([i],axis=1,inplace=True)
sim_condos.isna().sum()
condo_miss=['MostRecentSale','Longitude','Latitude','OHareNoise','Floodplain','RoadProximity','CensusTract']

for c in condo_miss:
    sim_condos[c]=sim_condos[c].fillna(sim_condos[c].mode()[0])
    holdout_condos[c]=holdout_condos[c].fillna(holdout_condos[c].mode()[0])
sim_condos.isna().sum()
sim_condos.drop(['PropertyAddress'],axis=1,inplace=True)
holdout_condos.drop(['PropertyAddress'],axis=1,inplace=True)
sim_condos['month'].value_counts(normalize=True)
sim_condos['area']=sim_condos['Pin'].astype(str).str[0:2]
sim_condos['subarea']=sim_condos['Pin'].astype(str).str[2:4]

holdout_condos['area']=holdout_condos['Pin'].astype(str).str[0:2]
holdout_condos['subarea']=holdout_condos['Pin'].astype(str).str[2:4]


sim_condos['totalEstimate']=sim_condos['EstimateLand']+sim_condos['EstimateBuilding']
holdout_condos['totalEstimate']=holdout_condos['EstimateLand']+holdout_condos['EstimateBuilding']

sim_condos['ln_LE']=np.log(sim_condos['EstimateLand'])
holdout_condos['ln_LE']=np.log(holdout_condos['EstimateLand'])
sim_condos.corr()
condo_categ=['MostRecentSale','RoadProximity','Porch','year','month','TownshipName','TownshipLocation','area','subarea']

for i in condo_categ:
    sim_condo_categ=pd.DataFrame(pd.get_dummies(sim_condos[condo_categ],drop_first=True))
    holdout_condo_categ=pd.DataFrame(pd.get_dummies(holdout_condos[condo_categ],drop_first=True))
sim_condo_categ.shape, holdout_condo_categ.shape
condo_cont=['EstimateLand','CondoPercentOwnership','LandSquareFeet','totalEstimate','EstimateBuilding','ln_LE']
sim_condo_final=pd.concat([sim_condos[condo_cont],sim_condo_categ,sim_condos['SalePrice']],axis=1)
holdout_condo_final=pd.concat([holdout_condos[condo_cont],holdout_condo_categ],axis=1)
print(sim_condo_final.shape,holdout_condo_final.shape)
sim_condo_final_copy=sim_condo_final[sim_condo_final['SalePrice']>15000]
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


X=sim_condo_final_copy.drop(['SalePrice'],axis=1)
y=(sim_condo_final_copy['SalePrice'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

import xgboost
model=xgboost.XGBRegressor()
model.fit(X_train,y_train)

y_pred_c=model.predict(holdout_condo_final)
X=sim_condo_final_copy.drop(['SalePrice'],axis=1)
y=np.log(sim_condo_final_copy['SalePrice'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from catboost import CatBoostRegressor
model = CatBoostRegressor(iterations=500,
                          learning_rate=1,
                          depth=5)
model.fit(X_train,y_train)
y_pred_2=model.predict(holdout_condo_final)
r2_score(y_test,y_pred_2)
pred_c_exp=pd.DataFrame(np.c_[holdout_condos.Id.astype(str), y_pred_2], columns = ['Id', 'SalePrice'])
submission=pd.concat([pred_nc,pred_c_exp])
submission.shape
submission.to_csv('_s_tarter_submissions.csv', index = False, header = True)