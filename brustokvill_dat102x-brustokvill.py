%matplotlib inline
import pandas as pd
pd.options.display.max_columns=100

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pylab as plot
params={
    'axes.labelsize':"large",
    'xtick.labelsize':'x-large',
    'legend.fontsize':20,
    'figure.dpi':150,
    'figure.figsize':[25,7]
}
plot.rcParams.update(params)
#ignore warnings
import warnings 
warnings.filterwarnings('ignore')
#read the training values
x=pd.read_csv('../input/DAT102x_Predicting_Heart_Disease_Mortality_-_Training_values.csv',index_col=0)
#read training labels
y=pd.read_csv('../input/DAT102x_Predicting_Heart_Disease_Mortality_-_Training_labels.csv',index_col=0)
data=pd.concat([x,y],axis=1)
x.shape
x.columns.tolist()
x.dtypes
#A list that record the name of numerical features
numerical_features=x.dtypes[x.dtypes=='float64'].index.tolist()
x.describe().transpose()
x.isnull().sum()
x['area__rucc'].value_counts()
x['area__urban_influence'].value_counts()
x['econ__economic_typology'].value_counts()
x['yr'].value_counts()
y.head(3)
y.describe()
#For a normal traing_lables, there should not exist any missing value.
#Thus, this is just for checking.
y.isnull().sum()
sns.distplot(y['heart_disease_mortality_per_100k'],
             label="Skewness : %.2f"%(y['heart_disease_mortality_per_100k']\
                                      .skew())).legend(loc="best")
f, axes = plt.subplots(10, 3,figsize=(6*2,20*2))
plt.subplots_adjust( wspace=0.35, hspace=0.35)

for i,ax in zip(numerical_features,axes.flat):
        
        ax.scatter(x = x[i], y = y['heart_disease_mortality_per_100k'],s=1.5)
        ax.set_xlabel(i,size=5.5*2)
        ax.set_ylabel('heart_disease_mortality_per_100k',size=5.5*2)
        ax.tick_params(labelsize=8)
        
temp=data.corr().loc[['heart_disease_mortality_per_100k']]
temp.transpose().sort_values(by='heart_disease_mortality_per_100k', ascending=False)
sns.set(font_scale = 1.6)
g=sns.violinplot(x='area__rucc',y='heart_disease_mortality_per_100k',hue='yr',data=data)
g1=g.set_xticklabels(g.get_xticklabels(),rotation=90)
sns.set(font_scale = 1.6)
g=sns.violinplot(x='area__urban_influence',y='heart_disease_mortality_per_100k',hue='yr',data=data)
g1=g.set_xticklabels(g.get_xticklabels(),rotation=90)
sns.set(font_scale = 1.6)
g=sns.violinplot(x='econ__economic_typology',y='heart_disease_mortality_per_100k',hue='yr',data=data)
g1=g.set_xticklabels(g.get_xticklabels(),rotation=90)
x.drop('health__homicides_per_100k',axis=1,inplace=True)
x['econ__pct_uninsured_adults']=x['econ__pct_uninsured_adults'].fillna(x['econ__pct_uninsured_adults'].mean())
x['econ__pct_uninsured_children']=x['econ__pct_uninsured_children'].fillna(x['econ__pct_uninsured_children'].mean())
x['demo__pct_female']=x['demo__pct_female'].fillna(x['demo__pct_female'].mean())
x['demo__pct_below_18_years_of_age']=x['demo__pct_below_18_years_of_age'].fillna(x['demo__pct_below_18_years_of_age'].mean())
x['demo__pct_aged_65_years_and_older']=x['demo__pct_aged_65_years_and_older'].fillna(x['demo__pct_aged_65_years_and_older'].mean())
x['demo__pct_hispanic']=x['demo__pct_hispanic'].fillna(x['demo__pct_hispanic'].mean())
x['demo__pct_non_hispanic_african_american']=x['demo__pct_non_hispanic_african_american'].fillna(x['demo__pct_non_hispanic_african_american'].mean())
x['demo__pct_non_hispanic_white']=x['demo__pct_non_hispanic_white'].fillna(x['demo__pct_non_hispanic_white'].mean())
x['demo__pct_american_indian_or_alaskan_native']=x['demo__pct_american_indian_or_alaskan_native'].fillna(x['demo__pct_american_indian_or_alaskan_native'].mean())
x['demo__pct_asian']=x['demo__pct_asian'].fillna(x['demo__pct_asian'].mean())
x['health__pct_adult_obesity']=x['health__pct_adult_obesity'].fillna(x['health__pct_adult_obesity'].mean())
x['health__pct_diabetes']=x['health__pct_diabetes'].fillna(x['health__pct_diabetes'].mean())
x['health__pct_physical_inacticity']=x['health__pct_physical_inacticity'].fillna(x['health__pct_physical_inacticity'].mean())
#Having large number of missing values
x['health__pct_adult_smoking']=x['health__pct_adult_smoking'].fillna(x['health__pct_adult_smoking'].median())
x['health__pct_low_birthweight']=x['health__pct_low_birthweight'].fillna(x['health__pct_low_birthweight'].median())
x['health__pct_excessive_drinking']=x['health__pct_excessive_drinking'].fillna(x['health__pct_excessive_drinking'].mean())
x['health__air_pollution_particulate_matter']=x['health__air_pollution_particulate_matter'].fillna(x['health__air_pollution_particulate_matter'].median())
x['health__motor_vehicle_crash_deaths_per_100k']=x['health__motor_vehicle_crash_deaths_per_100k'].fillna(x['health__motor_vehicle_crash_deaths_per_100k'].mean())
x['health__pop_per_dentist']=x['health__pop_per_dentist'].fillna(x['health__pop_per_dentist'].median())
x['health__pop_per_primary_care_physician']=x['health__pop_per_primary_care_physician'].fillna(x['health__pop_per_primary_care_physician'].median())

#Because I forget that I should fill NaNs in 'data' not in 'x', I definite the 'data' again.
data=pd.concat([x,y],axis=1)
data=data.drop(data[data['econ__pct_unemployment']>0.20].index)
data=data.drop(data[(data['demo__pct_below_18_years_of_age']>0.40)&(data['heart_disease_mortality_per_100k']>500)].index)
data=data.drop(data[(data['health__pct_adult_obesity']>0.40)&(data['heart_disease_mortality_per_100k']<230)].index)
data=data.drop(data[(data['health__pct_low_birthweight']>0.20)&(data['heart_disease_mortality_per_100k']<200)].index)
data=data.drop(data[(data['health__motor_vehicle_crash_deaths_per_100k']>100)&(data['heart_disease_mortality_per_100k']<350)].index)
def process_area__rucc(data):
    area_rucc_dummies=pd.get_dummies(data['area__rucc'],prefix="area__rucc")
    
    data=pd.concat([data,area_rucc_dummies],axis=1)
    data.drop('area__rucc',axis=1,inplace=True)
    
    return data
data=process_area__rucc(data)

    
def process_area__urban_influence(data):
    area__urban_influence_dummies=pd.get_dummies(data['area__urban_influence'],prefix="area__urban_influence")
    
    data=pd.concat([data,area__urban_influence_dummies],axis=1)
    data.drop('area__urban_influence',axis=1,inplace=True)
    
    return data

data=process_area__urban_influence(data)
def process_econ__economic_typology(data):
    econ__economic_typology_dummies=pd.get_dummies(data['econ__economic_typology'],prefix="econ__economic_typology")
    
    data=pd.concat([data,econ__economic_typology_dummies],axis=1)
    data.drop('econ__economic_typology',axis=1,inplace=True)
    
    return data

data=process_econ__economic_typology(data)
def process_yr(data):
    yr_dummies=pd.get_dummies(data['yr'],prefix="yr")
    data=pd.concat([data,yr_dummies],axis=1)
    data.drop('yr',axis=1,inplace=True)
    return data
data=process_yr(data)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

scalers=[]
X={}
temp=numerical_features.copy()
temp.remove('health__homicides_per_100k')

scalers.append(MinMaxScaler())
scalers.append(MaxAbsScaler())
scalers.append(StandardScaler())
scalers.append(RobustScaler())
scalers.append(Normalizer())
for scaler in scalers:
    tdata=data.copy()
    tdata[temp]=scaler.fit_transform(tdata[temp])
    X[str(scaler)]=tdata
del temp
#data=X[str(scalers[1])]
x=data.drop('heart_disease_mortality_per_100k',axis=1)
y=data[['heart_disease_mortality_per_100k']]
from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
x_train,y_train=x,y

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor

import os
#I import this path because of Xgboost. You should change this path to your own, or you can just skip the part of Xgboost.
mingw_path = 'X:\\Program Files X\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
from sklearn.linear_model import LinearRegression,LassoCV, Ridge, LassoLarsCV,ElasticNetCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
def compute_score(model,x,y,scoring='neg_mean_squared_error'):
    y=y.values.ravel()
    xval=cross_val_score(model,x,y,cv=5,scoring=scoring)
    return xval
# turn run_gs to True if you want to run the gridsearch again.
run_gs= False

if run_gs:
    parameter_grid={
                 'max_depth' : [40],
                 'n_estimators': [200,250],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2 ],
                 'min_samples_leaf': [1],
                 'bootstrap': [True, False],}

    forest=RandomForestRegressor()
    cross_validation=StratifiedKFold(n_splits=5)

    grid_search=GridSearchCV(forest,
                            scoring='neg_mean_squared_error',
                            param_grid=parameter_grid,
                            cv=cross_validation,
                            verbose=1)
    grid_search.fit(x_train,y_train.values.ravel())

    model= grid_search
    parameters=grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

else:
    parameters = {'bootstrap': False, 'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    RFG_model=RandomForestRegressor(**parameters)
    RFG_model.fit(x_train,y_train)

    print(compute_score(RFG_model,x_train,y_train))
#mean_squared_error(y_test,RFG_model.predict(x_test))
features = pd.DataFrame()
features['feature'] = x_train.columns
features['importance'] = RFG_model.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features=features.iloc[-10:,:]
f, ax = plt.subplots(1, 1,figsize=(5,5))
features.set_index('feature', inplace=True)
features.plot(kind='barh', ax=ax,fontsize=5)
ax.legend(prop={'size': 6})
RFG_top10=features.index.tolist()
run_gs= False

if run_gs:
    parameter_grid={
                 
                 'max_depth' : [20],
                 'min_child_weight': [4,6,8],
                 'gamma': [0.3,0.5,0.7 ],
                                    }

    forest=xgb.XGBRegressor()
    cross_validation=StratifiedKFold(n_splits=5)

    grid_search=GridSearchCV(forest,
                            scoring='neg_mean_squared_error',
                            param_grid=parameter_grid,
                            cv=cross_validation,
                            verbose=1)

    grid_search.fit(x_train,y_train.values.ravel())
    model= grid_search

    parameters=grid_search.best_params_


    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

else:
    parameters = {'gamma': 0.5, 'max_depth': 20, 'min_child_weight': 6}
    xgb_model=xgb.XGBRegressor()
    xgb_model.fit(x_train,y_train)
print(compute_score(xgb_model,x_train,y_train))
#mean_squared_error(y_test,xgb_model.predict(x_test))
features = pd.DataFrame()
features['feature'] = x_train.columns
features['importance'] = xgb_model.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features=features.iloc[-10:,:]
f, ax = plt.subplots(1, 1,figsize=(5,5))
features.set_index('feature', inplace=True)
features.plot(kind='barh', ax=ax,fontsize=5)
ax.legend(prop={'size': 6})
xgb_top10=features.index.tolist()
run_gs= False

if run_gs:
    parameter_grid={
                 'max_depth' : [10,20,40],
                 'n_estimators': [100,200,300],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'learning_rate':[0.01,0.05]               
                                    }

    forest=GradientBoostingRegressor()
    cross_validation=StratifiedKFold(n_splits=5)

    grid_search=GridSearchCV(forest,
                            scoring='neg_mean_squared_error',
                            param_grid=parameter_grid,
                            cv=cross_validation,
                            verbose=1)

    grid_search.fit(x_train,y_train.values.ravel())
    model= grid_search

    parameters=grid_search.best_params_


    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

else:
    parameters =  {'learning_rate': 0.05, 'max_depth': 10, 'max_features': 'log2', 'n_estimators': 300}

    GBoost=GradientBoostingRegressor(**parameters)
    GBoost.fit(x_train,y_train)
print(compute_score(GBoost,x_train,y_train))
#mean_squared_error(y_test,GBoost.predict(x_test))
features = pd.DataFrame()
features['feature'] = x_train.columns
features['importance'] = GBoost.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features=features.iloc[-10:,:]
f, ax = plt.subplots(1, 1,figsize=(5,5))
features.set_index('feature', inplace=True)
features.plot(kind='barh', ax=ax,fontsize=5)
ax.legend(prop={'size': 6})

GBoost_top10=features.index.tolist()
run_gs= False

if run_gs:
    parameter_grid={
                 'max_depth' : [-1,20,40],
                 'min_data_in_leaf' :[20,30],
                 'n_estimators' :[800],
                 'learning_rate' :[0.05,0.03]               
                                    }

    forest=lgb.LGBMRegressor()
    cross_validation=StratifiedKFold(n_splits=5)

    grid_search=GridSearchCV(forest,
                            scoring='neg_mean_squared_error',
                            param_grid=parameter_grid,
                            cv=cross_validation,
                            verbose=1)

    grid_search.fit(x_train,y_train.values.ravel())
    model= grid_search

    parameters=grid_search.best_params_


    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

else:
    parameters =  {'learning_rate': 0.05, 'max_depth': 20, 'min_data_in_leaf': 20, 'n_estimators': 800}

    LGBMR=lgb.LGBMRegressor(**parameters)
    LGBMR.fit(x_train,np.array(y_train.iloc[:,0]))
print(compute_score(LGBMR,x_train,y_train))
#mean_squared_error(y_test,LGBMR.predict(x_test))
features = pd.DataFrame()
features['feature'] = x_train.columns
features['importance'] = LGBMR.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features=features.iloc[-10:,:]
f, ax = plt.subplots(1, 1,figsize=(5,5))
features.set_index('feature', inplace=True)
features.plot(kind='barh', ax=ax,fontsize=5)
ax.legend(prop={'size': 6})
LGBMR_top10=features.index.tolist()
#Find the intersection of the four top10 important feature
s1=set(RFG_top10)
s2=set(xgb_top10)
s3=set(GBoost_top10)
s4=set(LGBMR_top10)
print(s1&s2&s3&s4)

del s1,s2,s3,s4
lassocv = LassoCV(eps=1e-7) 
lassocv.fit(x_train,y_train.values.ravel())
#mean_squared_error(y_test,lassocv.predict(x_test))
print(compute_score(lassocv,x_train,y_train))
ridge = Ridge(alpha=1e-8) 
ridge.fit(x_train,y_train.values.ravel())
#mean_squared_error(y_test,ridge.predict(x_test))
print(compute_score(ridge,x_train,y_train))
lassolarscv = LassoLarsCV(eps=1e-7)
lassolarscv.fit(x_train,y_train.values.ravel())
#mean_squared_error(y_test,lassolarscv.predict(x_test))

print(compute_score(lassolarscv,x_train,y_train))
elasticnetcv = ElasticNetCV(eps=1e-10)
elasticnetcv.fit(x_train,y_train.values.ravel())
#mean_squared_error(y_test,elasticnetcv.predict(x_test))
print(compute_score(elasticnetcv,x_train,y_train))
#linear_predict=(lassocv.predict(x_test)+ridge.predict(x_test).ravel()+lassolarscv.predict(x_test)+elasticnetcv.predict(x_test))/4
#mean_squared_error(y_test,linear_predict)
#tree_predict=(GBoost.predict(x_test)*0.3+xgb_model.predict(x_test)*0.1+RFG_model.predict(x_test)*0.3+LGBMR.predict(x_test)*0.3)
#mean_squared_error(y_test,tree_predict)
#mean_squared_error(y_test,tree_predict*0.9+linear_predict*0.1)
x=pd.read_csv('../input/DAT102x_Predicting_Heart_Disease_Mortality_-_Test_values.csv',index_col=0)
x.drop('health__homicides_per_100k',axis=1,inplace=True)
x['econ__pct_uninsured_adults']=x['econ__pct_uninsured_adults'].fillna(x['econ__pct_uninsured_adults'].mean())
x['econ__pct_uninsured_children']=x['econ__pct_uninsured_children'].fillna(x['econ__pct_uninsured_children'].mean())
x['demo__pct_female']=x['demo__pct_female'].fillna(x['demo__pct_female'].mean())
x['demo__pct_below_18_years_of_age']=x['demo__pct_below_18_years_of_age'].fillna(x['demo__pct_below_18_years_of_age'].mean())
x['demo__pct_aged_65_years_and_older']=x['demo__pct_aged_65_years_and_older'].fillna(x['demo__pct_aged_65_years_and_older'].mean())
x['demo__pct_hispanic']=x['demo__pct_hispanic'].fillna(x['demo__pct_hispanic'].mean())
x['demo__pct_non_hispanic_african_american']=x['demo__pct_non_hispanic_african_american'].fillna(x['demo__pct_non_hispanic_african_american'].mean())
x['demo__pct_non_hispanic_white']=x['demo__pct_non_hispanic_white'].fillna(x['demo__pct_non_hispanic_white'].mean())
x['demo__pct_american_indian_or_alaskan_native']=x['demo__pct_american_indian_or_alaskan_native'].fillna(x['demo__pct_american_indian_or_alaskan_native'].mean())
x['demo__pct_asian']=x['demo__pct_asian'].fillna(x['demo__pct_asian'].mean())
x['health__pct_adult_obesity']=x['health__pct_adult_obesity'].fillna(x['health__pct_adult_obesity'].mean())
x['health__pct_diabetes']=x['health__pct_diabetes'].fillna(x['health__pct_diabetes'].mean())
x['health__pct_physical_inacticity']=x['health__pct_physical_inacticity'].fillna(x['health__pct_physical_inacticity'].mean())
#Having large number of missing values
x['health__pct_adult_smoking']=x['health__pct_adult_smoking'].fillna(x['health__pct_adult_smoking'].median())
x['health__pct_low_birthweight']=x['health__pct_low_birthweight'].fillna(x['health__pct_low_birthweight'].median())
x['health__pct_excessive_drinking']=x['health__pct_excessive_drinking'].fillna(x['health__pct_excessive_drinking'].mean())
x['health__air_pollution_particulate_matter']=x['health__air_pollution_particulate_matter'].fillna(x['health__air_pollution_particulate_matter'].median())
x['health__motor_vehicle_crash_deaths_per_100k']=x['health__motor_vehicle_crash_deaths_per_100k'].fillna(x['health__motor_vehicle_crash_deaths_per_100k'].mean())
x['health__pop_per_dentist']=x['health__pop_per_dentist'].fillna(x['health__pop_per_dentist'].median())
x['health__pop_per_primary_care_physician']=x['health__pop_per_primary_care_physician'].fillna(x['health__pop_per_primary_care_physician'].median())
x=process_area__rucc(x)
x=process_area__urban_influence(x)
x=process_econ__economic_typology(x)
x=process_yr(x)
x_test=x
linear_predict=(lassocv.predict(x_test)+ridge.predict(x_test).ravel()+lassolarscv.predict(x_test)+elasticnetcv.predict(x_test))/4

#By the score, I decide not to add Xgboost in the ensemble.
tree_predict=(GBoost.predict(x_test)+RFG_model.predict(x_test)+LGBMR.predict(x_test))/3
#tree_predict=(GBoost.predict(x_test)*0.3+xgb_model.predict(x_test)*0.1+RFG_model.predict(x_test)*0.3+LGBMR.predict(x_test)*0.3)

output = tree_predict*0.5+linear_predict*0.5
output=output.astype(int)
sns.distplot(output)
df_output = pd.DataFrame()
aux = pd.read_csv('../input/DAT102x_Predicting_Heart_Disease_Mortality_-_Test_values.csv')
df_output['row_id'] = aux['row_id']
df_output['heart_disease_mortality_per_100k'] = output
df_output[['row_id','heart_disease_mortality_per_100k']].to_csv('heart_disease_mortality_per_100k.csv', index=False)