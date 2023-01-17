import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('default')

import numpy as np

from numpy import random

import warnings

warnings.filterwarnings('ignore')



from sklearn import metrics,preprocessing

from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn import svm

from sklearn import model_selection

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

import warnings

# more Classifiers

## 

import sys

from itertools import product

from sklearn.tree import DecisionTreeRegressor



from IPython.display import display

from scipy.stats import skew

import lightgbm as lgb

import os

from tqdm import tqdm

#from pandas_profiling import ProfileReport

from sklearn.cluster import KMeans

from sklearn import preprocessing
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

            

train.head()
catcols=train.select_dtypes(include='object').columns.values.tolist()

catcols.remove('Date')

numcols=train.select_dtypes(include='number').columns.values[1:-1].tolist()



#

train[catcols].describe().merge(test[catcols].describe(),left_index=True,right_index=True,suffixes=('_train','_test'))
# Missing values

pd.concat([round(100*train[catcols].isnull().sum()/train.shape[0],2).to_frame('train'),

           round(100*test[catcols].isnull().sum()/test.shape[0],2).to_frame('test')],axis=1)
print(f'TRAIN -> date_min= {train["Date"].min()} ; date_max= {train["Date"].max()}')

print(f'TEST -> date_min= {test["Date"].min()} ; date_max= {test["Date"].max()}')
len(set(train.Date.unique()) & set(test.Date.unique()))
def prepare_features(data):

    # lower Province_State & Country_Region in order to use them to add more information

    data['Province_State']=data['Province_State'].str.lower()

    data['Country_Region']=data['Country_Region'].str.lower()

    

    # Create a new feature = weither the Province_State is known or not

    data['UnkownProvince_State']=data['Province_State'].isnull().astype(int)

    

    # Fill missing Province_State & Country_Region missing values 

    data.fillna({'Province_State':''},inplace=True)



    

    # Remove non-alpha charachters 

    data['Province_State']=data['Province_State'].apply(lambda x: ''.join([ch for ch in x if ch.isalpha()]))

    data['Country_Region']=data['Country_Region'].apply(lambda x: ''.join([ch for ch in x if ch.isalpha()]))

    

    # Create a new feature = Country_Region frequency

    data['Country_RegionFreq']=data['Country_Region'].map(data['Country_Region'].value_counts(1).to_dict())

    

    return data



catcols.append('UnkownProvince_State')
# clean & transform features

train=prepare_features(train)

test=prepare_features(test)

#

train.head(3)
train['Country_Region'].nunique()
cols_to_keep=['Country Name','Population ages 0-14, total', 'Population ages 15-64, female','Population ages 15-64, male',

'Population ages 15-64, total','Population ages 65 and above, total',

'Population ages 80 and above, female (% of female population)',

'Population ages 80 and above, male (% of male population)','Population, male',

'Population, total','Rural population (% of total population)']



add_inf=pd.read_csv('/kaggle/input/world-population-and-development-indicators/data.csv')[cols_to_keep]

add_inf.head(3)
add_inf['Country Name'].replace({'bahamasthe':'bahamas','bruneidarussalam':'bahamas','czechrepublic':'czechia',

                   'congodemrep':'congokinshasa','congorep':'congobrazzaville','egyptarabrep':'egypt',

                  'gambiathe':'gambia','iranislamicrep':'iran','korearep':'koreasouth','unitedstates':'us',

                   'kyrgyzrepublic':'kyrgyzstan','russianfederation':'russia','stkittsandnevis':'saintkittsandnevis',

                  'stlucia':'saintlucia','stvincentandthegrenadines':'saintvincentandthegrenadines',

                  'slovakrepublic':'slovakia','syrianarabrepublic':'syria','venezuelarb':'venezuela'},inplace=True)

print('us' in add_inf['Country Name'].unique())



add_inf.drop_duplicates(subset=['Country Name'],inplace=True)

print(add_inf['Country Name'].nunique())
print('From train:{}  ;  From add_inf:{}  ; intersection:{}'.format(train['Country_Region'].nunique(),add_inf['Country Name'].nunique(),

                                                                    len(set(train['Country_Region']) & set(add_inf['Country Name']))))



print('From test:{}  ;  From add_inf:{}  ; intersection:{}'.format(test['Country_Region'].nunique(),add_inf['Country Name'].nunique(),

                                                                    len(set(test['Country_Region']) & set(add_inf['Country Name']))))
def scale_popultaion_features(data):

    # replace 0 with a more probable value

    data['Population ages 0-14, total'].replace(0,data['Population ages 0-14, total'].mode(),inplace=True)

    data['Population ages 15-64, total'].replace(0,data['Population ages 15-64, total'].mode(),inplace=True)

    data['Population ages 65 and above, total'].replace(0,data['Population ages 65 and above, total'].mode(),inplace=True)

    data['Population, total'].replace(0,data['Population, total'].mode(),inplace=True)

    

    # Normalize

    data['Population ages 15-64, male']=data['Population ages 15-64, male']/data['Population ages 15-64, total']

    data['Population ages 15-64, female']=data['Population ages 15-64, female']/data['Population ages 15-64, total']

    #

    data['Population ages 0-14, total']=data['Population ages 0-14, total']/data['Population, total']

    data['Population ages 15-64, total']=data['Population ages 15-64, total']/data['Population, total']    

    data['Population ages 65 and above, total']=data['Population ages 65 and above, total']/data['Population, total'] 

    #

    data['Population, male']=data['Population, male']/data['Population, total'] 

    

    # rescale % features (bring them back to [0,1] interval)

    pcq_features=[col for col in data.columns if '%' in col]

    print(pcq_features)

    data[pcq_features]=.01*data[pcq_features]

    data.drop(columns=['Population, total'],inplace=True)

    # fillna

    data.fillna(data.median().to_dict(),inplace=True)

    return data
add_inf=scale_popultaion_features(add_inf)

add_inf.describe()
set(train['Country_Region'])-set(add_inf['Country Name'])
# (1) add world population & developement indicators 

print('train shape: {}  ;  test shape: {}'.format(train.shape,test.shape))

#

train=train.merge(add_inf,left_on='Country_Region',right_on='Country Name',how='left')

test=test.merge(add_inf,left_on='Country_Region',right_on='Country Name',how='left')

#

print('train shape: {}  ;  test shape: {}'.format(train.shape,test.shape))



train.head()
cols_to_keep=['country', 'total_covid_19_tests',

       'total_covid_19_tests_per_million_people',

       'inform_risk', 'inform_p2p_hazard_and_exposure_dimension',

       'people_using_at_least_basic_sanitation_services',

       'inform_vulnerability', 'inform_health_conditions',

       'inform_epidemic_vulnerability', 'mortality_rate_under_5',

       'prevalence_of_undernourishment', 'inform_lack_of_coping_capacity',

       'inform_access_to_healthcare',

       'inform_epidemic_lack_of_coping_capacity', 'physicians_density',

       'current_health_expenditure_per_capita',

       'maternal_mortality_ratio', 'entry_date', 'category',

       'measure', 'global-school-closures']



who_data=pd.read_csv('/kaggle/input/whodata/WHO data.csv')[cols_to_keep]

who_data.drop_duplicates(subset=['country'],inplace=True)

print(who_data.shape)

who_data.head(2)
who_data['country'].replace({'capeverde':'caboverde','czechrepublic':'czechia','myanmar':'burma',

                            'congodemrep':'congokinshasa','congorep':'congobrazzaville','guinea':'guineabissau',

                            'swaziland':'eswatini','southkorea':'koreasouth','macedonia':'northmacedonia',

                            'timor':'timorleste','unitedstates':'us','unitedstatesvirginislands':'us',

                            'vatican':'holysee','palestine':'westbankandgaza'},inplace=True)



print(who_data['country'].nunique())

who_data.drop_duplicates(subset=['country'],inplace=True)
print('From train:{}  ;  From who-data:{}  ; intersection:{}'.format(train['Country_Region'].nunique(),who_data['country'].nunique(),

                                                                    len(set(train['Country_Region']) & set(who_data['country']))))



print('From test:{}  ;  From who-data:{}  ; intersection:{}'.format(test['Country_Region'].nunique(),who_data['country'].nunique(),

                                                                    len(set(test['Country_Region']) & set(who_data['country']))))
# Missing values

who_data.select_dtypes(include='number').isnull().sum()/who_data.shape[0]
# (2) add WHO health system information 

print('train shape: {}  ;  test shape: {}'.format(train.shape,test.shape))



train=train.merge(who_data,left_on='Country_Region',right_on='country',how='left')

test=test.merge(who_data,left_on='Country_Region',right_on='country',how='left')



# fillna

train.fillna(train.quantile(.15).to_dict(),inplace=True)

test.fillna(train.quantile(.15).to_dict(),inplace=True)



print('train shape: {}  ;  test shape: {}'.format(train.shape,test.shape))



train.head(3)
# convert Date into datetime format 

train['Date'] = pd.to_datetime(train['Date'], format = '%Y-%m-%d')

test['Date'] = pd.to_datetime(test['Date'], format = '%Y-%m-%d')
# Extract Date characteristics

def create_date_features(df):

    df['day'] = df['Date'].dt.day

    df['month'] = df['Date'].dt.month

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['weekofyear'] = df['Date'].dt.weekofyear

    df['Date_day_month'] = df['Date'].dt.strftime("%m%d").astype(int)

    return df

# 

train=create_date_features(train)

test=create_date_features(test)



# 

test.head(3)
# concatenate Province_State & Country_Region as "Province_State" ID

train['Province_State']=train['Country_Region']+' '+train['Province_State']

test['Province_State']=test['Country_Region']+' '+test['Province_State']



#

train['Province_State']=train['Province_State'].str.replace(' ','')

test['Province_State']=test['Province_State'].str.replace(' ','')





# Drop useless columns

train.drop(columns=['Country Name','Country_Region','country'],inplace=True)

test.drop(columns=['Country Name','Country_Region','country'],inplace=True)

#

train.head(2)
# weither the added mesures were applied at Date or not

train['measures_applied']=(train.Date.dt.strftime('%Y-%m-%d')>=train['entry_date']).astype(int)

test['measures_applied']=(test.Date.dt.strftime('%Y-%m-%d')>=test['entry_date']).astype(int)

#

train.drop(columns=['entry_date'],inplace=True)

test.drop(columns=['entry_date'],inplace=True)

#

train.head(3)
# encode categorical features

train=pd.get_dummies(columns=['category','measure','global-school-closures'],data=train)

test=pd.get_dummies(columns=['category','measure','global-school-closures'],data=test)

#

train.head(3)
# Display ConfirmedCases & Fatatilities charts for a random Province state



# Pick one random Province_State

province=np.random.choice(train['Province_State'].unique())

s=train.loc[train['Province_State']==province,['ConfirmedCases','Fatalities']]



plt.style.use('default')

plt.figure(figsize=(10,3))

plt.subplot(121)

s['ConfirmedCases'].plot(kind='area',color='deepskyblue',alpha=.4,label='ConfirmedCases')

s['Fatalities'].plot(kind='area',color='orangered',alpha=.4,label='Fatalities')

plt.legend()

plt.xticks(rotation=80)

plt.title('ConfirmedCases',fontsize=10)

#

plt.subplot(122)

s['Fatalities'].plot('area',color='orange',alpha=.4)

plt.xticks(rotation=80)

plt.title('Fatalities',fontsize=10)



plt.suptitle(province.upper(),fontsize=14)

plt.show()
# Show target (ConfirmedCases, Fatalities)

plt.figure(figsize=(10,3))

plt.subplot(121)

plt.hist(np.log(1+train['ConfirmedCases']),bins=100,edgecolor='k',facecolor='deepskyblue',density=True)

plt.title('log ConfirmedCases')

#

plt.subplot(122)

plt.hist(np.log(1+train['Fatalities']),bins=100,edgecolor='k',facecolor='orangered',density=True)

plt.title('log Fatalities')

plt.show()
s=test.select_dtypes(include='number').nunique()

binary_features=s[s==2].index.values

binary_features
# visualize binary features ditribution

def feat_pie(col):

    print('ConfirmedCases*{}  Correlation = {}'.format(col,np.corrcoef(train['ConfirmedCases'],train[col])[0,1]))

    print('Fatalities*{}  Correlation = {}'.format(col,np.corrcoef(train['Fatalities'],train[col])[0,1]))

    

    plt.figure(figsize=(7,3))

    plt.subplot(121)

    train[col].value_counts().plot.pie(autopct='%1.1f%%')

    plt.title('train')

    plt.subplot(122)

    test[col].value_counts().plot.pie(autopct='%1.1f%%')

    plt.title('test')

    plt.show()

    

feat_pie(np.random.choice(binary_features))
# fill binary data with 0

train.fillna({col:0 for col in binary_features},inplace=True)

test.fillna({col:0 for col in binary_features},inplace=True)



# fill missing values with mean-value

filling_dict=train.median().to_dict()

train.fillna(filling_dict,inplace=True)

test.fillna(filling_dict,inplace=True)

train.head(2)
features=test.select_dtypes(include='number').columns.values[1:]

features.shape
def feat_dist(col):

    print('ConfirmedCases*{}  Correlation = {}'.format(col,np.corrcoef(train['ConfirmedCases'],train[col])[0,1]))

    print('Fatalities*{}  Correlation = {}'.format(col,np.corrcoef(train['Fatalities'],train[col])[0,1]))

    

    plt.figure(figsize=(6,3))

    train[col].plot.hist(bins=100,edgecolor='k',facecolor='deepskyblue')

    plt.title(col)

    plt.show()

    

feat_dist(np.random.choice(features))
# Predict data and Create submission file from test data

out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



# fit a model for every Province, Then make prediction

for province in tqdm(train['Province_State'].unique()):

    # train set

    train_province = train.loc[(train['Province_State'] == province)].copy()

    

    # targets   

    y_Conf_true = train_province['ConfirmedCases']

    y_Fat_true = train_province['Fatalities']

    

    # Input

    X_train_prov = train_province[features]



    #  test set

    test_province = test[(test['Province_State']== province)].copy()



    X_test_Id = test_province['ForecastId']

    X_test_prov = test_province[features]



    # fit regressors & Make predictions

    reg_Conf = XGBRegressor(n_estimators=1000)

    reg_Conf.fit(X_train_prov, y_Conf_true)

    y_Conf_pred = reg_Conf.predict(X_test_prov)



    reg_Fat = XGBRegressor(n_estimators=1000)

    reg_Fat.fit(X_train_prov, y_Fat_true)

    y_Fat_pred = reg_Fat.predict(X_test_prov)



    predictions = pd.DataFrame({'ForecastId': X_test_Id, 'ConfirmedCases': y_Conf_pred, 'Fatalities': y_Fat_pred})

    out = pd.concat([out, predictions], axis=0)

    

out['ForecastId']=out['ForecastId'].astype(int)
out.head()
out.describe()
# features importances



def feature_importance(features,reg,nb=-1):

    if hasattr(reg,'feature_importances_'):

        feature_imp=reg.feature_importances_

    else:

        feature_imp=reg.coef_

    imp_=pd.DataFrame({'feature':features,'importance':feature_imp},index=range(len(features))).sort_values(by=['importance'],ascending=False)

    if nb==-1:

        nb=20

    imp_[imp_.index<nb].plot.bar(x='feature',y='importance',rot=90)

    plt.show()
# feature_importance(features,reg_Conf,nb=-1)
# submission file

out.to_csv('submission.csv',index=False)