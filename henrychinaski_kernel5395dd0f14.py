import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from pandas_profiling import ProfileReport
# Load Data
X_train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
X_test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')
X_submission = pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')
#train_profile = ProfileReport(X_train)
#train_profile
print(X_train.shape, X_test.shape, X_submission.shape)
X_test.head()
#X_train.info()
X_train['Date'] = pd.to_datetime(X_train['Date'])
X_test['Date'] = pd.to_datetime(X_test['Date'])
X_test['Date']
print(X_train.Country_Region.nunique())
countries = X_train.Country_Region.unique()
countries
countries_with_provinces = X_train[~X_train['Province_State'].isna()].Country_Region.unique() #complement of data entries of column 'Province_State'
countries_with_provinces
countries_no_province = [i for i in countries if i not in countries_with_provinces]
len(countries_no_province)
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


#This is for visualising the data. As there are 306 datasets only a fraction of 20 sets can be choosen here
_, ax = plt.subplots(23,4, figsize=(15, 30))
ax = ax.flatten()

for k, i in tqdm(enumerate(countries[92:184])):   #Define the part of the dataset you want to look at, e.g. [200:220]
    
    #sns.scatterplot(x=valid[valid['loc']==coords]['date'], y=valid[valid['loc']==coords]['confirmed'], label='y-valid',ax=ax[k])
    #sns.lineplot(x=train[train['loc']==coords]['date'], y=train[train['loc']==coords]['confirmed'], label='y-train',ax=ax[k])
    #sns.lineplot(x=valid[valid['loc']==coords]['date'], y=y_preds_, label=(f'y-preds, {fit_order}'),ax=ax[k])
    #ax[k].set_title(f'Confirmed cases: ({coords})')

    sns.lineplot(x=X_train[X_train['Country_Region'] == i].Date, y=X_train[X_train['Country_Region'] == i].Fatalities, label=(f'Fat: {i}'),ax=ax[k])
    sns.lineplot(x=X_train[X_train['Country_Region'] == i].Date, y=X_train[X_train['Country_Region'] == i].ConfirmedCases, label='CC', ax=ax[k])
    ax[k].set_xlabel('')
    ax[k].set_ylabel('')
    ax[k].set(xticklabels=[])
    
   
X_train['Province_State'] = X_train['Province_State'].fillna('unknown')
X_test['Province_State'] = X_test['Province_State'].fillna('unknown')
X_train[X_train['Country_Region'].isin(countries_with_provinces)].groupby(['Country_Region']).agg({'Province_State':'nunique'})
X_train['Date'] = X_train['Date'].dt.strftime("%m%d")
X_train['Date'] = X_train['Date'].astype(int) 

X_test['Date'] = X_test['Date'].dt.strftime("%m%d")
X_test['Date'] = X_test['Date'].astype(int) 
X_train['Province_State'] = X_train['Province_State'].fillna('unknown')
X_test['Province_State'] = X_test['Province_State'].fillna('unknown')
X_train['Province_State'] = X_train['Province_State'].astype('category')
X_train['Country_Region'] = X_train['Country_Region'].astype('category')

X_test['Province_State'] = X_test['Province_State'].astype('category')
X_test['Country_Region'] = X_test['Country_Region'].astype('category')
X_train
from xgboost import XGBRegressor
import xgboost as xgb

FEATURES = ['Date']
X_submission = pd.DataFrame(columns=['ForecastId', 'ConfirmedCases', 'Fatalities'])

for i in tqdm(X_train.Country_Region.unique()):
    z_train = X_train[X_train['Country_Region'] == i]
    z_test = X_test[X_test['Country_Region'] == i]
    for k in z_train.Province_State.unique():
        p_train = z_train[z_train['Province_State'] == k]
        p_test = z_test[z_test['Province_State'] == k]
        X_train_final = p_train[FEATURES]
        y1 = p_train['ConfirmedCases']
        y2 = p_train['Fatalities']
        model = xgb.XGBRegressor(n_estimators=2000)
        model.fit(X_train_final, y1)
        ConfirmedCasesPreds = model.predict(p_test[FEATURES])
        model.fit(X_train_final, y2)
        FatalitiesPreds = model.predict(p_test[FEATURES])
        
        p_test['ConfirmedCases'] = ConfirmedCasesPreds
        p_test['Fatalities'] = FatalitiesPreds
        X_submission = pd.concat([X_submission, p_test[['ForecastId', 'ConfirmedCases', 'Fatalities']]], axis=0)


X_submission
X_submission.to_csv('submission.csv', index=False)






