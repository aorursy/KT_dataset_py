# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')
# combine train and test
all_data = pd.merge(train, test, how='outer', on=['Country_Region','Province_State','Date'],indicator=True,sort=True)
# join country and province
all_data["Province_State"] = all_data["Province_State"].fillna('')
for index, row in all_data.iterrows():
    if all_data["Province_State"][index] != '':
        all_data["Country_Region"][index] = str(all_data["Province_State"][index]) + " (" + str(all_data["Country_Region"][index]) + ")"
all_data.drop(columns = "Province_State", inplace=True)
all_data.rename(columns = {"Country_Region" : "country_state"}, inplace=True)
# add days_since_t0, days_1st_conf, days_1st_fata
all_data["days_since_t0"] = 0
all_data["days_1st_conf"] = 0
all_data["days_1st_fata"] = 0
i = 0
for value in all_data["country_state"].unique():
    if i < len(all_data):
        j = 0
        c = 0
        f = 0
        while(all_data["country_state"][i] == value):
            all_data["days_since_t0"][i] = j
            if all_data["ConfirmedCases"][i]!=0 or pd.isna(all_data["ConfirmedCases"][i]):
                c += 1
            if all_data["Fatalities"][i]!=0 or pd.isna(all_data["Fatalities"][i]):
                f += 1
            all_data["days_1st_conf"][i] = c
            all_data["days_1st_fata"][i] = f
            j += 1; i += 1
            if i == len(all_data):
                break
all_data['lnc'] = np.log(all_data['ConfirmedCases'])   
all_data['lnf'] = np.log(all_data['Fatalities'])   
all_data['gr_conf'] = all_data.groupby('country_state').diff()['lnc']
all_data['gr_fata'] = all_data.groupby('country_state').diff()['lnf']
all_data['yc'] = np.log(all_data['gr_conf']+2)
all_data['yf'] = np.log(all_data['gr_fata']+2)
all_data['weight'] = np.exp(10*(all_data['days_since_t0']-71))
X1 = all_data.copy()
for i in range(1,2):
    X1['yc-{}'.format(i)] = X1.groupby('country_state').shift(i)['yc']
X1 = X1[X1['yc-1'].replace([np.inf, -np.inf], np.nan).notnull()]
X2 = X1[X1['yc'].replace([np.inf, -np.inf], np.nan).notnull()]
X_cc = X2[(X2['_merge']=="left_only") & (X2['days_1st_conf']>=35)]
# X_cc = X2[(X2['days_1st_conf']>=35)]

X_aux = all_data.copy()
X_aux['yc'] = X_aux['yc'].replace([np.inf, -np.inf], np.nan).fillna(0)
for i in range(1,14):
    X_aux['yc-{}'.format(i)] = X_aux.groupby('country_state').shift(i)['yc']
X_aux['yf-1'] = X_aux.groupby('country_state').shift()['yf']
X_aux = X_aux[X_aux['yf-1'].replace([np.inf, -np.inf], np.nan).notnull()]
X_aux = X_aux[X_aux['yf'].replace([np.inf, -np.inf], np.nan).notnull()]
#for i in range(1,14):
#    X_aux = X_aux[X_aux['yc-{}'.format(i)].replace([np.inf, -np.inf], np.nan).notnull()]
X_ft = X_aux[(X_aux['_merge']=="left_only") & (X_aux['days_1st_fata']>=0)&(X_aux['weight']>1.7e-35)]
from sklearn.linear_model import LinearRegression
reg_cc = LinearRegression().fit(X_cc[['yc-1', 'days_1st_conf']], X_cc['yc'], sample_weight=X_cc['weight'])
reg_ft = LinearRegression().fit(X_ft[['yf-1', 'yc-1','yc-2','yc-3','yc-4','yc-5','yc-6','yc-7','yc-8',
                                      'yc-9','yc-10','yc-11','yc-12','yc-13', 'days_1st_fata']], X_ft['yf'], sample_weight=X_ft['weight'])
print([reg_cc.coef_, reg_cc.intercept_])
print([reg_ft.coef_, reg_ft.intercept_])
countries = all_data['country_state'].unique()
for i in range(1,14):
    all_data['yc-{}'.format(i)] = all_data.groupby('country_state').shift(i)['yc']
all_data['yf-1'] = all_data.groupby('country_state').shift()['yf']

all_data['yc_predict'] = all_data['yc']
for i in range(1,14):
    all_data['yc_predict-{}'.format(i)] = all_data['yc-{}'.format(i)]
all_data['yf_predict'] = all_data['yf']
all_data['yf_predict-1'] = all_data['yf-1']
all_data['cc_predict'] = all_data['ConfirmedCases']
all_data['ft_predict'] = all_data['Fatalities']

init = 70
steps = 45

error_cc = [0]*steps
error_ft = [0]*steps
for i in range(1, steps):
    print(i)
    for c in countries:
        try:
            X_cc_temp = all_data.loc[(all_data['country_state']==c)&(all_data['days_since_t0']==init+i),['yc_predict-1', 'days_1st_conf']]
#             if (X_temp['yc_predict-1'].isnull()):
#                 X_temp['yc_predict-1'] = 1
            aux_cc = reg_cc.predict(X_cc_temp)
            all_data.loc[(all_data['country_state']==c)&(all_data['days_since_t0']==init+i),'yc_predict']= aux_cc
            for j in range(1,14):
                all_data.loc[(all_data['country_state']==c)&(all_data['days_since_t0']==init+i+j),'yc_predict-{}'.format(j)] = aux_cc
            cc_1 = all_data.loc[(all_data['country_state']==c)&(all_data['days_since_t0']==init+i-1),'cc_predict'].values
            all_data.loc[(all_data['country_state']==c)&(all_data['days_since_t0']==init+i),'cc_predict'] = cc_1*np.exp(np.exp(aux_cc)-2)
        except:
            error_cc[i-1] +=1 
        
        try:
            X_ft_temp = all_data.loc[(all_data['country_state']==c)&(all_data['days_since_t0']==init+i),['yf_predict-1','yc_predict-1', 
                                                                                                         'yc_predict-2','yc_predict-3',
                                                                                                         'yc_predict-4','yc_predict-5',
                                                                                                         'yc_predict-6','yc_predict-7',
                                                                                                         'yc_predict-8','yc_predict-9',
                                                                                                         'yc_predict-10','yc_predict-11',
                                                                                                         'yc_predict-12','yc_predict-13',
                                                                                                         'days_1st_fata']]
            aux_ft = reg_ft.predict(X_ft_temp)
            all_data.loc[(all_data['country_state']==c)&(all_data['days_since_t0']==init+i),'yf_predict']= aux_ft
            all_data.loc[(all_data['country_state']==c)&(all_data['days_since_t0']==init+i+1),'yf_predict-1'] = aux_ft
            ft_1 = all_data.loc[(all_data['country_state']==c)&(all_data['days_since_t0']==init+i-1),'ft_predict'].values
            all_data.loc[(all_data['country_state']==c)&(all_data['days_since_t0']==init+i),'ft_predict'] = ft_1*np.exp(np.exp(aux_ft)-2)
        except:
            error_ft[i-1] +=1 
test = all_data[all_data['ForecastId'].notnull()]
test.to_csv('submission.csv', index=False)