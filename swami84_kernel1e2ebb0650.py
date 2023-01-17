# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from scipy.optimize import curve_fit

from scipy.integrate import odeint

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):



    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

# test['Province_State'].fillna('', inplace = True)

# train['Province_State'].fillna('', inplace = True)

test['Date'] = pd.to_datetime(test['Date'])

train['Date'] = pd.to_datetime(train['Date'])

train['Province_State'] = train['Province_State'].fillna('None')

train['unique_id'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)



print('Total Number of Country in Training Data: ', train['Country_Region'].nunique())

print('Has in total number of Province or States: ', train['Province_State'].nunique())

print('Date range: ', min(train['Date']), max(train['Date']), 'Today number of days: ', train['Date'].nunique())



print('Total Number of Country in Test Data: ', test['Country_Region'].nunique())

print('Has in total number of Province or States: ', test['Province_State'].nunique())

print('Date range: ', min(test['Date']), max(test['Date']), 'Today number of days: ', test['Date'].nunique())



print('For the training dataset, the number of regions on the first day ', min(train['Date']), ' are ', train[train['Date'] == min(train['Date'])]['Country_Region'].nunique())
tot_confirmed = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

tot_fatalities = train.groupby(['Date']).agg({'Fatalities':['sum']})

tot_case_bydate = tot_confirmed.join(tot_fatalities)

tot_case_bydate.reset_index(inplace = True)

tot_case_bydate.head()



fig = px.scatter(tot_case_bydate, x = 'Date', y = 'ConfirmedCases')

fig.update_layout(title='Global Confirmed Cases - Cumulative')

fig.show()
fig = px.scatter(tot_case_bydate, x = 'Date', y = 'Fatalities')

fig.update_layout(title='Global Fatalities Cases - Cumulative')

fig.show()
def run_curve_fit_diff_S(y1,y2,N0):

   

    I0 = y1[0]

    R0 = y2[0]

    def deriv(y, t,N0, beta,gamma,tau):



        S,I,R,N = y

        dNdT = N*(1-(1/np.exp(tau)))

        dSdt = -beta * S * I/N

        

        dIdt = beta * S * I/N  - gamma * I

        dRdt = gamma * I

        return dSdt,dIdt,dRdt,dNdT

    

    def odeint_func(t,N0,beta,gamma,tau,I0,R0):

        

        S0 = (N0 - I0 - R0)

        y0 = S0, I0, R0,N0

        ret = odeint(deriv, y0, t, args=(N0,beta, gamma,tau))

        print()

        return np.ravel(np.vstack((ret[:,1],ret[:,2])))



    t = np.arange(0,len(y1),1)

    y_t = np.vstack((y1,y2))



    values , pcov = curve_fit(lambda t,beta,gamma,tau: odeint_func(t,N0,beta,gamma,tau,I0,R0), 

                          t, np.ravel(y_t) ,bounds=((0,0,-np.inf),(1,1,np.inf)),maxfev=999999)

        

    return values[0],values[1],values[2]
def deriv(y, t, N,beta,gamma):



            S,I,R = y

            dSdt = -beta * S * I/N

            dIdt = beta * S * I/N  - gamma * I

            dRdt = gamma * I

            return dSdt,dIdt,dRdt

def odeint_func(t,N,I0,R0,beta,gamma):

    

    S0 = (N - I0 - R0)

    y0 = S0, I0, R0

    ret = odeint(deriv, y0, t, args=(N, beta, gamma))



    return np.ravel(np.vstack((ret[:,1],ret[:,2])))
data = pd.read_csv('../input/worldpopulation/world_population.csv')

us_state_pop = pd.read_csv('../input/worldpopulation/state_population.csv')
pred_confirm = []

pred_fatality = []



betas,gammas = [],[]

for uid in train['unique_id'].unique():

    country, state = None, None

    df = train[train['unique_id'] == uid]

#     display(df)

    y_conf = df.set_index('Date')['ConfirmedCases'].values.flatten()

    y_death = df.set_index('Date')['Fatalities'].values.flatten()

    idx = np.argwhere(y_conf>0)[0][0]

    

    y_active = y_conf[idx:]

    y_deaths = y_death[idx:]

    if y_conf[-1] < 20:

        lreg = LinearRegression()

        pred_confirm.append(y_conf[56:69])

        pred_fatality.append(y_death[56:69])

        x = np.arange(0,len(y_conf),1)

        x = x.reshape(-1,1)

        x_test = np.arange(max(x) + 1, max(x) + 31, 1)

        x_test = x_test.reshape(-1,1)

        lreg.fit(x,y_conf)

        predict_c = lreg.predict(x_test)

        pred_confirm.append(predict_c)



        lreg.fit(x,y_death)

        predict_f = lreg.predict(x_test)

        pred_fatality.append(predict_f)

        continue



    

    

    if 'None' in uid:

        

        country = uid.split('_None')[0]

        if ' ' in country:

            country = country.replace(' ', '_')

        if country in data['countriesAndTerritories'].unique():

            N = data[data['countriesAndTerritories']==country]['popData2018'].values[0]

            

        else:

            N = 1000000

            

        

    elif 'US' in uid:

        state = uid.split('US_')[1]

        

        if state in us_state_pop['NAME'].unique():

            N = us_state_pop[us_state_pop['NAME']==state]['POPESTIMATE2019'].values[0]

        else: 

            N = 150000

        

    else:

        

        country = uid.split('_')[0]

        

        uniq_list = train['unique_id'].unique()

        tot = len([s for s in uniq_list if "United Kingdom" in s])

        if ' ' in country:

            country = country.replace(' ', '_')

        

        N = data[data['countriesAndTerritories']==country]['popData2018'].values[0]/tot

        

    beta,gamma,tau = run_curve_fit_diff_S(y_active,y_deaths,N)

    betas.append(beta)

    gammas.append(gamma)

    t_pred = np.arange(0,len(y_active)+30,1)

    pred_confirm.append(y_conf[56:69])

    pred_fatality.append(y_death[56:69])

    y_pred_conf = odeint_func(t_pred,N,y_active[0],y_deaths[0],beta, gamma)[:len(y_active)+30]

    y_pred_death = odeint_func(t_pred,N,y_active[0],y_deaths[0],beta, gamma)[30+len(y_active):]

    pred_confirm.append(y_pred_conf[-30:])

    pred_fatality.append(y_pred_death[-30:])
pred_confirm = [item for sublist in pred_confirm for item in sublist]

pred_fatality = [item for sublist in pred_fatality for item in sublist]
submission = pd.DataFrame({'ForecastId': test['ForecastId'], 

                           'ConfirmedCases': pred_confirm, 

                           'Fatalities': pred_fatality})

submission_int = submission.round(0)

submission_int = submission_int.astype(int)



submission_int.to_csv('submission.csv', index = False)

len(submission)