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
import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

train['Date'] = pd.to_datetime(train['Date']).dt.date

test['Date'] = pd.to_datetime(test['Date']).dt.date

df_train = train[train.ConfirmedCases>0]



test_start = min(test.Date)

df_regions = df_train[['Province_State', 'Country_Region']].drop_duplicates()
# Choose method

method = 1
if method == 1:

    from scipy.optimize import curve_fit

    from scipy.optimize import differential_evolution

    from scipy.optimize import minimize

    import warnings, datetime

    from statsmodels.regression.linear_model import OLS





    # We test the model on the entire data set df_cty here!!!

    country_names = pd.unique(test.Country_Region)



    def generate_Initial_Parameters(xData,yData):

        # min and max used for bounds

        maxX = max(xData)

        minX = min(xData)

        maxY = max(yData)

        minY = min(yData)



        parameterBounds = []

        parameterBounds.append([minX, maxX]) # search bounds for a0

        parameterBounds.append([minX, maxX]) # search bounds for b0

        parameterBounds.append([0.0, maxY]) # search bounds for a1

        parameterBounds.append([0.0, maxY]) # search bounds for b1



        def sumOfSquaredError(parameterTuple):

            warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm

            val = sigmoid(xData, *parameterTuple)

            return np.sum((yData - val) ** 2.0)



        # "seed" the numpy random number generator for repeatable results

        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)

        return result.x



    def sigmoid(x, a0, b0, a1, b1):

        y = a1 / (1 + np.exp(-a0*(x-b0)))+b1

        return y



    def sig(x,param):

        return (param[2]/(1 + np.exp(-param[0]*(x-param[1])))+param[3])

    num_regions = df_regions.shape[0]

    for i in range(num_regions): 

        if pd.notna(df_regions.iloc[i].Province_State):

            df_region = df_train.loc[(df_train.Country_Region==df_regions.iloc[i].Country_Region) & (df_train.Province_State==df_regions.iloc[i].Province_State)]

        else:

            df_region =  df_train.loc[(df_train.Country_Region==df_regions.iloc[i].Country_Region) & df_train.Province_State.isna()]



        y1_start = min(df_region.Date)

        y1 = df_region.ConfirmedCases

        x1_train = np.arange(len(y1))/100

        y1_scale = y1/max(y1)

        x1_test_start = (min(test.Date) - y1_start).days

        x1_test_end = (max(test.Date) - y1_start).days

        x1_test = np.arange(x1_test_start-1,x1_test_end)/100    



        # Confirmed Cases

        fig, ax = plt.subplots(figsize=(15,5))

        plt.title([df_regions.iloc[i].Country_Region, df_regions.iloc[i].Province_State,'Confirmed'])

        ax.plot(x1_train*100, y1, 'o', label='Confirmed_actual')

        if len(y1)>1:

            try:

                coef1 = curve_fit(sigmoid, x1_train,y1_scale, generate_Initial_Parameters(x1_train,y1_scale))[0]

                y1_pred = sig(x1_train,coef1)*max(y1)

                y1_test = sig(x1_test, coef1)*max(y1)

            except:

                x = x1_train

                def polynomial(p, x):

                    return p[0]+p[1]*x+p[2]*x**2+p[3]*x**3



                def constraint_1st_der(p):

                    return p[1]+2*p[2]*x+3*p[3]*x**2

                

                def constraint_2nd_der(p):

                    return 2*p[2]+6*p[3]*x



                def objective(p):

                    return ((polynomial(p, x)- y1)**2).sum()



                cons = (dict(type='ineq', fun=constraint_1st_der), dict(type='ineq', fun=constraint_2nd_der))

                res = minimize(objective, x0=np.array([0., 0., 0., 0.]), method='SLSQP', constraints=cons)

                

                if res.success:

                    y1_pred = polynomial(res.x, x1_train)

                    y1_test = polynomial(res.x, x1_test)

                else:

                    poly_fit1 = np.poly1d(np.polyfit(x1_train,y1,deg = 2))

                    y1_pred = poly_fit1(x1_train)

                    y1_test = poly_fit1(x1_test)

        else:

            y1_pred = np.full(len(x1_train),y1)

            y1_test = np.full(len(x1_test),y1)   





        ax.plot(x1_train*100,y1_pred, label='Confirmed_train')

        ax.plot(x1_test*100,y1_test, label='Confirmed_test')

        ax.legend()

        if pd.isnull(df_regions.iloc[i].Province_State):

            test.loc[(test.Country_Region==df_regions.iloc[i].Country_Region) & test.Province_State.isna(),'ConfirmedCases'] = y1_test

        else:        

            test.loc[(test.Country_Region==df_regions.iloc[i].Country_Region) & \

                     (test.Province_State==df_regions.iloc[i].Province_State),'ConfirmedCases'] = y1_test



        # Death

        if len(df_region.loc[df_region.Fatalities>0,'Date'])>0:

            y2_start = min(df_region.loc[df_region.Fatalities>0,'Date'])

            y2 = df_region.loc[df_region.Fatalities>0,'Fatalities']        

            x2_train = np.arange(len(y2))/100    

            y2_scale = y2/max(y2) 

            x2_test_start = (min(test.Date) - y2_start).days

            x2_test_end = (max(test.Date) - y2_start).days

            x2_test = np.arange(x2_test_start-1,x2_test_end)/100     

            fig, ax = plt.subplots(figsize=(15,5))        

            ax.plot(x2_train*100, y2, 'o', label='Death_actual')

            plt.title([df_regions.iloc[i].Country_Region, df_regions.iloc[i].Province_State,'Death'])



            if len(y2)>1:

                try:

                    coef2 = curve_fit(sigmoid, x2_train,y2_scale, generate_Initial_Parameters(x2_train,y2_scale))[0]

                    y2_pred = sig(x2_train,coef2)*max(y2)

                    y2_test = sig(x2_test, coef2)*max(y2)

                except:

                    x = x2_train

                    def polynomial(p, x):

                        return p[0]+p[1]*x+p[2]*x**2+p[3]*x**3



                    def constraint_1st_der(p):

                        return p[1]+2*p[2]*x+3*p[3]*x**2

                    

                    def constraint_2nd_der(p):

                        return 2*p[2]+6*p[3]*x



                    def objective(p):

                        return ((polynomial(p, x)- y2)**2).sum()



                    cons = (dict(type='ineq', fun=constraint_1st_der), dict(type='ineq', fun=constraint_2nd_der))

                    res = minimize(objective, x0=np.array([0., 0., 0., 0.]), method='SLSQP', constraints=cons)



                    if res.success:

                        y2_pred = polynomial(res.x, x2_train)

                        y2_test = polynomial(res.x, x2_test)

                    else:

#                         olsmod2 = OLS(np.log(y2),x2_train)

#                         olsres2 = olsmod2.fit()

#                         y2_pred = np.exp(olsres2.predict(x2_train))

#                         y2_test = np.exp(olsres2.predict(x2_test))

                        poly_fit2 = np.poly1d(np.polyfit(x2_train,y2,deg = 2))

                        y2_pred = poly_fit2(x2_train)

                        y2_test = poly_fit2(x2_test)

            else:

                y2_pred = np.full(len(x2_train),y2)

                y2_test = np.full(len(x2_test),y2)    



            ax.plot(x2_train*100,y2_pred, label='Death_train')

            ax.plot(x2_test*100,y2_test, label='Death_test')

            ax.legend()

            if pd.isnull(df_regions.iloc[i].Province_State):

                test.loc[(test.Country_Region==df_regions.iloc[i].Country_Region) & test.Province_State.isna(),'Fatalities'] = y2_test

            else:

                test.loc[(test.Country_Region==df_regions.iloc[i].Country_Region) & \

                         (test.Province_State==df_regions.iloc[i].Province_State),'Fatalities'] = y2_test



        #else:

        #    print(df_regions.iloc[i], ' has 0 deaths')

        submission = test.copy()

        submission.loc[submission.ConfirmedCases.isna(),'ConfirmedCases'] = 0

        submission.loc[submission.Fatalities.isna(),'Fatalities'] = 0

        submission['ConfirmedCases'] = submission.ConfirmedCases.astype(int)

        submission['Fatalities'] = submission.Fatalities.astype(int)

        submission.loc[submission.Fatalities<0,'Fatalities'] = 0

        submission.to_csv('/kaggle/working/submission.csv')

        submission[['ForecastId','ConfirmedCases','Fatalities']].to_csv('submission.csv',index = False)

from sklearn.preprocessing import LabelEncoder,MinMaxScaler

from xgboost import XGBRegressor

import timeit

from random import sample





df_train = train.copy()

df_test = test.copy()



# Concate country and state name

df_train['Province_State'].fillna('',inplace =True)

df_test['Province_State'].fillna('',inplace =True)

df_train['Country_Region'] = df_train.Country_Region +'_'+df_train.Province_State

df_test['Country_Region'] = df_test.Country_Region +'_'+df_test.Province_State



# Calculate the days since the begining of training data

train_start = min(df_train.Date)

df_train['Days'] = df_train['Date'] - train_start

df_train['Days'] = df_train.Days.dt.days

df_test['Days'] = df_test['Date'] - train_start

df_test['Days'] = df_test.Days.dt.days



# Encode country and region name

le = LabelEncoder()

df_train['Country_Region'] = le.fit_transform(df_train['Country_Region'])

df_test['Country_Region'] = le.transform(df_test['Country_Region'])



# define y variable

y_cc = df_train.ConfirmedCases.astype(int)

y_ft = df_train.Fatalities.astype(int)



# Prepare training data for ConfirmedCases

x_train_cc = df_train[['Country_Region','Days']]

train_sample = sample(range(df_train.shape[0]),df_train.shape[0]*7//10)

vald_sample = [i for i in range(df_train.shape[0]) if i not in train_sample]

x_train1_cc = x_train_cc.iloc[train_sample]

y_train1_cc = y_cc.iloc[train_sample]

x_train2_cc = x_train_cc.iloc[vald_sample]

y_train2_cc = y_cc.iloc[vald_sample]

x_test_cc = df_test[['Country_Region','Days']]



# Scale input

mms = MinMaxScaler()

x_train_cc = mms.fit_transform(x_train_cc)

x_train1_cc = mms.transform(x_train1_cc)

x_train2_cc = mms.transform(x_train2_cc)

x_test_cc = mms.transform(x_test_cc)

# Train XGB model for Confirmed Cases

start = timeit.default_timer()

model_cc = XGBRegressor(n_estimators = 1500 , random_state = 0 , max_depth = 15)

model_cc.fit(x_train_cc,y_cc)

#eval_set = [(x_train2_cc, y_train2_cc)]

#model_cc.fit(x_train1_cc, y_train1_cc, early_stopping_rounds=10, eval_set=eval_set, verbose=False)

print(np.round(timeit.default_timer() - start,decimals =0),'seconds')



y_train_cc_pred = model_cc.predict(x_train_cc)

y_test_cc_pred = model_cc.predict(x_test_cc)



# Projeciton for training

df_train['Pred_CC'] = np.round(y_train_cc_pred,decimals=0)

df_test['Pred_CC'] = np.round(y_test_cc_pred,decimals=0)

fig, ax = plt.subplots(figsize=(15,5))

tmp = df_train.pivot_table(index ='Date', values =['ConfirmedCases','Pred_CC'],aggfunc = sum).reset_index()

ax.plot(tmp.Date,tmp.ConfirmedCases,'o',label ='Actual')

ax.plot(tmp.Date,tmp.Pred_CC,label ='Pred')

plt.legend()



# Projeciton for test

fig, ax = plt.subplots(figsize=(15,5))

tmp = df_test.pivot_table(index ='Date', values =['Pred_CC'],aggfunc = sum).reset_index()

ax.plot(tmp.Date,tmp.Pred_CC,label ='Pred')

# Train XGB model for Fatalities

#x_train_ft = np.array(df_train[['Country_Region','Days','ConfirmedCases']])

x_train_ft = np.array(df_train[['Country_Region','Days']])

x_train_ft = mms.fit_transform(x_train_ft)

x_test_ft = x_test_cc



start = timeit.default_timer()

model_ft = XGBRegressor(n_estimators = 1500 , random_state = 0 , max_depth = 15)

model_ft.fit(x_train_ft,y_ft)

#eval_set = [(x_train2_cc, y_train2_cc)]

#model_cc.fit(x_train1_cc, y_train1_cc, early_stopping_rounds=10, eval_set=eval_set, verbose=False)

print(np.round(timeit.default_timer() - start,decimals =0),'seconds')



y_train_ft_pred = model_ft.predict(x_train_ft)

y_test_ft_pred = model_ft.predict(x_test_ft)



# Projeciton for training

df_train['Pred_FT'] = np.round(y_train_ft_pred,decimals=0)

df_test['Pred_FT'] = np.round(y_test_ft_pred,decimals=0)

fig, ax = plt.subplots(figsize=(15,5))

tmp = df_train.pivot_table(index ='Date', values =['Fatalities','Pred_FT'],aggfunc = sum).reset_index()

ax.plot(tmp.Date,tmp.Fatalities,'o',label ='Actual')

ax.plot(tmp.Date,tmp.Pred_FT,label ='Pred')

plt.legend()



# Projeciton for test

fig, ax = plt.subplots(figsize=(15,5))

tmp = df_test.pivot_table(index ='Date', values =['Pred_FT'],aggfunc = sum).reset_index()

ax.plot(tmp.Date,tmp.Pred_FT,label ='Pred')

# Submission

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

submission['ConfirmedCases'] = y_test_cc_pred.astype(int)

submission['Fatalities'] = y_test_ft_pred.astype(int)

submission.head()

submission.to_csv('submission.csv',index=False)
#submission = pd.read_csv('/kaggle/input/covid19-week4/submission.csv')

#submission = submission.merge(test[['ForecastId','Country_Region','Province_State','Date']],on ='ForecastId', how='left')

#submission['Date'] = pd.to_datetime(submission['Date']).dt.date

#fig, ax = plt.subplots(figsize=(15,5))

#country ='US'

#submission.loc[submission.Country_Region==country].groupby('Date').ConfirmedCases.sum().plot(ax = ax)

#submission.loc[submission.Country_Region=='China'].groupby('Date').Fatalities.sum().plot(ax = ax)
