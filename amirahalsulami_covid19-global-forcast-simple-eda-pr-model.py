import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import dates

import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
print ("How many Province on train set ==> " +str(len(train['Province/State'].unique())))

print ("How many country on train set ==> " +str(len(train['Country/Region'].unique())))

print ("Date period for train set ==> " +train['Date'].unique()[0]+" to "+train['Date'].unique()[-1])

print ("How many Province on test set ==> " +str(len(test['Province/State'].unique())))

print ("How many country on test set ==> " +str(len(test['Country/Region'].unique())))

print ("Date period for test set ==> " +test['Date'].unique()[0]+" to "+test['Date'].unique()[-1])
train['Date_datetime'] = train['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))
train.head()
def plot_trend_by_date(df,value ='ConfirmedCases',title=None, mode='subplot'):

    ax = plt.gca()

    xaxis = df['Date_datetime'].tolist()

    if value == 'ConfirmedCases':

        yaxis = df['ConfirmedCases']

    else:

        yaxis = df['Fatalities']

        

    xaxis = dates.date2num(xaxis)

    hfmt = dates.DateFormatter('%m\n%d')

    ax.xaxis.set_major_formatter(hfmt)



    plt.xlabel('Date')

    if value == 'ConfirmedCases':

        plt.ylabel('ConfirmedCases')

    else:

        plt.ylabel('Fatalities')

    plt.title(title)

    plt.plot(xaxis, yaxis)

    plt.tight_layout()



    plt.show()
for country in train['Country/Region'].unique():

    country_pd_train = train[train['Country/Region']==country]

    if country_pd_train['Province/State'].isna().unique()==True:

        plt_title = country+' ConfirmedCase'

        plot_trend_by_date(country_pd_train,value = 'ConfirmedCases',title = plt_title)

    else:

        state_count = len(country_pd_train['Province/State'].unique())

        row = state_count//4+1

        column = 4

        fig =plt.figure(figsize = (4*6.4,row*4.8))

        index = 1

        for state in country_pd_train['Province/State'].unique():

            state_pd = country_pd_train[country_pd_train['Province/State']==state]

            plt_title = country+'  '+state+' ConfirmedCases'

            ax = fig.add_subplot(row,column,index)

            xaxis = state_pd['Date_datetime'].tolist()

            yaxis = state_pd['ConfirmedCases']

            xaxis = dates.date2num(xaxis)

            hfmt = dates.DateFormatter('%m\n%d')

            ax.xaxis.set_major_formatter(hfmt)



            plt.xlabel('Date')

            plt.ylabel('ConfirmedCases')

            plt.title(plt_title)

            ax.plot(xaxis, yaxis)

            index += 1

        plt.show() 

            #plot_trend_by_date(state_pd,value = 'ConfirmedCases',title = plt_title)
for country in train['Country/Region'].unique():

    country_pd_train = train[train['Country/Region']==country]

    if country_pd_train['Province/State'].isna().unique()==True:

        plt_title = country+' Fatalities'

        plot_trend_by_date(country_pd_train,value = 'Fatalities',title = plt_title)

    else:

        state_count = len(country_pd_train['Province/State'].unique())

        row = state_count//4+1

        column = 4

        fig =plt.figure(figsize = (4*6.4,row*4.8))

        index = 1

        for state in country_pd_train['Province/State'].unique():

            state_pd = country_pd_train[country_pd_train['Province/State']==state]

            plt_title = country+'  '+state+' Fatalities'

            ax = fig.add_subplot(row,column,index)

            xaxis = state_pd['Date_datetime'].tolist()

            yaxis = state_pd['Fatalities']

            xaxis = dates.date2num(xaxis)

            hfmt = dates.DateFormatter('%m\n%d')

            ax.xaxis.set_major_formatter(hfmt)



            plt.xlabel('Date')

            plt.ylabel('Fatalities')

            plt.title(plt_title)

            ax.plot(xaxis, yaxis)

            index += 1

        plt.show() 

            #plot_trend_by_date(state_pd,value = 'ConfirmedCases',title = plt_title)
from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression



for country in train['Country/Region'].unique():

    print ('training model for country ==>'+country)

    country_pd_train = train[train['Country/Region']==country]

    country_pd_test = test[test['Country/Region']==country]

    if country_pd_train['Province/State'].isna().unique()==True:

        x = np.array(range(len(country_pd_train))).reshape((-1,1))

        y = country_pd_train['ConfirmedCases']

        model = Pipeline([('poly', PolynomialFeatures(degree=2)),

                         ('linear', LinearRegression(fit_intercept=False))])

        model = model.fit(x, y)

        predict_x = (np.array(range(len(country_pd_test)))+50).reshape((-1,1))

        test.loc[test['Country/Region']==country,'ConfirmedCases'] = model.predict(predict_x)

    else:

        for state in country_pd_train['Province/State'].unique():

            state_pd = country_pd_train[country_pd_train['Province/State']==state] 

            state_pd_test = country_pd_test[country_pd_test['Province/State']==state] 

            x = np.array(range(len(state_pd))).reshape((-1,1))

            y = state_pd['ConfirmedCases']

            model = Pipeline([('poly', PolynomialFeatures(degree=2)),

                         ('linear', LinearRegression(fit_intercept=False))])

            model = model.fit(x, y)

            predict_x = (np.array(range(len(state_pd_test)))+50).reshape((-1,1))

            test.loc[(test['Country/Region']==country)&(test['Province/State']==state),'ConfirmedCases'] = model.predict(predict_x)
for country in train['Country/Region'].unique():

    print ('training model for country ==>'+country)

    country_pd_train = train[train['Country/Region']==country]

    country_pd_test = test[test['Country/Region']==country]

    if country_pd_train['Province/State'].isna().unique()==True:

        x = np.array(range(len(country_pd_train))).reshape((-1,1))

        y = country_pd_train['Fatalities']

        model = Pipeline([('poly', PolynomialFeatures(degree=2)),

                         ('linear', LinearRegression(fit_intercept=False))])

        model = model.fit(x, y)

        predict_x = (np.array(range(len(country_pd_test)))+50).reshape((-1,1))

        test.loc[test['Country/Region']==country,'Fatalities'] = model.predict(predict_x)

    else:

        for state in country_pd_train['Province/State'].unique():

            state_pd = country_pd_train[country_pd_train['Province/State']==state] 

            state_pd_test = country_pd_test[country_pd_test['Province/State']==state] 

            x = np.array(range(len(state_pd))).reshape((-1,1))

            y = state_pd['Fatalities']

            model = Pipeline([('poly', PolynomialFeatures(degree=2)),

                         ('linear', LinearRegression(fit_intercept=False))])

            model = model.fit(x, y)

            predict_x = (np.array(range(len(state_pd_test)))+50).reshape((-1,1))

            test.loc[(test['Country/Region']==country)&(test['Province/State']==state),'Fatalities'] = model.predict(predict_x)
submit = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')

submit['Fatalities'] = test['Fatalities'].astype('int')

submit['ConfirmedCases'] = test['ConfirmedCases'].astype('int')

submit.to_csv('submission.csv',index=False)
submit.head()