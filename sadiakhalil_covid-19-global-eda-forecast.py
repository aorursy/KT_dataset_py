import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from itertools import cycle, islice

import seaborn as sb

from matplotlib import dates



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")#index_col=0

train_data.head()
train_data.loc[:, ['ConfirmedCases', 'Fatalities']].describe()
test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")#index_col=0

test_data.tail(10)
n_train = train_data.shape[0]

n_train_col = train_data.shape[1]

n_test = test_data.shape[0]

n_test_col = test_data.shape[1]

print('number of training records:', n_train, ', number of columns:', n_train_col )

print('number of test records:', n_test, ', number of columns:', n_test_col )
def getColumnInfo(df):

    n_province =  df['Province/State'].nunique()

    n_country  =  df['Country/Region'].nunique()

    start_date =  df['Date'].unique()[0]

    end_date   =  df['Date'].unique()[-1]

    return n_province, n_country, start_date, end_date



n_prov_train, n_count_train, start_date_train, end_date_train = getColumnInfo(train_data)

n_prov_test,  n_count_test,  start_date_test,  end_date_test  = getColumnInfo(test_data)



print ('<=== Training data ===> \n# of Province/State: '+str(n_prov_train),', # of Country/Region:'+str(n_count_train),', Time Period: '+start_date_train+' to '+end_date_train)

print ('<=== Testing  data ===> \n# of Province/State: '+str(n_prov_test),', # of Country/Region:'+str(n_count_test),', Time Period: '+start_date_test+' to '+end_date_test)
def df_masked(threshold, df):

    mask = df > threshold

    tail_prob = df.loc[~mask].sum()

    df = df.loc[mask]

    df['all other'] = tail_prob

    return df



def get_frequency_plot(df, ax, xlabel, title, color):

    ax.set_xlabel(xlabel)

    ax.set_ylabel('Normalized Frequency')

    ax.set_title(title)

    df_mask = df_masked(0.01, df)

    df_mask.plot(kind='bar', color=color)

    plt.xticks(rotation=25)

    xlocs, xlabs = plt.xticks()

    for i, v in enumerate(df_mask):

         plt.text(xlocs[i] - 0.255, v+0.01, str(round(v,2)))



###############            

prob_confirm_check_train = train_data.ConfirmedCases.value_counts(normalize=True)

prob_fatal_check_train = train_data.Fatalities.value_counts(normalize=True)

plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(15,5))

ax0 = fig.add_subplot(1,2,1)

get_frequency_plot(prob_confirm_check_train, ax0, 'Number of Confirmed Cases', 'Confirmed Cases Record', 'orange')

ax1 = fig.add_subplot(1,2,2)

get_frequency_plot(prob_fatal_check_train, ax1, 'Number of Fatalities', 'Fatalities Record','red')



plt.show()



n_confirm_train = train_data.ConfirmedCases.value_counts()[1:].sum()

n_fatal_train = train_data.Fatalities.value_counts()[1:].sum()



print('=== Confirmed Cases in Training Dataset ===')

print('Percentage of 0 confirmed case records: {0:<2.1f}% \n' 

      'Percentage of confirmed case records: {1:<2.1f}% \n'

      'Ratio of confirmed cases records: {2:<2.0f}/{3:<2.0f} '.format(prob_confirm_check_train[0]*100, prob_confirm_check_train[1:].sum()*100, n_confirm_train, n_train))

print('\n=== Fatalities in Training Dataset ===')

print('Percentage of 0 fatalities: {0:<2.1f}% \n' 

      'Percentage of fatalities: {1:<2.1f}% \n'

      'Ratio of fatality records: {2:<2.0f}/{3:<2.0f} '.format(prob_fatal_check_train[0]*100, prob_fatal_check_train[1:].sum()*100, n_fatal_train, n_train))
train_data_by_country = train_data.groupby(['Country/Region'],as_index=True).agg({'ConfirmedCases': 'max', 'Fatalities': 'max'})

train_data_by_country_confirm = train_data_by_country.sort_values(by=["ConfirmedCases"], ascending=False)

train_data_by_country_confirm.head(10)
from itertools import cycle, islice

discrete_col = list(islice(cycle(['orange', 'r', 'g', 'k', 'b', 'c', 'm']), None, len(train_data_by_country_confirm.head(20))))

plt.rcParams.update({'font.size': 22})

train_data_by_country_confirm.head(20).plot(figsize=(15,10), kind='barh', color=discrete_col)

plt.legend(["Confirmed Cases", "Fatalities > 100"]);

plt.xlabel("Number of Covid-19 affectees")

plt.title("Confirmed Cases by Country in Training Data")

ylocs, ylabs = plt.yticks()

for i, v in enumerate(train_data_by_country_confirm.head(20)["ConfirmedCases"][:]):

    plt.text(v+0.01, ylocs[i]-0.25, str(int(v)), fontsize=12)

for i, v in enumerate(train_data_by_country_confirm.head(20)["Fatalities"][:]):

    if v > 100: #disply for only >100 fatalities

        plt.text(v+0.01,ylocs[i]+0.1,str(int(v)),fontsize=12)    
train_data['Date'] = pd.to_datetime(train_data['Date'])

train_data_by_date = train_data.groupby(['Date'],as_index=False).agg({'ConfirmedCases': 'max','Fatalities': 'max'})

train_data_by_date.head()
train_data_by_country_fatal = train_data_by_country[train_data_by_country['Fatalities']>200]

train_data_by_country_fatal = train_data_by_country_fatal.sort_values(by=['Fatalities'],ascending=False).reset_index()

train_data_by_country_fatal.head(20)
df_merge_by_country = pd.merge(train_data,train_data_by_country_fatal['Country/Region'],on=['Country/Region'],how='inner')

df_max_fatality_country = df_merge_by_country.groupby(["Date","Country/Region"],as_index=False).agg({'ConfirmedCases': 'sum','Fatalities': 'sum'})

df_max_fatality_country.head(20)

import datetime as dt

date_list = train_data_by_date["Date"].tolist()

xticks = [dt.datetime.strftime(t,'%Y-%m-%d') for t in date_list]

xticks = [tick for i,tick in enumerate(xticks) if i%4==0 ]# split labels into equally spaced ticks



def setCosmetics(ax, xticks, xLabel, yLabel, title):

    ax.set_xticks(xticks)

    ax.set_xticklabels([i for i in xticks])

    plt.setp(ax.get_xticklabels(), rotation=90)

    ax.set_xlabel(xLabel)

    ax.set_ylabel(yLabel)

    ax.set_title(title)

    ax.yaxis.grid(linestyle='dotted')

    ax.spines['right'].set_color('none')

    ax.spines['top'].set_color('none')

    ax.spines['left'].set_color('none')

    ax.spines['bottom'].set_color('none')

    ax.legend()

    

plt.rcParams.update({'font.size': 16})



fig,(ax0,ax1) = plt.subplots(1,2,figsize=(15, 5),sharey=True)



ax0.plot(train_data_by_date['Date'],train_data_by_date['Fatalities'], color='r',marker='o',linewidth=2,label='Global')

setCosmetics(ax0,xticks,'Date','Fatalities','Time Evolution of Global Fatalities')



countries = df_max_fatality_country['Country/Region'].unique().tolist()

for country in countries:

    match = df_max_fatality_country['Country/Region']==country

    df_fatality_by_country = df_max_fatality_country[match]

    ax1.plot(df_fatality_by_country["Date"],df_fatality_by_country['Fatalities'],marker='o',linewidth=2,label=country)

    setCosmetics(ax1,xticks,'Date','Fatalities','Time Evolution of National Fatalities')

    

fig,(ax2,ax3) = plt.subplots(1,2,figsize=(15,5),sharey=True)     

ax2.plot(train_data_by_date['Date'],train_data_by_date['ConfirmedCases'], color='orange',marker='o',linewidth=2,label='Global')

setCosmetics(ax2,xticks,'Date','ConfirmedCases','Time Evolution of Global Confirmed Cases')



for country in countries:

    df_fatality_by_country = df_max_fatality_country[df_max_fatality_country['Country/Region']==country]

    ax3.plot(df_fatality_by_country["Date"],df_fatality_by_country['ConfirmedCases'],marker='o',linewidth=2,label=country)

    setCosmetics(ax3,xticks,'Date','ConfirmedCases','Time Evolution of National Confirmed Cases')
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, HuberRegressor

from sklearn.metrics import r2_score 

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.pipeline import make_pipeline

from tqdm import tqdm



plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(15,8)) 

ax0 = fig.add_subplot(1,2,1)

for country in tqdm(countries): 

    df_country_train = df_max_fatality_country[df_max_fatality_country['Country/Region']==country] 

    df_country_test = test_data[test_data['Country/Region']==country]  

    days_in_train_by_country = df_country_train.Date.nunique()

    days_in_test_by_country  = df_country_test.Date.nunique()

    x_train = np.array(range(days_in_train_by_country)).reshape((-1,1))   

    y_train = df_country_train['Fatalities']   

    x_test = (np.array(range(days_in_test_by_country))+45).reshape((-1,1)) #allow overlap of few days in training data       

    model = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False))  

    model = model.fit(x_train, y_train)       

    y_predict = model.predict(x_test)  

    ax0.plot(x_test , y_predict,linewidth=2, label='predict_'+country)

    ax0.plot(x_train , y_train, linewidth=2, color='r', linestyle='dotted', label='train_'+country)

    ax0.set_title("Prediction vs Training for Fatalities")

    ax0.set_xlabel("Number of days")

    ax0.set_ylabel("Fatalities")

    ax0.legend()

    

ax1 = fig.add_subplot(1,2,2)

for country in tqdm(countries): 

    df_country_train = df_max_fatality_country[df_max_fatality_country['Country/Region']==country] 

    df_country_test = test_data[test_data['Country/Region']==country]  

    days_in_train_by_country = df_country_train.Date.nunique()

    days_in_test_by_country  = df_country_test.Date.nunique()

    x_train = np.array(range(days_in_train_by_country)).reshape((-1,1))   

    y_train = df_country_train['ConfirmedCases']   

    x_test = (np.array(range(days_in_test_by_country))+45).reshape((-1,1)) #allow overlap of few days in training data       

    model = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False))  

    model = model.fit(x_train, y_train)       

    y_predict = model.predict(x_test) 

    ax1.plot(x_test , y_predict,linewidth=2, label='predict_'+country)

    ax1.plot(x_train , y_train, linewidth=2, color='r', linestyle='dotted', label='train_'+country)

    ax1.set_title("Prediction vs Training for Confirmed Cases")

    ax1.set_xlabel("Number of days")

    ax1.set_ylabel("Confirmed Cases")

    ax1.legend()    

nCountries= train_data['Country/Region'].unique()  

for country in tqdm(nCountries): 

    df_country_train = train_data[train_data['Country/Region']==country] 

    df_country_test = test_data[test_data['Country/Region']==country]

    if df_country_train['Province/State'].isna().unique()==True:   

        days_in_train_by_country = df_country_train.Date.nunique()

        days_in_test_by_country  = df_country_test.Date.nunique()

        x_train = np.array(range(days_in_train_by_country)).reshape((-1,1))    

        y_train = df_country_train['Fatalities']        

        model = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False))  

        model = model.fit(x_train, y_train)

        x_test = (np.array(range(days_in_test_by_country))+days_in_train_by_country-14).reshape((-1,1))  

        y_predict = model.predict(x_test)   

        test_data.loc[test_data['Country/Region']==country,'Fatalities'] = y_predict

    else: # use Province/State data when available

        for state in df_country_train['Province/State'].unique():

            df_state_train = df_country_train[df_country_train['Province/State']==state] 

            df_state_test = df_country_test[df_country_test['Province/State']==state]                    

            days_in_train_by_state = df_state_train.Date.nunique()

            days_in_test_by_state  = df_state_test.Date.nunique()            

            x_train = np.array(range(days_in_train_by_state)).reshape((-1,1))

            y_train = df_state_train['Fatalities']

            model = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False))

            model = model.fit(x_train, y_train)  

            x_test = (np.array(range(days_in_test_by_state))+days_in_train_by_state-14).reshape((-1,1)) 

            y_predict = model.predict(x_test) 

            y_predict = [ip if ip>=0 else 0 for ip in y_predict]

            test_data.loc[(test_data['Country/Region']==country)&(test_data['Province/State']==state),'Fatalities'] = y_predict



        

for country in tqdm(nCountries): 

    df_country_train = train_data[train_data['Country/Region']==country] 

    df_country_test = test_data[test_data['Country/Region']==country]

    if df_country_train['Province/State'].isna().unique()==True:    

        days_in_train_by_country = df_country_train.Date.nunique()

        days_in_test_by_country  = df_country_test.Date.nunique()

        x_train = np.array(range(days_in_train_by_country)).reshape((-1,1))    

        y_train = df_country_train['ConfirmedCases']        

        model = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False))  

        model = model.fit(x_train, y_train)

        x_test = (np.array(range(days_in_test_by_country))+days_in_train_by_country-14).reshape((-1,1)) 

        y_predict = model.predict(x_test)   

        test_data.loc[test_data['Country/Region']==country,'ConfirmedCases'] = y_predict

    else: # use Province/State data when available

        for state in df_country_train['Province/State'].unique():

            df_state_train = df_country_train[df_country_train['Province/State']==state] 

            df_state_test = df_country_test[df_country_test['Province/State']==state]                    

            days_in_train_by_state = df_state_train.Date.nunique()

            days_in_test_by_state  = df_state_test.Date.nunique()            

            x_train = np.array(range(days_in_train_by_state)).reshape((-1,1))

            y_train = df_state_train['ConfirmedCases']

            model = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False))

            model = model.fit(x_train, y_train)  

            x_test = (np.array(range(days_in_test_by_state))+days_in_train_by_state-14).reshape((-1,1)) 

            #print(x_test)

            y_predict = model.predict(x_test) 

            y_predict = [ip if ip>=0 else 0 for ip in y_predict]

            test_data.loc[(test_data['Country/Region']==country)&(test_data['Province/State']==state),'ConfirmedCases'] = y_predict

submit_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")#, index_col=0)

submit_data['Fatalities'] = test_data['Fatalities'].astype('int')

submit_data['ConfirmedCases'] = test_data['ConfirmedCases'].astype('int')

submit_data.to_csv('submission.csv', index=False)

submit_data.head(10)
submit_data.describe()