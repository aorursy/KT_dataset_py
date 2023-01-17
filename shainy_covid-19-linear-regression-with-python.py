# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import plotly.offline as py

import seaborn as sb

import matplotlib.dates as dates



import datetime as dt

from itertools import cycle, islice

py.init_notebook_mode(connected=True)



import plotly.express as px

import plotly.graph_objects as go



from itertools import cycle, islice

from sklearn.linear_model import LinearRegression

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

display(train_data.head())

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

display(test_data.head())

df_sub=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")
sum_of_data = pd.pivot_table(train_data, values=['ConfirmedCases','Fatalities'], index=['Date'],aggfunc=np.sum)

display(sum_of_data.max())

train_data['NewConfirmedCases'] = train_data['ConfirmedCases'] - train_data['ConfirmedCases'].shift(1)

train_data['NewConfirmedCases'] = train_data['NewConfirmedCases'].fillna(0.0)

train_data['NewFatalities']     = train_data['Fatalities'] - train_data['Fatalities'].shift(1)

train_data['NewFatalities']     = train_data['NewFatalities'].fillna(0.0)#.astype(int)

train_data['MortalityRate']     = train_data['Fatalities'] / train_data['ConfirmedCases']

train_data['MortalityRate']     = train_data['MortalityRate'].fillna(0.0)

train_data['GrowthRate']        = train_data['NewConfirmedCases']/train_data['NewConfirmedCases'].shift(1)

train_data['GrowthRate']        = train_data['GrowthRate'].replace([-np.inf, np.inf],  0.0)

train_data['GrowthRate']        = train_data['GrowthRate'].fillna(0.0) 

display(train_data.head())

def ColumnInfo(df):

    n_province =  df['Province_State'].nunique()

    n_country  =  df['Country_Region'].nunique()

    n_days     =  df['Date'].nunique()

    start_date =  df['Date'].unique()[0]

    end_date   =  df['Date'].unique()[-1]

    return n_province, n_country, n_days, start_date, end_date



n_train = train_data.shape[0]

n_test = test_data.shape[0]



n_prov_train, n_count_train, n_train_days, start_date_train, end_date_train = ColumnInfo(train_data)

n_prov_test,  n_count_test,  n_test_days,  start_date_test,  end_date_test  = ColumnInfo(test_data)



print ('<==Train data==> \n # of Province_State: '+str(n_prov_train),', # of Country_Region:'+str(n_count_train), 

       ', Time Period: '+str(start_date_train)+' to '+str(end_date_train), '==> days:',str(n_train_days))

print("\n Countries with Province/State information:  ", train_data[train_data['Province_State'].isna()==False]['Country_Region'].unique())

print ('\n <==Test  data==> \n # of Province_State: '+str(n_prov_test),', # of Country_Region:'+str(n_count_test),

       ', Time Period: '+start_date_test+' to '+end_date_test, '==> days:',n_test_days)



df_test = test_data.loc[test_data.Date > '2020-04-03']

overlap_days = n_test_days - df_test.Date.nunique()

print('\n overlap days with training data: ', overlap_days, ', total days: ', n_train_days+n_test_days-overlap_days)



prob_confirm_check_train = train_data.ConfirmedCases.value_counts(normalize=True)

prob_fatal_check_train = train_data.Fatalities.value_counts(normalize=True)



n_confirm_train = train_data.ConfirmedCases.value_counts()[1:].sum()

n_fatal_train = train_data.Fatalities.value_counts()[1:].sum()



print('Percentage of confirmed case records = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_confirm_train, n_train, prob_confirm_check_train[1:].sum()*100))

print('Percentage of fatality records = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_fatal_train, n_train, prob_fatal_check_train[1:].sum()*100))



train_data_by_country = train_data.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum',

                                                                                         'GrowthRate':'last' })

#display(train_data_by_country.tail(10))

max_train_date = train_data['Date'].max()

train_data_by_country_confirm = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)').sort_values('ConfirmedCases', ascending=False)

train_data_by_country_confirm.set_index('Country_Region', inplace=True)



train_data_by_country_confirm.style.background_gradient(cmap='PuBu_r').format({'ConfirmedCases': "{:.0f}", 'GrowthRate': "{:.2f}"})
discrete_col = list(islice(cycle(['purple', 'r', 'g', 'k', 'b', 'c', 'm']), None, len(train_data_by_country_confirm.head(30))))

plt.rcParams.update({'font.size': 22})

train_data_by_country_confirm.head(20).plot(figsize=(20,15), kind='barh', color=discrete_col)

plt.legend(["Confirmed Cases", "Fatalities"]);

plt.xlabel("Number of Covid-19 Affectees")

plt.title("First 20 Countries with Highest Confirmed Cases")

ylocs, ylabs = plt.yticks()

for i, v in enumerate(train_data_by_country_confirm.head(20)["ConfirmedCases"][:]):

    plt.text(v+0.01, ylocs[i]-0.25, str(int(v)), fontsize=12)

for i, v in enumerate(train_data_by_country_confirm.head(20)["Fatalities"][:]):

    if v > 0: #disply for only >300 fatalities

        plt.text(v+0.01,ylocs[i]+0.1,str(int(v)),fontsize=12) 

def reformat_time(reformat, ax):

    ax.xaxis.set_major_locator(dates.WeekdayLocator())

    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))    

    if reformat: #reformat again if you wish

        date_list = train_data_by_date.reset_index()["Date"].tolist()

        x_ticks = [dt.datetime.strftime(t,'%Y-%m-%d') for t in date_list]

        x_ticks = [tick for i,tick in enumerate(x_ticks) if i%8==0 ]# split labels into same number of ticks as by pandas

        ax.set_xticklabels(x_ticks, rotation=90)

    # cosmetics

    ax.yaxis.grid(linestyle='dotted')

    ax.spines['right'].set_color('none')

    ax.spines['top'].set_color('none')

    ax.spines['left'].set_color('none')

    ax.spines['bottom'].set_color('none')



train_data['Date'] = pd.to_datetime(train_data['Date'])

train_data_by_date = train_data.groupby(['Date'],as_index=True).agg({'ConfirmedCases': 'sum','Fatalities': 'sum', 

                                                                     'NewConfirmedCases':'sum', 'NewFatalities':'sum', 'MortalityRate':'mean'})

num0 = train_data_by_date._get_numeric_data() 

num0[num0 < 0.0] = 0.0

#display(train_data_by_date.head())



## ======= Sort by countries with fatalities > 500 ========      

        

   

train_data_by_country_max = train_data.groupby(['Country_Region'],as_index=True).agg({'ConfirmedCases': 'max', 'Fatalities': 'max'})

train_data_by_country_fatal = train_data_by_country_max[train_data_by_country_max['Fatalities']>500]

train_data_by_country_fatal = train_data_by_country_fatal.sort_values(by=['Fatalities'],ascending=False).reset_index()

#display(train_data_by_country_fatal.head(20))



df_merge_by_country = pd.merge(train_data,train_data_by_country_fatal['Country_Region'],on=['Country_Region'],how='inner')

df_max_fatality_country = df_merge_by_country.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum',

                                                                                                     'Fatalities': 'sum',

                                                                                                     'NewConfirmedCases':'sum',

                                                                                                     'NewFatalities':'sum',

                                                                                                     'MortalityRate':'mean'})



num1 = df_max_fatality_country._get_numeric_data() 

num1[num1 < 0.0] = 0.0

df_max_fatality_country.set_index('Date',inplace=True)

#display(df_max_fatality_country.head(20))

     





countries = train_data_by_country_fatal['Country_Region'].unique()



plt.rcParams.update({'font.size': 16})



fig,(ax0,ax1) = plt.subplots(1,2,figsize=(15, 8))

fig,(ax2,ax3) = plt.subplots(1,2,figsize=(15, 8))#,sharey=True)



train_data_by_date.ConfirmedCases.plot(ax=ax0, x_compat=True, title='Confirmed Cases Globally', legend='Confirmed Cases',

                                       color=discrete_col)#, logy=True)

reformat_time(0,ax0)

train_data_by_date.NewConfirmedCases.plot(ax=ax0, x_compat=True, linestyle='dotted', legend='New Confirmed Cases',

                                          color=discrete_col)#, logy=True)

reformat_time(0,ax0)



train_data_by_date.Fatalities.plot(ax=ax2, x_compat=True, title='Fatalities Globally', legend='Fatalities', color='r')

reformat_time(0,ax2)

train_data_by_date.NewFatalities.plot(ax=ax2, x_compat=True, linestyle='dotted', legend='Daily Deaths',color='r')#tell pandas not to use its own datetime format

reformat_time(0,ax2)



for country in countries:

    match = df_max_fatality_country.Country_Region==country

    df_fatality_by_country = df_max_fatality_country[match] 

    df_fatality_by_country.ConfirmedCases.plot(ax=ax1, x_compat=True, title='Confirmed Cases Nationally')

    reformat_time(0,ax1)

    df_fatality_by_country.Fatalities.plot(ax=ax3, x_compat=True, title='Fatalities Nationally')

    reformat_time(0,ax3)

    

ax1.legend(countries)

ax3.legend(countries)

fig = plt.figure()

fig,(ax4,ax5) = plt.subplots(1,2,figsize=(20, 8))

#train_data_by_date.loc[(train_data_by_date.ConfirmedCases > 200)]#useless, its already summed.

train_data_by_date.MortalityRate.plot(ax=ax4, x_compat=True, legend='Mortality Rate',color='purple')#tell pandas not to use its own datetime format

reformat_time(0,ax4)



for num, country in enumerate(countries):

    match = df_max_fatality_country.Country_Region==country 

    df_fatality_by_country = df_max_fatality_country[match] 

    df_fatality_by_country.MortalityRate.plot(ax=ax5, x_compat=True, title='Average Mortality Rate Nationally')    

    reformat_time(0,ax5)



ax5.legend(countries, loc='center left',bbox_to_anchor=(1.0, 0.5))     

 
train_data_by_max_date = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)')

train_data_by_max_date.loc[:, 'MortalityRate'] = train_data_by_max_date.loc[:,'Fatalities']/train_data_by_max_date.loc[:,'ConfirmedCases']

train_data_by_mortality = train_data_by_max_date.sort_values('MortalityRate', ascending=False)

train_data_by_mortality.set_index('Country_Region', inplace=True)

#display(train_data_by_mortality.head())



palette = plt.get_cmap('PuRd_r')

rainbow_col = [palette(1.*i/20.0) for i in range(20)]



train_data_by_mortality.MortalityRate.head(20).plot(figsize=(15,10), kind='barh', color=rainbow_col)

plt.xlabel("Mortality Rate")

plt.title("First 20 Countries with Highest Mortality Rate")

ylocs, ylabs = plt.yticks() 
print(f"Unique Countries: {len(train_data.Country_Region.unique())}")



train_dates=list(train_data.Date.unique())

print(f"Period : {len(train_data.Date.unique())} days")

print(f"From : {train_data.Date.min()} To : {train_data.Date.max()}")
print(f"Unique Regions: {train_data.shape[0]/75}")

train_data.Country_Region.value_counts()
print(f"Number of rows without Country_Region : {train_data.Country_Region.isna().sum()}")



train_data["UniqueRegion"]=train_data.Country_Region

train_data.UniqueRegion[train_data.Province_State.isna()==False]=train_data.Province_State+" , "+train_data.Country_Region

train_data[train_data.Province_State.isna()==False]
# delete the unwanted columns/attributes.

train_data.drop(labels=["Id","Province_State","Country_Region"], axis=1, inplace=True)

train_data
test_dates=list(test_data.Date.unique())

print(f"Period :{len(test_data.Date.unique())} days")

print(f"From : {test_data.Date.min()} To : {test_data.Date.max()}")

print(f"Total Regions : {test_data.shape[0]/43}")





test_data["UniqueRegion"]=test_data.Country_Region

test_data.UniqueRegion[test_data.Province_State.isna()==False]=test_data.Province_State+" , "+test_data.Country_Region

test_data.drop(labels=["Province_State","Country_Region"], axis=1, inplace=True)

len(test_data.UniqueRegion.unique())

only_train_dates=set(train_dates)-set(test_dates)

print("Only train dates : ",len(only_train_dates))

#dates in train and test

intersection_dates=set(test_dates)&set(train_dates)

print("Intersection dates : ",len(intersection_dates))

#dates in only test

only_test_dates=set(test_dates)-set(train_dates)

print("Only Test dates : ",len(only_test_dates))

df_test_temp=pd.DataFrame()

df_test_temp["Date"]=test_data.Date

df_test_temp["ConfirmedCases"]=0.0

df_test_temp["Fatalities"]=0.0

df_test_temp["UniqueRegion"]=test_data.UniqueRegion

df_test_temp["Delta"]=1.0       

     

display(df_test_temp) 
%%time

final_df=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","UniqueRegion"])



for region in train_data.UniqueRegion.unique():

    df_temp=train_data[train_data.UniqueRegion==region].reset_index()

    df_temp["Delta"]=1.0

    size_train=df_temp.shape[0]

    for i in range(1,df_temp.shape[0]):

        if(df_temp.ConfirmedCases[i-1]>0):

            df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]

            #number of days for delta trend

    n=5     



    #delta as average of previous n days

    delta_avg=df_temp.tail(n).Delta.mean()



    #delta as trend for previous n days

    delta_list=df_temp.tail(n).Delta

    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()

    df_test_app=df_test_temp[df_test_temp.UniqueRegion==region]

    

    X=np.arange(1,n+1).reshape(-1,1)

    Y=delta_list

    model=LinearRegression()

    model.fit(X,Y)



    df_temp=pd.concat([df_temp,df_test_app])

    df_temp=df_temp.reset_index()



    for i in range (size_train, df_temp.shape[0]):

        n=n+1

        df_temp.Delta[i]=max(1,model.predict(np.array([n]).reshape(-1,1))[0])

        df_temp.ConfirmedCases[i]=round(df_temp.ConfirmedCases[i-1]*df_temp.Delta[i],0)

        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)





    size_test=df_temp.shape[0]-df_test_temp[df_test_temp.UniqueRegion==region].shape[0]



    df_temp=df_temp.iloc[size_test:,:]

    

    df_temp=df_temp[["Date","ConfirmedCases","Fatalities","UniqueRegion"]]

    final_df=pd.concat([final_df,df_temp], ignore_index=True)

    

final_df.shape

    
df_sub.Fatalities=final_df.Fatalities

df_sub.ConfirmedCases=final_df.ConfirmedCases

df_sub.to_csv("submission.csv", index=None)