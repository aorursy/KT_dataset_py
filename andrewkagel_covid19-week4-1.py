# Author: Anirban Mitra(amitra@cs.iitr.ac.in)



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.figure_factory as ff

import plotly.io as pio

pio.templates.default = "plotly_dark"



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import datetime

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

train_df = train_df.drop(['Id'],axis=1)

train_df.rename(columns={'Country_Region':'Country'}, inplace=True)

test_df.rename(columns={'Country_Region':'Country'}, inplace=True)

train_df['Province_State'].fillna("None*", inplace = True)

test_df['Province_State'].fillna("None*", inplace = True)

renameCountryNames = {

    "Congo (Brazzaville)": "Congo1",

    "Congo (Kinshasa)": "Congo2",

    "Cote d'Ivoire": "Côte d'Ivoire",

    "Czechia": "Czech Republic (Czechia)",

    "Korea, South": "South Korea",

    "Saint Kitts and Nevis": "Saint Kitts & Nevis",

    "Saint Vincent and the Grenadines": "St. Vincent & Grenadines",

    "Taiwan*": "Taiwan",

    "US": "United States"

}

#train_df_modified.replace({'Country': renameCountryNames}, inplace=True)

train_df.replace({'Country': renameCountryNames}, inplace=True)

test_df.replace({'Country': renameCountryNames}, inplace=True)

specific_countries = ['United States', 'United Kingdom','Netherlands']

days_df = train_df['Date'].apply(lambda dt: datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.datetime.strptime('2020-01-21', '%Y-%m-%d')).apply(lambda x : str(x).split()[0]).astype(int)

train_df['Days'] = days_df

days_df = test_df['Date'].apply(lambda dt: datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.datetime.strptime('2020-01-21', '%Y-%m-%d')).apply(lambda x : str(x).split()[0]).astype(int)

test_df['Days'] = days_df



all_countries = train_df['Country'].unique()



display(train_df.tail())

display(test_df.head())



ww_df = train_df.groupby('Date')[['ConfirmedCases', 'Fatalities']].sum().reset_index()

ww_df['new_case'] = ww_df['ConfirmedCases'] - ww_df['ConfirmedCases'].shift(1)

ww_df['new_deaths'] = ww_df['Fatalities'] - ww_df['Fatalities'].shift(1)

country_df = train_df.groupby(['Date', 'Country'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()

target_date = country_df['Date'].max()

train_end_day = train_df['Days'].max()

test_start_day = test_df['Days'].min()

test_end_day = test_df['Days'].max()

display(country_df[country_df['Country']=='France'][80:])
py.init_notebook_mode()

top_country_df = country_df.query('(Date == @target_date) & (ConfirmedCases > 2000)').sort_values('ConfirmedCases', ascending=False)

print(len(top_country_df))

top_country_melt_df = pd.melt(top_country_df, id_vars='Country', value_vars=['ConfirmedCases', 'Fatalities'])

display(top_country_df.head())

display(top_country_melt_df.head())

fig = px.bar(top_country_melt_df.iloc[::-1],

             x='value', y='Country', color='variable', barmode='group',

             title=f'Confirmed Cases/Deaths on {target_date}', text='value', height=1500, orientation='h')

fig.show()
country_province_df = train_df[train_df['Country']=='United States'].groupby(['Date', 'Province_State'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()



top_province_df = country_province_df.query('(Date == @target_date)').sort_values('ConfirmedCases', ascending=False)

top30_provinces = top_province_df.sort_values('ConfirmedCases', ascending=False).iloc[:30]['Province_State'].unique()



country_province_df['prev_cases'] = country_province_df.groupby('Province_State')['ConfirmedCases'].shift(1)

country_province_df['New Case'] = country_province_df['ConfirmedCases'] - country_province_df['prev_cases']

country_province_df['New Case'].fillna(0, inplace=True)

country_province_df['prev_deaths'] = country_province_df.groupby('Province_State')['Fatalities'].shift(1)

country_province_df['New Death'] = country_province_df['Fatalities'] - country_province_df['prev_deaths']

country_province_df['New Death'].fillna(0, inplace=True)



display(country_province_df[-30:])

for province in top30_provinces:

    present_country_df = country_province_df[country_province_df['Province_State']==province]

    px.bar(present_country_df,

              x='Date', y='New Case', color='Province_State',

              title=f'United States : DAILY NEW Confirmed cases in '+province).show()
top30_countries = top_country_df.sort_values('ConfirmedCases', ascending=False).iloc[:30]['Country'].unique()

display(country_df[:20])

country_df['prev_cases'] = country_df.groupby('Country')['ConfirmedCases'].shift(1)

country_df['New Case'] = country_df['ConfirmedCases'] - country_df['prev_cases']

country_df['New Case'].fillna(0, inplace=True)

country_df['prev_deaths'] = country_df.groupby('Country')['Fatalities'].shift(1)

country_df['New Death'] = country_df['Fatalities'] - country_df['prev_deaths']

country_df['New Death'].fillna(0, inplace=True)

top30_country_df = country_df[country_df['Country'].isin(top30_countries)]

display(country_df[:10])

for country in top30_countries:

    present_country_df = top30_country_df[top30_country_df['Country']==country]

    px.bar(present_country_df,

              x='Date', y='New Case', color='Country',

              title=f'DAILY NEW Confirmed cases in '+country).show()
def get_time_series(df,country_name,insert=False):

    # for some countries, data is spread over several Provinces

    if df[df['Country'] == country_name]['Province_State'].nunique() > 1:

        country_table = df[df['Country'] == country_name] 

        if insert:

          country_df = pd.DataFrame(pd.pivot_table(country_table, values = ['ConfirmedCases','Fatalities','Days'],

                               index='Date', aggfunc=sum).to_records())

          return country_df.set_index('Date')[['ConfirmedCases','Fatalities']] 

        return country_table.set_index('Date')[['Province_State','ConfirmedCases','Fatalities','Days']]

    df = df[(df['Country'] == country_name)]

    return df.set_index('Date')[['ConfirmedCases','Fatalities','Days']]





def get_time_series_province(province):

    # for some countries, data is spread over several Provinces

    df = full_table[(full_table['Province_State'] == province)]

    return df.set_index('Date')[['ConfirmedCases','Fatalities']]



province_country_dfs = {}

no_province_country_dfs = {}

absent_country_in_age_data_dfs = {}



province_country_dfs_list=[]

no_province_country_dfs_list=[]

absent_country_in_age_data_dfs_list=[]



province_countries=train_df[train_df['Province_State']!="None*"]['Country'].unique()

no_province_countries=train_df[train_df['Province_State']=="None*"]['Country'].unique()

no_province_countries= [x for x in no_province_countries if x not in province_countries] #exclude countries like denmark



# print(province_countries)

# print(no_province_countries)



for country  in province_countries:

    province_country_dfs[country] = get_time_series(train_df,country)

for country  in no_province_countries:

    no_province_country_dfs[country] = get_time_series(train_df,country)

    

print([x for x in no_province_countries if x  in top30_countries])

print([x for x in province_countries if x  in top30_countries])





assert(len([x for x in all_countries if x not in list(no_province_countries)+list(province_countries)])==0)

display(province_country_dfs['United States'])
import datetime



from numpy import array

import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense



print(tf.__version__)
train_df['Province_State'].fillna("None*", inplace = True) 

test_df['Province_State'].fillna("None*", inplace = True) 

test_df["ConfirmedCases"] = np.nan

test_df["Fatalities"] = np.nan

test_df = test_df.set_index('Date')

display(test_df)

prediction={}
def build_model(n_steps):

    model = Sequential()

    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, 1)))

    model.add(LSTM(50, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='RMSprop', loss='mse')

    return model



def split_sequence(sequence, n_steps):

	X, y = list(), list()

	for i in range(len(sequence)):

		# find the end of this pattern

		end_ix = i + n_steps

		# check if we are beyond the sequence

		if end_ix > len(sequence)-1:

			break

		# gather input and output parts of the pattern

		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

		X.append(seq_x)

		y.append(seq_y)

	return array(X), array(y)
display(train_df[train_df['Country']=='Afghanistan'].tail())

print(province_countries)

max_train_date = train_df['Date'].max()
for country in province_countries:

 #if country in ['France','Denmark','United Kingdom']:

 #if country in ['United States']:

    current_country_provinces = province_country_dfs[country]['Province_State'].unique()

    for province in current_country_provinces:

      #if province in ['New York','Rhode Island','Texas']:

        current_considered_country_df = province_country_dfs[country][province_country_dfs[country]['Province_State']==province][['ConfirmedCases','Fatalities','Days']].reset_index()

        print(country+" "+province)

        current_considered_country_df_copy=current_considered_country_df

        

        for i in range(train_end_day-test_start_day+1):

            test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==i+test_start_day), 'ConfirmedCases'] = current_considered_country_df.loc[current_considered_country_df['Days'] == i+test_start_day, 'ConfirmedCases'].values[0]

            test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==i+test_start_day), 'Fatalities'] = current_considered_country_df.loc[current_considered_country_df['Days'] == i+test_start_day, 'Fatalities'].values[0]

        

        indexNames = current_considered_country_df[ current_considered_country_df['ConfirmedCases'] == 0 ].index

        current_considered_country_df.drop(indexNames , inplace=True)

        

#         x_train_full= current_considered_country_df[['Days','Population','Density','Below14','15-64','Over65']].to_numpy()

        cases_train = np.diff(current_considered_country_df['ConfirmedCases'].to_numpy())

        fatalities_train = np.diff(current_considered_country_df['Fatalities'].to_numpy())

        #there are many anomalies in data,like new cases per day are negative, which refers to previous incorrect reporting or wrongful data entry,

        # nevertheless we need to consider this as is or trim down to zero

        cases_train[cases_train<0] = 0

        fatalities_train[fatalities_train<0] = 0

        

        fatal_rate = 0.0

        if(current_considered_country_df['Fatalities'].to_numpy()[-1]>0):

            fatal_rate = current_considered_country_df['Fatalities'].to_numpy()[-1]/current_considered_country_df['ConfirmedCases'].to_numpy()[-1]

            print("fatal rate is: "+str(fatal_rate))

        

        #get last 15 days increasing average

        cases_increase_avg = 0

        days=0

        for i in range(len(cases_train)-1):

            cases_increase_avg +=  (cases_train[i+1] -cases_train[i])

            days+=1

        if(days>0):

            cases_increase_avg = int(cases_increase_avg/days)

            

        days=0

        fatal_increase_avg=0

        for i in range(len(fatalities_train)-1):

            fatal_increase_avg +=  (fatalities_train[i+1] -fatalities_train[i])

            days+=1

        if(days>0):

            fatal_increase_avg = int(fatal_increase_avg/days)

        del current_considered_country_df

        n_steps = max(int(len(cases_train)*0.1),3)

#         print("case increase avg: "+str(cases_increase_avg)+" "+ str(days))

#         print("cases train len: "+str(len(cases_train)))

        

        # get avg per day increase trend:

        avg_weekly_per_day_case = []

        avg_window = 4

        avg_step = 2 

        if int(len(cases_train)/avg_window) > avg_step:

            for i in range(int(len(cases_train)/avg_window)):

                    temp_list = cases_train[i*avg_window:i*avg_window+avg_window]

                    avg_weekly_per_day_case.append(np.sum(temp_list)/len(temp_list))

            avg_weekly_per_day_case = np.array(avg_weekly_per_day_case)

            #print("window avg: "+str(avg_weekly_per_day_case))

            # predict weekly avg

            X_weekly_avg_val, y_weekly_avg_val = split_sequence(avg_weekly_per_day_case,avg_step)

            X_weekly_avg_val = np.reshape(X_weekly_avg_val, (X_weekly_avg_val.shape[0],X_weekly_avg_val.shape[1],1))  #####

            model_weekly_avg=build_model(avg_step)

            model_weekly_avg.fit(X_weekly_avg_val, y_weekly_avg_val, epochs=50, verbose=0)

            new_entry_avg=X_weekly_avg_val[len(X_weekly_avg_val)-1]

            for i in range(int(30/avg_window)+1):

                weekly_avg_predict_next = model_weekly_avg.predict(np.reshape(new_entry_avg,(1,avg_step,1)), verbose=0).astype(int)

                avg_weekly_per_day_case = np.append(avg_weekly_per_day_case,weekly_avg_predict_next[0])



                last_series = np.reshape(new_entry_avg,(1,avg_step,1))

                new_entry_avg=np.delete(last_series,[0])

                new_entry_avg = np.insert(new_entry_avg,avg_step-1,weekly_avg_predict_next[0])

            #print("7 weeks avg after predict: "+str(avg_weekly_per_day_case))

        

        X_cases_val, y_cases_val = split_sequence(cases_train,n_steps)

        X_cases_val = np.reshape(X_cases_val, (X_cases_val.shape[0],X_cases_val.shape[1],1))

        

        X_fatal_val, y_fatal_val = split_sequence(fatalities_train,n_steps)

        X_fatal_val = np.reshape(X_fatal_val, (X_fatal_val.shape[0],X_fatal_val.shape[1],1))

        

        assert(len(X_fatal_val)==len(X_cases_val))

        assert(len(y_fatal_val)==len(y_cases_val))

        

        model_cases=build_model(n_steps)

        model_cases.fit(X_cases_val, y_cases_val, epochs=50, verbose=0)

        cases_predict_next = model_cases.predict(np.reshape(X_cases_val[len(X_cases_val)-1],(1,n_steps,1)), verbose=0).astype(int)

        cases_predict_next[0] =  np.array([max(0,cases_predict_next[0])])

        

        model_fatalities=build_model(n_steps)

        model_fatalities.fit(X_fatal_val, y_fatal_val, epochs=50, verbose=0)

        fatality_predict_next = model_fatalities.predict(np.reshape(X_fatal_val[len(X_fatal_val)-1],(1,n_steps,1)), verbose=0).astype(int)

        fatality_predict_next[0] =  np.array([max(0,fatality_predict_next[0])])

        fatality_predict_next[0] = np.array([max(fatality_predict_next[0],cases_predict_next[0]*fatal_rate)])

        

        test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==train_end_day+1), 'ConfirmedCases'] = test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==train_end_day), 'ConfirmedCases'].values[0] + cases_predict_next[0]

        test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==train_end_day+1), 'Fatalities'] = test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==train_end_day), 'Fatalities'].values[0] + fatality_predict_next[0]

            

#         print(cases_predict_next[0])

#         print(X_cases_val[len(X_cases_val)-1])

        new_entry_cases=X_cases_val[len(X_cases_val)-1]

        new_entry_fatal=X_fatal_val[len(X_fatal_val)-1]

        

        for i in range(test_end_day-train_end_day-1):

            

          # Cases

          last_series = np.reshape(new_entry_cases,(1,n_steps,1))

          new_entry_cases=np.delete(last_series,[0])

          new_entry_cases = np.insert(new_entry_cases,n_steps-1,cases_predict_next[0])

        

          cases_predict_next = model_cases.predict(np.reshape(new_entry_cases,(1,n_steps,1)), verbose=0).astype(int)

          if(cases_predict_next[0]-new_entry_cases[n_steps-1]>cases_increase_avg):

            cases_predict_next =  np.array([max(0,new_entry_cases[n_steps-1]+cases_increase_avg)])

          if (province in ['Kentucky','New Mexico','Sint Maarten','Cayman Islands','Isle of Man']):

            cases_predict_next[0] = avg_weekly_per_day_case[-int(30/avg_window)-1+int(i/avg_window)]

          cases_predict_next[0] =  np.array([max(0,cases_predict_next[0])])

#           print(np.array(new_entry_cases))

#           print(cases_predict_next[0])



          # fatality

          last_series = np.reshape(new_entry_fatal,(1,n_steps,1))

          new_entry_fatal=np.delete(last_series,[0])

          new_entry_fatal = np.insert(new_entry_fatal,n_steps-1,fatality_predict_next[0])

        

          fatality_predict_next = model_fatalities.predict(np.reshape(new_entry_fatal,(1,n_steps,1)), verbose=0).astype(int)

          if(fatality_predict_next[0]-new_entry_fatal[n_steps-1]>fatal_increase_avg):

            fatality_predict_next[0] =  max(0,new_entry_fatal[n_steps-1]+fatal_increase_avg)

          fatality_predict_next[0] =  np.array([max(0,fatality_predict_next[0])])

          fatality_predict_next[0] = np.array([max(fatality_predict_next[0],int(cases_predict_next[0]*fatal_rate))])

        

          test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==i+train_end_day+2), 'ConfirmedCases'] = test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==i+train_end_day+1), 'ConfirmedCases'].values[0] + cases_predict_next[0]

          test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==i+train_end_day+2), 'Fatalities'] = test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==i+train_end_day+1), 'Fatalities'].values[0] + fatality_predict_next[0]

        del model_fatalities

        del model_cases
country_province_df = test_df[test_df['Country']=='United States'].groupby(['Date', 'Province_State'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()



country_province_df['prev_cases'] = country_province_df.groupby('Province_State')['ConfirmedCases'].shift(1)

country_province_df['New Case'] = country_province_df['ConfirmedCases'] - country_province_df['prev_cases']

country_province_df['New Case'].fillna(0, inplace=True)

country_province_df['prev_deaths'] = country_province_df.groupby('Province_State')['Fatalities'].shift(1)

country_province_df['New Death'] = country_province_df['Fatalities'] - country_province_df['prev_deaths']

country_province_df['New Death'].fillna(0, inplace=True)



display(country_province_df.head())

for province in top30_provinces:

    present_country_df = country_province_df[country_province_df['Province_State']==province]

    px.bar(present_country_df,

              x='Date', y='New Case', color='Province_State',

              title=f'United States : DAILY NEW Confirmed cases in '+province).show()
test_country_df = test_df.groupby(['Date', 'Country'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()

display(test_country_df[test_country_df['Country']=='Australia'][:20])

for country in [x for x in province_countries if x  in top30_countries]:

    present_country_df = test_country_df[test_country_df['Country']==country].reset_index()

    present_country_df['prev_cases'] = present_country_df.groupby('Country')['ConfirmedCases'].shift(1)

    present_country_df['New Case'] = present_country_df['ConfirmedCases'] - present_country_df['prev_cases']

    present_country_df['New Case'].fillna(0, inplace=True)

    present_country_df['prev_deaths'] = present_country_df.groupby('Country')['Fatalities'].shift(1)

    present_country_df['New Death'] = present_country_df['Fatalities'] - present_country_df['prev_deaths']

    present_country_df['New Death'].fillna(0, inplace=True)

    #display(present_country_df[:20])

    px.bar(present_country_df,

              x='Date', y='New Case', color='Country',

              title=f'DAILY NEW Confirmed cases in '+country).show()

for country in no_province_countries:

  #if country in ['South Korea', 'Norway', 'Peru', 'Poland', 'Portugal', 'Russia', 'Spain', 'Sweden', 'Switzerland', 'Turkey']:

    current_considered_country_df = no_province_country_dfs[country][['ConfirmedCases','Fatalities','Days']].reset_index()

    print(country)

    

    for i in range(train_end_day-test_start_day+1):

            test_df.loc[(test_df['Country']==country) & (test_df['Days']==i+test_start_day), 'ConfirmedCases'] = current_considered_country_df.loc[current_considered_country_df['Days'] == i+test_start_day, 'ConfirmedCases'].values[0]

            test_df.loc[(test_df['Country']==country) & (test_df['Days']==i+test_start_day), 'Fatalities'] = current_considered_country_df.loc[current_considered_country_df['Days'] == i+test_start_day, 'Fatalities'].values[0]



    indexNames = current_considered_country_df[ current_considered_country_df['ConfirmedCases'] == 0 ].index

    current_considered_country_df.drop(indexNames , inplace=True)

    

    cases_train = np.diff(current_considered_country_df['ConfirmedCases'].to_numpy())

    fatalities_train = np.diff(current_considered_country_df['Fatalities'].to_numpy())

    fatal_rate = 0.0

    if(current_considered_country_df['Fatalities'].to_numpy()[-1]>0):

        fatal_rate = current_considered_country_df['Fatalities'].to_numpy()[-1]/current_considered_country_df['ConfirmedCases'].to_numpy()[-1]

        print("fatal rate is: "+str(fatal_rate))

    

    

    #get previous days increasing average

    cases_increase_avg = 0

    days=0

    for i in range(len(cases_train)-1):

        cases_increase_avg +=  (cases_train[i+1] -cases_train[i])

        days+=1

    if(days>0):

        cases_increase_avg = int(cases_increase_avg/days)

    fatal_increase_avg = 0

    days=0

    for i in range(len(fatalities_train)-1):

        fatal_increase_avg +=  (fatalities_train[i+1] -fatalities_train[i])

        days+=1

    if(days>0):

        fatal_increase_avg = int(fatal_increase_avg/days)

    

    n_steps = max(int(len(cases_train)*0.15),2)

#     print(len(cases_train))

#     print("cases increase avg: "+str(cases_increase_avg)+" "+ str(days))

#     print("cases train len: "+str(len(cases_train)))

    

    # get avg per day increase trend:

    avg_weekly_per_day_case = []

    avg_window = 4

    avg_step = 2 

#     if int(len(cases_train)/avg_window) > avg_step:

#         for i in range(int(len(cases_train)/avg_window)):

#                 temp_list = cases_train[i*avg_window:i*avg_window+avg_window]

#                 avg_weekly_per_day_case.append(np.sum(temp_list)/len(temp_list))

#         avg_weekly_per_day_case = np.array(avg_weekly_per_day_case)

#         #print("window avg: "+str(avg_weekly_per_day_case))

#         # predict weekly avg

#         X_weekly_avg_val, y_weekly_avg_val = split_sequence(avg_weekly_per_day_case,avg_step)

#         X_weekly_avg_val = np.reshape(X_weekly_avg_val, (X_weekly_avg_val.shape[0],X_weekly_avg_val.shape[1],1))  #####

#         model_weekly_avg=build_model(avg_step)

#         model_weekly_avg.fit(X_weekly_avg_val, y_weekly_avg_val, epochs=50, verbose=0)

#         new_entry_avg=X_weekly_avg_val[len(X_weekly_avg_val)-1]

#         for i in range(int(30/avg_window)+1):

#             weekly_avg_predict_next = model_weekly_avg.predict(np.reshape(new_entry_avg,(1,avg_step,1)), verbose=0).astype(int)

#             avg_weekly_per_day_case = np.append(avg_weekly_per_day_case,weekly_avg_predict_next[0])



#             last_series = np.reshape(new_entry_avg,(1,avg_step,1))

#             new_entry_avg=np.delete(last_series,[0])

#             new_entry_avg = np.insert(new_entry_avg,avg_step-1,weekly_avg_predict_next[0])

#         del weekly_avg_predict_next

#         del model_weekly_avg

#         #print("7 weeks avg after predict: "+str(avg_weekly_per_day_case))

    

    X_cases_val, y_cases_val = split_sequence(cases_train,n_steps)

    X_cases_val = np.reshape(X_cases_val, (X_cases_val.shape[0],X_cases_val.shape[1],1))

    model_cases=build_model(n_steps)

    model_cases.fit(X_cases_val, y_cases_val, epochs=50, verbose=0)

    cases_predict_next = model_cases.predict(np.reshape(X_cases_val[len(X_cases_val)-1],(1,n_steps,1)), verbose=0).astype(int)

    cases_predict_next[0] =  np.array([max(0,cases_predict_next[0])])

    

    X_fatal_val, y_fatal_val = split_sequence(fatalities_train,n_steps)    

    X_fatal_val = np.reshape(X_fatal_val, (X_fatal_val.shape[0],X_fatal_val.shape[1],1))

    model_fatalities=build_model(n_steps)

    model_fatalities.fit(X_fatal_val, y_fatal_val, epochs=50, verbose=0)

    fatality_predict_next = model_fatalities.predict(np.reshape(X_fatal_val[len(X_fatal_val)-1],(1,n_steps,1)), verbose=0).astype(int)

    fatality_predict_next[0] =  np.array([max(0,fatality_predict_next[0])])

    fatality_predict_next[0] = np.array([max(fatality_predict_next[0],cases_predict_next[0]*fatal_rate)])

    

    test_df.loc[(test_df['Country']==country) & (test_df['Days']==train_end_day+1), 'ConfirmedCases'] = test_df.loc[(test_df['Country']==country) & (test_df['Days']==train_end_day), 'ConfirmedCases'].values[0] + cases_predict_next[0]

    test_df.loc[(test_df['Country']==country) & (test_df['Days']==train_end_day+1), 'Fatalities'] = test_df.loc[(test_df['Country']==country) & (test_df['Days']==train_end_day), 'Fatalities'].values[0] + fatality_predict_next[0]



#     print(cases_predict_next[0])

#     print(X_cases_val[len(X_cases_val)-1])

    

    new_entry_cases=X_cases_val[len(X_cases_val)-1]

    new_entry_fatal=X_fatal_val[len(X_fatal_val)-1]

        

    for i in range(test_end_day-train_end_day-1):



      # after 7 days rates will mostly decline in these countries

      last_series = np.reshape(new_entry_cases,(1,n_steps,1))

      new_entry_cases=np.delete(last_series,[0])

      new_entry_cases = np.insert(new_entry_cases,n_steps-1,cases_predict_next[0])



      cases_predict_next = model_cases.predict(np.reshape(new_entry_cases,(1,n_steps,1)), verbose=0).astype(int)

      if(cases_predict_next[0]-new_entry_cases[n_steps-1]>cases_increase_avg):

        cases_predict_next[0] =  np.array([max(0,new_entry_cases[n_steps-1]+cases_increase_avg)])

      cases_predict_next[0] =  np.array([max(0,cases_predict_next[0])])

#       print(np.array(new_entry_cases))

#       print(cases_predict_next[0])



      last_series = np.reshape(new_entry_fatal,(1,n_steps,1))

      new_entry_fatal=np.delete(last_series,[0])

      new_entry_fatal = np.insert(new_entry_fatal,n_steps-1,fatality_predict_next[0])

      #X_fatal_val = np.append(X_fatal_val, np.reshape(new_entry_fatal,(1,n_steps,1)), axis=0)



      fatality_predict_next = model_fatalities.predict(np.reshape(new_entry_fatal,(1,n_steps,1)), verbose=0).astype(int)

      if(fatality_predict_next[0]-new_entry_fatal[n_steps-1]>fatal_increase_avg):

        fatality_predict_next[0] =  max(0,new_entry_fatal[n_steps-1]+fatal_increase_avg) 

      fatality_predict_next[0] =  np.array([max(0,fatality_predict_next[0])])

      fatality_predict_next[0] = np.array([max(fatality_predict_next[0],int(cases_predict_next[0]*fatal_rate))])



      test_df.loc[(test_df['Country']==country)  & (test_df['Days']==i+train_end_day+2), 'ConfirmedCases'] = test_df.loc[(test_df['Country']==country)  & (test_df['Days']==i+train_end_day+1), 'ConfirmedCases'].values[0] + cases_predict_next[0]

      test_df.loc[(test_df['Country']==country) & (test_df['Days']==i+train_end_day+2), 'Fatalities'] = test_df.loc[(test_df['Country']==country)  & (test_df['Days']==i+train_end_day+1), 'Fatalities'].values[0] + fatality_predict_next[0]

    del model_fatalities

    del model_cases
test_df_copy = test_df

submission_df_copy = submission_df



submit = pd.DataFrame()

submit['ForecastId'] = test_df['ForecastId']

submit['ConfirmedCases'] = test_df['ConfirmedCases']

submit['Fatalities'] = test_df['Fatalities']

submit = submit.reset_index()

submit = submit.drop(['Date'], axis=1)

display(submit.tail())



submit.to_csv('submission.csv',index=False)