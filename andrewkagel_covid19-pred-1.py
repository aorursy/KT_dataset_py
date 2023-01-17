# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import datetime
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
submission_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
train_df = train_df.drop(['Id'],axis=1)
train_df.rename(columns={'Country_Region':'Country'}, inplace=True)
test_df.rename(columns={'Country_Region':'Country'}, inplace=True)
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

days_df = train_df['Date'].apply(lambda dt: datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.datetime.strptime('2020-01-21', '%Y-%m-%d')).apply(lambda x : str(x).split()[0]).astype(int)
train_df['Days'] = days_df
days_df = test_df['Date'].apply(lambda dt: datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.datetime.strptime('2020-01-21', '%Y-%m-%d')).apply(lambda x : str(x).split()[0]).astype(int)
test_df['Days'] = days_df

all_countries = train_df['Country'].unique()

display(train_df[train_df['Country']=='Congo1'])
display(test_df[test_df['Country']=='Congo1'])
def get_time_series(df,country_name,insert=False):
    # for some countries, data is spread over several Provinces
    if df[df['Country'] == country_name]['Province_State'].nunique() > 1:
        country_table = df[df['Country'] == country_name]
        country_table['Province_State'].fillna("None*", inplace = True) 
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

# x = train_df[~train_df['Province_State'].isnull()]['Country_Region'].value_counts()
some_countries = ['China', 'US', 'Italy', 'Spain', 'Germany', 'Iran','India', 'Australia', 'Korea', 'France', 'Switzerland','France','UK','Turkey']
#print(absent_country_in_age_data)

province_country_dfs = {}
no_province_country_dfs = {}
absent_country_in_age_data_dfs = {}

province_country_dfs_list=[]
no_province_country_dfs_list=[]
absent_country_in_age_data_dfs_list=[]

province_countries=train_df[train_df['Province_State'].isna()==False]['Country'].unique()
no_province_countries=train_df[train_df['Province_State'].isna()]['Country'].unique()
no_province_countries= [x for x in no_province_countries if x not in province_countries] #exclude countries like denmark

# print(province_countries)
# print(no_province_countries)

for country  in province_countries:
    province_country_dfs[country] = get_time_series(train_df,country)
#     province_country_dfs[country]['Below14'] = province_country_dfs[country]['Below14']*province_country_dfs[country]['Population']
#     province_country_dfs[country]['15-64'] = province_country_dfs[country]['15-64']*province_country_dfs[country]['Population']
#     province_country_dfs[country]['Over65'] = province_country_dfs[country]['Over65']*province_country_dfs[country]['Population']
    #province_country_dfs[country]['Country'] = country
    #rovince_country_dfs_list.append(province_country_dfs[country])
for country  in no_province_countries:
#     if country in absent_country_in_age_data:
#         absent_country_in_age_data_dfs[country] = get_time_series(train_df,country)
#         #province_country_dfs[country]['Country'] = country
#         absent_country_in_age_data_dfs_list.append(absent_country_in_age_data_dfs[country])
#         continue
    no_province_country_dfs[country] = get_time_series(train_df,country)
    #no_province_country_dfs[country]['Country'] = country
#     no_province_country_dfs[country]['Below14'] = no_province_country_dfs[country]['Below14']*no_province_country_dfs[country]['Population']
#     no_province_country_dfs[country]['15-64'] = no_province_country_dfs[country]['15-64']*no_province_country_dfs[country]['Population']
#     no_province_country_dfs[country]['Over65'] = no_province_country_dfs[country]['Over65']*no_province_country_dfs[country]['Population']
#     no_province_country_dfs_list.append(no_province_country_dfs[country])
# train_province_df = pd.concat(province_country_dfs_list)
# train_no_province_df = pd.concat(no_province_country_dfs_list)

# train_df_modified = pd.concat([train_province_df,train_no_province_df],sort=True)

assert(len([x for x in all_countries if x not in list(no_province_countries)+list(province_countries)])==0)
#train_df.groupby(by='Country_Region')[['ConfirmedCases','Fatalities']].sum()
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
display(test_df[test_df['Country']=='Afghanistan'])
print(province_countries)
for country in province_countries:
    current_country_provinces = province_country_dfs[country]['Province_State'].unique()
    for province in current_country_provinces:
        current_considered_country_df = province_country_dfs[country][province_country_dfs[country]['Province_State']==province][['ConfirmedCases','Fatalities','Days']].reset_index()
        print(country+" "+province + " " + str(len(current_considered_country_df)))
        
#         days_df = current_considered_country_df['Date'].apply(lambda dt: datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.datetime.strptime('2020-01-21', '%Y-%m-%d')).apply(lambda x : str(x).split()[0]).astype(int)
#         current_considered_country_df['Days'] = days_df
        current_considered_country_df_copy=current_considered_country_df
        
        for i in range(13):
            test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==i+65), 'ConfirmedCases'] = current_considered_country_df.loc[current_considered_country_df['Days'] == i+65, 'ConfirmedCases'].values[0]
            test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==i+65), 'Fatalities'] = current_considered_country_df.loc[current_considered_country_df['Days'] == i+65, 'Fatalities'].values[0]
        
        indexNames = current_considered_country_df[ current_considered_country_df['ConfirmedCases'] == 0 ].index
        current_considered_country_df.drop(indexNames , inplace=True)
        
#         x_train_full= current_considered_country_df[['Days','Population','Density','Below14','15-64','Over65']].to_numpy()
        cases_train = np.diff(current_considered_country_df['ConfirmedCases'].to_numpy())
        fatalities_train = np.diff(current_considered_country_df['Fatalities'].to_numpy())
        print(np.diff(cases_train))
        #get last 15 days increasing average
        cases_increase_avg = 0
        days=0
        for i in range(max(0,len(cases_train)-10),len(cases_train)-1):
            cases_increase_avg +=  (cases_train[i+1] -cases_train[i])
            days+=1
        if(days>0):
            cases_increase_avg = int(cases_increase_avg/days)
            
        days=0
        fatal_increase_avg=0
        for i in range(max(0,len(fatalities_train)-10),len(fatalities_train)-1):
            fatal_increase_avg +=  (fatalities_train[i+1] -fatalities_train[i])
            days+=1
        if(days>0):
            fatal_increase_avg = int(fatal_increase_avg/days)
        del current_considered_country_df
    
        n_steps = max(int(len(cases_train)*0.2),2)
        print(len(cases_train))
        print("avg: "+str(cases_increase_avg)+str(days))
        
        X_cases_val, y_cases_val = split_sequence(cases_train,n_steps)
        X_fatal_val, y_fatal_val = split_sequence(fatalities_train,n_steps)
        
        X_cases_val = np.reshape(X_cases_val, (X_cases_val.shape[0],X_cases_val.shape[1],1))
        X_fatal_val = np.reshape(X_fatal_val, (X_fatal_val.shape[0],X_fatal_val.shape[1],1))
        
        print(X_cases_val.shape)
#         X_cases_test, y_cases_test = split_sequence(cases_test_v, n_steps)
        assert(len(X_fatal_val)==len(X_cases_val))
        assert(len(y_fatal_val)==len(y_cases_val))
        
        model_cases=build_model(n_steps)
        model_cases.fit(X_cases_val, y_cases_val, epochs=50, verbose=0)
        cases_predict_next = model_cases.predict(np.reshape(X_cases_val[len(X_cases_val)-1],(1,n_steps,1)), verbose=0).astype(int)
        
        model_fatalities=build_model(n_steps)
        model_fatalities.fit(X_fatal_val, y_fatal_val, epochs=50, verbose=0)
        fatality_predict_next = model_fatalities.predict(np.reshape(X_fatal_val[len(X_fatal_val)-1],(1,n_steps,1)), verbose=0).astype(int)
        
        test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==78), 'ConfirmedCases'] = test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==77), 'ConfirmedCases'].values[0] + cases_predict_next[0]
        test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==78), 'Fatalities'] = test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==77), 'Fatalities'].values[0] + fatality_predict_next[0]
            
        print(cases_predict_next[0])
        print(X_cases_val[len(X_cases_val)-1])
        new_entry_cases=X_cases_val[len(X_cases_val)-1]
        new_entry_fatal=X_fatal_val[len(X_fatal_val)-1]
        
        for i in range(30):
            
          # after 7 days rates will mostly decline in these countries
          last_series = np.reshape(new_entry_cases,(1,n_steps,1))
          new_entry_cases=np.delete(last_series,[0])
          new_entry_cases = np.insert(new_entry_cases,n_steps-1,cases_predict_next[0])
          #X_cases_val = np.append(X_cases_val, np.reshape(new_entry_cases,(1,n_steps,1)), axis=0)
    
          #y_cases_val = np.append(y_cases_val,cases_predict_next[0],axis=0)
          #model_cases.fit(X_cases_val, y_cases_val, epochs=50, verbose=0)
        
          cases_predict_next = model_cases.predict(np.reshape(new_entry_cases,(1,n_steps,1)), verbose=0).astype(int)
          if(cases_predict_next[0]-new_entry_cases[n_steps-1]>cases_increase_avg):
            cases_predict_next =  np.array([max(0,new_entry_cases[n_steps-1]+cases_increase_avg)])
          print(np.array(new_entry_cases))
          print(cases_predict_next[0])

          last_series = np.reshape(new_entry_fatal,(1,n_steps,1))
          new_entry_fatal=np.delete(last_series,[0])
          new_entry_fatal = np.insert(new_entry_fatal,n_steps-1,fatality_predict_next[0])
        
          fatality_predict_next = model_fatalities.predict(np.reshape(new_entry_fatal,(1,n_steps,1)), verbose=0).astype(int)
          if(fatality_predict_next[0]-new_entry_fatal[n_steps-1]>fatal_increase_avg):
            fatality_predict_next[0] =  max(0,new_entry_fatal[n_steps-1]+fatal_increase_avg)  
        
          test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==i+79), 'ConfirmedCases'] = test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==78), 'ConfirmedCases'].values[0] + cases_predict_next[0]
          test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==i+79), 'Fatalities'] = test_df.loc[(test_df['Country']==country) & (test_df['Province_State']==province) & (test_df['Days']==i+78), 'Fatalities'].values[0] + fatality_predict_next[0]

        del cases_predict_next
        del fatality_predict_next
        del model_cases
        del model_fatalities
for country in no_province_countries:
    if country not in ["Italy","Germany","Spain","Turkey","France","Switzerland"]:
      continue
    current_considered_country_df = no_province_country_dfs[country][['ConfirmedCases','Fatalities','Days']].reset_index()
    print(country+" " + str(len(current_considered_country_df)))

#     days_df = current_considered_country_df['Date'].apply(lambda dt: datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.datetime.strptime('2020-01-21', '%Y-%m-%d')).apply(lambda x : str(x).split()[0]).astype(int)
#     current_considered_country_df['Days'] = days_df
    
    for i in range(13):
            test_df.loc[(test_df['Country']==country) & (test_df['Days']==i+65), 'ConfirmedCases'] = current_considered_country_df.loc[current_considered_country_df['Days'] == i+65, 'ConfirmedCases'].values[0]
            test_df.loc[(test_df['Country']==country) & (test_df['Days']==i+65), 'Fatalities'] = current_considered_country_df.loc[current_considered_country_df['Days'] == i+65, 'Fatalities'].values[0]

    indexNames = current_considered_country_df[ current_considered_country_df['ConfirmedCases'] == 0 ].index
    current_considered_country_df.drop(indexNames , inplace=True)
    
    #x_train_full= current_considered_country_df[['Days','Population','Density','Below14','15-64','Over65']].to_numpy()
    cases_train = np.diff(current_considered_country_df['ConfirmedCases'].to_numpy())
    fatalities_train = np.diff(current_considered_country_df['Fatalities'].to_numpy())
    
    #get last 7/15 days increasing average
    cases_increase_avg = 0
    days=0
    for i in range(max(0,len(cases_train)-7),len(cases_train)-1):
        cases_increase_avg +=  (cases_train[i+1] -cases_train[i])
        days+=1
    if(days>0):
        cases_increase_avg = int(cases_increase_avg/days)
    fatal_increase_avg = 0
    days=0
    for i in range(max(0,len(fatalities_train)-7),len(fatalities_train)-1):
        fatal_increase_avg +=  (fatalities_train[i+1] -fatalities_train[i])
        days+=1
    if(days>0):
        fatal_increase_avg = int(fatal_increase_avg/days)
    
#     x_train_val= x_train_full[:50,:]
#     cases_train_val = cases_train[:50]
#     fatalities_train_val = fatalities_train[:50]

#     x_test_val= x_train_full[50:,:]
#     cases_test_val = cases_train[50:]
#     fatalities_test_val = fatalities_train[50:]
    
    n_steps = max(int(len(cases_train)*0.1),3)
    print(len(cases_train))
    print("cases avg: "+str(cases_increase_avg)+" "+ str(days))
    
    X_cases_val, y_cases_val = split_sequence(cases_train,n_steps)
    X_fatal_val, y_fatal_val = split_sequence(fatalities_train,n_steps)

    X_cases_val = np.reshape(X_cases_val, (X_cases_val.shape[0],X_cases_val.shape[1],1))
    X_fatal_val = np.reshape(X_fatal_val, (X_fatal_val.shape[0],X_fatal_val.shape[1],1))

    print(X_cases_val.shape)
#         X_cases_test, y_cases_test = split_sequence(cases_test_v, n_steps)
#     assert(len(X_fatal_val)==len(X_cases_val))
#     assert(len(y_fatal_val)==len(y_cases_val))

    model_cases=build_model(n_steps)
    model_cases.fit(X_cases_val, y_cases_val, epochs=50, verbose=0)
    cases_predict_next = model_cases.predict(np.reshape(X_cases_val[len(X_cases_val)-1],(1,n_steps,1)), verbose=0).astype(int)

    model_fatalities=build_model(n_steps)
    model_fatalities.fit(X_fatal_val, y_fatal_val, epochs=50, verbose=0)
    fatality_predict_next = model_fatalities.predict(np.reshape(X_fatal_val[len(X_fatal_val)-1],(1,n_steps,1)), verbose=0).astype(int)

    test_df.loc[(test_df['Country']==country) & (test_df['Days']==78), 'ConfirmedCases'] = test_df.loc[(test_df['Country']==country) & (test_df['Days']==77), 'ConfirmedCases'].values[0] + cases_predict_next[0]
    test_df.loc[(test_df['Country']==country) & (test_df['Days']==78), 'Fatalities'] = test_df.loc[(test_df['Country']==country) & (test_df['Days']==77), 'Fatalities'].values[0] + fatality_predict_next[0]

    print(cases_predict_next[0])
    print(X_cases_val[len(X_cases_val)-1])
    
    new_entry_cases=X_cases_val[len(X_cases_val)-1]
    new_entry_fatal=X_fatal_val[len(X_fatal_val)-1]
        
    for i in range(30):

      # after 7 days rates will mostly decline in these countries
      last_series = np.reshape(new_entry_cases,(1,n_steps,1))
      new_entry_cases=np.delete(last_series,[0])
      new_entry_cases = np.insert(new_entry_cases,n_steps-1,cases_predict_next[0])
      #X_cases_val = np.append(X_cases_val, np.reshape(new_entry_cases,(1,n_steps,1)), axis=0)

      #y_cases_val = np.append(y_cases_val,cases_predict_next[0],axis=0)
      #model_cases.fit(X_cases_val, y_cases_val, epochs=50, verbose=0)

      cases_predict_next = model_cases.predict(np.reshape(new_entry_cases,(1,n_steps,1)), verbose=0).astype(int)
      if(cases_predict_next[0]-new_entry_cases[n_steps-1]>cases_increase_avg):
        cases_predict_next =  np.array([max(0,new_entry_cases[n_steps-1]+cases_increase_avg)])
      print(np.array(new_entry_cases))
      print(cases_predict_next[0])

      last_series = np.reshape(new_entry_fatal,(1,n_steps,1))
      new_entry_fatal=np.delete(last_series,[0])
      new_entry_fatal = np.insert(new_entry_fatal,n_steps-1,fatality_predict_next[0])
      #X_fatal_val = np.append(X_fatal_val, np.reshape(new_entry_fatal,(1,n_steps,1)), axis=0)

      fatality_predict_next = model_fatalities.predict(np.reshape(new_entry_fatal,(1,n_steps,1)), verbose=0).astype(int)
      if(fatality_predict_next[0]-new_entry_fatal[n_steps-1]>fatal_increase_avg):
        fatality_predict_next[0] =  max(0,new_entry_fatal[n_steps-1]+fatal_increase_avg)  

      test_df.loc[(test_df['Country']==country)  & (test_df['Days']==i+79), 'ConfirmedCases'] = test_df.loc[(test_df['Country']==country)  & (test_df['Days']==78), 'ConfirmedCases'].values[0] + cases_predict_next[0]
      test_df.loc[(test_df['Country']==country) & (test_df['Days']==i+79), 'Fatalities'] = test_df.loc[(test_df['Country']==country)  & (test_df['Days']==i+78), 'Fatalities'].values[0] + fatality_predict_next[0]

test_df['ConfirmedCases'].fillna(100, inplace = True)
test_df['Fatalities'].fillna(10, inplace = True) 
submit = pd.DataFrame()
submit['ForecastId'] = test_df['ForecastId']
submit['ConfirmedCases'] = test_df['ConfirmedCases']
submit['Fatalities'] = test_df['Fatalities']
display(submit)
submit.to_csv('submission.csv')
