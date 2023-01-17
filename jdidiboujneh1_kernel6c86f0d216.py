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
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")#index_col=0
display(train_data.head())
test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")#index_col=0
display(test_data.head())
sum_df = pd.pivot_table(train_data, values=['ConfirmedCases','Fatalities'], index=['Date'],aggfunc=np.sum)
display(sum_df.max())

#UPDATES TO ADD THE NEW COLUMNS
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

def getColumnInfo(df):
    n_province =  df['Province_State'].nunique()
    n_country  =  df['Country_Region'].nunique()
    n_days     =  df['Date'].nunique()
    start_date =  df['Date'].unique()[0]
    end_date   =  df['Date'].unique()[-1]
    return n_province, n_country, n_days, start_date, end_date

n_train = train_data.shape[0]
n_test = test_data.shape[0]

n_prov_train, n_count_train, n_train_days, start_date_train, end_date_train = getColumnInfo(train_data)
n_prov_test,  n_count_test,  n_test_days,  start_date_test,  end_date_test  = getColumnInfo(test_data)

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

print('Percentage of confirmed case  = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_confirm_train, n_train, prob_confirm_check_train[1:].sum()*100))
print('Percentage of fatality  = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_fatal_train, n_train, prob_fatal_check_train[1:].sum()*100))

train_data_by_country = train_data.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum',
                                                                                         'GrowthRate':'mean' })
max_train_date = train_data['Date'].max()
train_data_by_country_confirm = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)').sort_values('ConfirmedCases', ascending=False)
train_data_by_country_confirm.set_index('Country_Region', inplace=True)
display(train_data_by_country_confirm.head())

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle, islice
discrete_col = list(islice(cycle(['BLUE', 'r', 'g', 'k', 'b', 'c', 'm']), None, len(train_data_by_country_confirm.head(30))))
plt.rcParams.update({'font.size': 24})
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


train_data_by_country_max = train_data.groupby(['Country_Region'],as_index=True).agg({'ConfirmedCases': 'max', 'Fatalities': 'max'})
train_data_by_country_fatal = train_data_by_country_max[train_data_by_country_max['Fatalities']>500]
train_data_by_country_fatal = train_data_by_country_fatal.sort_values(by=['Fatalities'],ascending=False).reset_index()
display(train_data_by_country_fatal.head(20))

df_merge_by_country = pd.merge(train_data,train_data_by_country_fatal['Country_Region'],on=['Country_Region'],how='inner')
df_max_fatality_country = df_merge_by_country.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum',
                                                                                                     'Fatalities': 'sum',
                                                                                                     'NewConfirmedCases':'sum',
                                                                                                     'NewFatalities':'sum',
                                                                                                     'MortalityRate':'mean'})

fig = plt.figure()
fig,(ax4,ax5) = plt.subplots(1,2,figsize=(20, 8))

train_data_by_date.MortalityRate.plot(ax=ax4, x_compat=True, legend='Mortality Rate',color='r')
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
display(train_data_by_mortality.head())

palette = plt.get_cmap('gist_rainbow')

rainbow_col = [palette(1.*i/20.0) for i in range(20)]

train_data_by_mortality.MortalityRate.head(20).plot(figsize=(15,10), kind='barh', color=rainbow_col)
plt.xlabel("Mortality Rate")
plt.title("First 20 Countries with Highest Mortality Rate")
ylocs, ylabs = plt.yticks()

world_df = train_data_by_country.query('Date == @max_train_date')
world_df.loc[:,'Date']           = world_df.loc[:,'Date'].apply(str)
world_df.loc[:,'Confirmed_log']  = round(np.log10(world_df.loc[:,'ConfirmedCases'] + 1), 3)
world_df.loc[:,'Fatalities_log'] = np.log10(world_df.loc[:,'Fatalities'] + 1)
world_df.loc[:,'MortalityRate']  = round(world_df.loc[:, 'Fatalities'] / world_df.loc[:,'ConfirmedCases'], 3)
world_df.loc[:,'AveGrowthFactor']  = round(world_df.loc[:,'GrowthRate'], 3)
world_df.drop(['GrowthRate'], axis=1) #drop as this is actually the average over all the 74 days
display(world_df.head())

#Models

train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

# Replacing all the Province_State that are null by the Country_Region values
train_df.Province_State.fillna(train_df.Country_Region, inplace=True)
test_df.Province_State.fillna(test_df.Country_Region, inplace=True)

# Handling the Date column
# 1. Converting the object type column into datetime type
train_df.Date = train_df.Date.apply(pd.to_datetime)
test_df.Date = test_df.Date.apply(pd.to_datetime)

#Extracting Date and Month from the datetime and converting the feature as int
train_df.Date = train_df.Date.dt.strftime("%m%d").astype(int)
test_df.Date = test_df.Date.dt.strftime("%m%d").astype(int)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_df.Country_Region = le.fit_transform(train_df.Country_Region)
train_df['Province_State'] = le.fit_transform(train_df['Province_State'])

test_df.Country_Region = le.fit_transform(test_df.Country_Region)
test_df['Province_State'] = le.fit_transform(test_df['Province_State'])


X_train = train_df.drop(["Id", "ConfirmedCases", "Fatalities"], axis = 1) 
 
Y_train_CC = train_df["ConfirmedCases"] 
Y_train_Fat = train_df["Fatalities"] 

X_test = test_df.drop(["ForecastId"], axis = 1) 
from sklearn.model_selection import ShuffleSplit, cross_val_score
skfold = ShuffleSplit(random_state=7)


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

clf_bgr_CC = BaggingRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 40)
clf_bgr_Fat = BaggingRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 20)

bgr_acc = cross_val_score(clf_bgr_CC, X_train, Y_train_CC, cv = skfold)
bgr_acc_fat = cross_val_score(clf_bgr_Fat, X_train, Y_train_Fat, cv = skfold)
print (bgr_acc.mean(), bgr_acc_fat.mean())

clf_bgr_CC.fit(X_train, Y_train_CC)
Y_pred_CC = clf_bgr_CC.predict(X_test) 

clf_bgr_Fat.fit(X_train, Y_train_Fat)
Y_pred_Fat = clf_bgr_Fat.predict(X_test) 

print (Y_pred_Fat)
df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
soln = pd.DataFrame({'ForecastId': test_df.ForecastId, 'ConfirmedCases': Y_pred_CC, 'Fatalities': Y_pred_Fat})
df_out = pd.concat([df_out, soln], axis=0)
df_out.ForecastId = df_out.ForecastId.astype('int')
df_out.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
