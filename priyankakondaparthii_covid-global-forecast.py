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
#importing the necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing

import time

import datetime

from datetime import datetime



import plotly.express as px

import plotly.graph_objects as go



import warnings

warnings.filterwarnings('ignore')

#reading the data set

#test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

#train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

df_pop = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')

covid_pop = pd.read_csv('../input/covid19-global-forecasting-locations-population/locations_population.csv')
display(train.describe())

display(train.tail(5))

display(train.shape)

train[['Province_State','Country_Region','Date']].describe()
print("Max Date: ", max(train['Date']), ", Min Date: ", min(train['Date']))

print("Max Id: ", max(train['Id']), ", Unique Ids: ", train['Id'].nunique())

print("Max Date: ", max(test['Date']), ", Min Date: ", min(test['Date']))
train.isnull().sum()
#plotting the daily trend

ww_df = train.groupby('Date')[['ConfirmedCases', 'Fatalities']].sum().reset_index()

# shift is used to take the previous value

ww_df['New_Cases'] = ww_df['ConfirmedCases'] - ww_df['ConfirmedCases'].shift(1)

#ww_df.tail()

#pd.melt is used to create a simillar table like a pivot in excel



trend_df = pd.melt(ww_df, id_vars=['Date'], value_vars=['ConfirmedCases', 'Fatalities', 'New_Cases'])

trend_df.head(5)



fig = px.line(trend_df, x="Date", y="value", color='variable', title="Worldwide Confirmed/Death trend")

fig.show()
plt.style.use(['tableau-colorblind10'])



#plotting the top 8 Countries with most Fatalities as per latest day



df_Country = train.groupby(['Country_Region'])[["Fatalities","ConfirmedCases"]].max().nlargest(8,'Fatalities')

#df_Country = train[train['Date']=='2020-03-31'].groupby(['Country_Region'])[["Fatalities","ConfirmedCases"]].sum().nlargest(8,'Fatalities')

#bycountry = train.groupby('Country_Region')['Fatalities'].max().sort_values(ascending=False).to_frame().reset_index()



#fatality % = Fatality_Count / Confirmed_Count



df_Country['Fatality_Percentage'] = df_Country['Fatalities']/ df_Country['ConfirmedCases']

df_Country = df_Country.reset_index()

df_Country.sort_values('Fatality_Percentage',inplace=True)

figure, axes = plt.subplots(1, 2,figsize=(12,4))

df_Country.plot(ax= axes[0],x = 'Country_Region', y = ["Fatalities","ConfirmedCases"],kind='bar', title = 'Fatalities Vs Confirmed')

df_Country.plot(ax= axes[1],x = 'Country_Region', y = ["Fatality_Percentage"],kind='bar', title = 'Fatalities divided by Confirmed')
#getting the dates with at least 1 Fatality and atleast 1 confirmed case separately

cond1 = train.Fatalities >=1

cond2 = train.ConfirmedCases >= 1

train_Fatal = train[['Country_Region','Date']][(cond1)]

train_Confirm = train[['Country_Region','Date']][(cond2)]



#getting the 1st Confirmed case date for each country

Confirm_Min_Max = train_Confirm.groupby("Country_Region", as_index=False)["Date"].agg(["min","max"])



#getting the 1st Fatality date for each country

Fatal_Min_Max = train_Fatal.groupby("Country_Region", as_index=False)["Date"].agg(["min","max"])



#left outer join the above 2 dataFrames



Country_Dates = pd.merge(Confirm_Min_Max, Fatal_Min_Max, how='left', on=['Country_Region'])

Country_Dates = Country_Dates.rename(columns={'min_x': 'fst_Confirmed', 'max_x': 'last_Confirmed', 'min_y': 'fst_Fatal','max_y': 'last_Fatal'})

    

#setting a default date for null date values

#should remove null day count to get fatality rate (to avoid Division with nulls)



#Date_Cols = ['fst_Confirmed','last_Confirmed','fst_Fatal','last_Fatal']



#Country_Dates[Date_Cols] = Country_Dates[Date_Cols].fillna(pd.to_datetime('2015-01-01'))



Country_Dates['Fatality_Days'] = pd.to_datetime(Country_Dates['last_Fatal']) - pd.to_datetime(Country_Dates['fst_Fatal'])

Country_Dates = Country_Dates.reset_index()

#Country_Dates.head(5)
# converting date counts to integer values



Country_Dates['Fatality_Days'] = Country_Dates['Fatality_Days'] / np.timedelta64(1, 'D')



Country_Dates.head(5)
#consider the country wise fatalities and Confirmed cases



#Country_Counts = train.groupby(['Country_Region'])[["Fatalities","ConfirmedCases"]].sum()

cond1 = train.Date == '2020-04-12'

Country_Counts = train[['Country_Region','Date','Fatalities','ConfirmedCases']][(cond1)]

#Country_Counts = train[train['Date']=='2020-03-31']

#Getting min max dates and counts grouped by country



df_Country2 = pd.merge(Country_Dates, Country_Counts, how='left', on=['Country_Region'])



df_Country2['Fatality_Rate'] = round(df_Country2['Fatalities']/pd.to_numeric(df_Country2['Fatality_Days']),2)

df_Country2['Fatality_Rate'] = round(df_Country2['Fatalities']/pd.to_numeric(df_Country2['Fatality_Days']),2)



#null fatality rates are set as 0

#infinity fatality rates are set as NaN



df_Country2['Fatality_Rate'] = df_Country2['Fatality_Rate'].fillna(0)

df_Country2 = df_Country2.replace([np.inf, -np.inf], np.nan)

#df_Country2['Country_Region'].describe()



#top 8 countries with the largest Fatality Rates



df_Country3 = df_Country2[df_Country2['Fatality_Rate'] > 0].groupby(['Country_Region'])[["Fatality_Rate"]].sum().nlargest(8,'Fatality_Rate')

df_Country3 = df_Country3.reset_index()
#df_Country3.head(5)

#df_Country3.plot(x='Country_Region', y= ["Fatality_Rate"], kind = 'bar', title = 'Top 8 countries with largest fatality rate (Fatalities/Day)')



fig = px.bar(df_Country3, x='Country_Region', y='Fatality_Rate',color='Fatality_Rate')

fig.show()
confirmed_China = train[train['Country_Region']=='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_China = train[train['Country_Region']=='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_China = confirmed_China.join(fatalities_China)



confirmed_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'Fatalities':['sum']})

total_Italy = confirmed_Italy.join(fatalities_Italy)



confirmed_Spain = train[train['Country_Region']=='Spain'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_Spain = train[train['Country_Region']=='Spain'].groupby(['Date']).agg({'Fatalities':['sum']})

total_Spain = confirmed_Spain.join(fatalities_Spain)



confirmed_Iran = train[train['Country_Region']=='Iran'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_Iran = train[train['Country_Region']=='Iran'].groupby(['Date']).agg({'Fatalities':['sum']})

total_Iran = confirmed_Iran.join(fatalities_Iran)



confirmed_France = train[train['Country_Region']=='France'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_France = train[train['Country_Region']=='France'].groupby(['Date']).agg({'Fatalities':['sum']})

total_France = confirmed_France.join(fatalities_France)



confirmed_Netherlands = train[train['Country_Region']=='Netherlands'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_Netherlands = train[train['Country_Region']=='Netherlands'].groupby(['Date']).agg({'Fatalities':['sum']})

total_Netherlands = confirmed_Netherlands.join(fatalities_Netherlands)



confirmed_UK = train[train['Country_Region']=='United Kingdom'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_UK = train[train['Country_Region']=='United Kingdom'].groupby(['Date']).agg({'Fatalities':['sum']})

total_UK = confirmed_UK.join(fatalities_UK)



confirmed_USA = train[train['Country_Region']=='US'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_USA = train[train['Country_Region']=='US'].groupby(['Date']).agg({'Fatalities':['sum']})

total_USA = confirmed_USA.join(fatalities_USA)

plt.figure(figsize=(24,18))



plt.subplot(3, 3, 1)

total_China.plot(ax=plt.gca(), title='China')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(3, 3, 2)

total_Italy.plot(ax=plt.gca(), title='Italy')



plt.subplot(3, 3, 3)

total_Spain.plot(ax=plt.gca(), title='Spain')



plt.subplot(3, 3, 4)

total_Iran.plot(ax=plt.gca(), title='Iran')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(3, 3, 5)

total_France.plot(ax=plt.gca(), title='France')



plt.subplot(3, 3, 6)

total_Netherlands.plot(ax=plt.gca(), title='Netherlands')



plt.subplot(3, 3, 7)

total_UK.plot(ax=plt.gca(), title='United Kingdom')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(3, 3, 8)

total_USA.plot(ax=plt.gca(), title='USA')
#selecting only the needed columns from population 

#df_pop.columns

df_pop_sel = df_pop[['Country (or dependency)','Population (2020)','Density (P/Km²)','Land Area (Km²)','Med. Age','Urban Pop %']]



#renaming the columns

df_pop_sel.columns = ['Country_Region', 'Population (2020)', 'Density (P/Km²)','Land Area (Km²)', 'Med. Age', 'Urban Pop %']



# Remove the % sign from Urban Pop % field

df_pop_sel['Urban Pop %'] = df_pop_sel['Urban Pop %'].str.rstrip('%')



# Replace United States by US

df_pop_sel.loc[df_pop_sel['Country_Region']=='United States', 'Country_Region'] = 'US'



df_country_pop = pd.merge(df_pop_sel, df_Country, how='inner', on=['Country_Region'])

df_country_pop
df_country_pop["pop_factor_fatal"] = df_country_pop['Fatalities']/df_country_pop['Population (2020)']

df_country_pop["pop_factor_confirmed"] = df_country_pop['ConfirmedCases']/df_country_pop['Population (2020)']

#df_country_pop = df_country_pop.reset_index(drop = True)

df_country_pop.sort_values('pop_factor_fatal',inplace=True)

df_country_pop

df_country_pop.plot(x = 'Country_Region', y = ["pop_factor_fatal","pop_factor_confirmed"],kind='barh', title = 'Fatalities, Confirmed considering population',figsize = (8,4))
# filtering data before 26th March as for the competition rules for prediction



cond1 = train['Date'] < '2020-04-01'

train_fil = train[(cond1)]

display(train_fil['Date'].max())

display(train_fil['Date'].min())
display(train_fil.head(5))

display(covid_pop.head(5))
#preprocessing



train_fil['Date'] = pd.to_datetime(train_fil['Date'])

train_fil['Day_num'] = preprocessing.LabelEncoder().fit_transform(train_fil.Date)

train_fil['Day'] = train_fil['Date'].dt.day

train_fil['Month'] = train_fil['Date'].dt.month

train_fil['Year'] = train_fil['Date'].dt.year



train_fil_copy = train_fil



train_fil['Province_State'].fillna("None", inplace=True)



#Checking null values

train_fil[train_fil.iloc[:,0:].isnull().any(axis = 1)].iloc[:,0:].head()
# there is a new data set uploaded for population details for covid data set. I am using this for the population data



covid_pop_sel = covid_pop[['Province.State','Country.Region','Population']]

covid_pop_sel['Province.State'].fillna("None", inplace=True)



#Checking null values

covid_pop_sel[covid_pop_sel.iloc[:,0:].isnull().any(axis = 1)].iloc[:,0:].head()
#joining population details as previously done in EDA



df_train_pop = train_fil.merge(covid_pop_sel, left_on = ['Country_Region','Province_State'], right_on = ['Country.Region','Province.State'], how = 'left')

#df_train_pop[df_train_pop.iloc[:,1:].isnull().any(axis = 1)].iloc[:,1:].head(3)
display(covid_pop_sel[covid_pop_sel['Country.Region']=='Italy'])

display(train_fil[train_fil['Country_Region']=='Italy'])
#getting Country_Regions with Null population (These have not joined correctly when used the primary key as Country_Region)

#check = df_train_pop[df_train_pop['Country_Region']=='Canada'][df_train_pop['Population'].isnull()]

df_train_pop['Province_State'][df_train_pop['Population'].isnull()].unique()

df_train_pop.reindex()

#check[check['Date']=='2020-03-25']
#code to check column wise null count

#pd.DataFrame(population_raw.isnull().sum()).T



#wildcard matching 

#df_pop_sel[df_pop_sel['Country_Region'].str.match('Taiwan')]
# Replace Country names to match the train data set

#df_pop_sel.loc[df_pop_sel['Country_Region']=='Czech Republic (Czechia)', 'Country_Region'] = 'Czechia'

#df_pop_sel.loc[df_pop_sel['Country_Region']=='Taiwan', 'Country_Region'] = 'Taiwan*'
#merging the data set again, after mapping the country names

#df_train_pop = train_fil.merge(df_pop_sel, left_on = 'Country_Region', right_on = 'Country_Region', how = 'left')

#df_train_pop[df_train_pop.iloc[:,1:].isnull().any(axis = 1)].iloc[:,1:].head(3)
df_train_pop[df_train_pop.iloc[:,1:].isnull().any(axis = 1)].iloc[:,1:].head(3)
#treating null values

df_train_pop['Province.State'].fillna("None", inplace=True)

df_train_pop['Country.Region'].fillna("None", inplace=True)

df_train_pop['Population'] = df_train_pop['Population'].fillna(0)
#df_train_pop[['Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']] = df_train_pop[['Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']].fillna(0)
missings_count = {col:df_train_pop[col].isnull().sum() for col in df_train_pop.columns}

missings = pd.DataFrame.from_dict(missings_count, orient='index')

print(missings.nlargest(30, 0))
#Arranging the data set as needed for the SIR model



df_train_pop.head(5)
df_fin_selc = df_train_pop[['Province_State','Country_Region','Date','ConfirmedCases','Fatalities','Day_num','Day','Month','Year','Population']]
df_fin_selc[df_fin_selc['Date']=='2020-03-25'].head(5)
df_fin_selc['I'] = df_fin_selc['ConfirmedCases']-df_fin_selc['Fatalities']

df_fin_selc['R'] = df_fin_selc['Fatalities']

df_fin_selc['S'] = df_fin_selc['Population']-df_fin_selc['ConfirmedCases']

df_SIR = df_fin_selc[['Province_State','Country_Region','Population','Day_num','S','I','R']].reindex()

df_SIR[df_SIR['Country_Region'] == 'Italy'].tail(5)
df_SIR[df_SIR['Country_Region']=='United Kingdom'].plot(x='Day_num',y= ['I','R'],kind = 'line')
import numpy as np

from scipy.integrate import odeint

from scipy import integrate, optimize
def SIR_testmodel(y,t,bta,gmma):

    S, I, R = y

    

    dS_dt = -1*bta*I*S/N

    dI_dt = (bta*I*S/N) - gmma*I

    dR_dt = gmma*I

    

    return ([dS_dt, dI_dt, dR_dt])
#defining initial conditions



N = 1

S00 = 0.9

I00 = 0.1

R00 = 0.0

bta = 0.35

gmma = 0.1



t = np.linspace(0,100,1000)



sol = odeint(SIR_testmodel,[S00,I00,R00],t,args = (bta,gmma))

sol = np.array(sol)
#plotting results



plt.figure(figsize=(6,4))

plt.plot(t, sol[:,0],label = "S(t)")

plt.plot(t, sol[:,1],label = "I(t)")

plt.plot(t, sol[:,2],label = "R(t)")

plt.legend()

plt.show()
Italy_SIR_df = df_SIR[df_SIR['Country_Region']=='Italy']

#covid_pop_sel[covid_pop_sel['Country.Region']=='Italy']

Italy_SIR_df = Italy_SIR_df.reset_index(drop = True)

Italy_SIR_df.tail(5)
display(covid_pop_sel[covid_pop_sel['Country.Region']=='Italy'])
Country_Dates[Country_Dates['Country_Region'] == 'Italy']
cond1 = Italy_SIR_df.I >= 1

test = Italy_SIR_df['Day_num'][(cond1)]

test

Italy_SIR_df.iloc[:10]
Italy_S = Italy_SIR_df['S']

Italy_I = Italy_SIR_df['I']

Italy_R = Italy_SIR_df['R']



Italy_s = np.array(Italy_S, dtype=float)

Italy_i = np.array(Italy_I, dtype=float)

Italy_r = np.array(Italy_R, dtype=float)



display(Italy_s[9],Italy_i[9], Italy_r[9])
def SIR_model(y,t,beta,gamma):

    S, I, R = y

    N = 60480000

    dS_dt = -1*beta*I*S/N

    dI_dt = (beta*I*S/N) - gamma*I

    dR_dt = gamma*I

    

    return ([dS_dt, dI_dt, dR_dt])



def fit_odeint(x, beta, gamma):

    return integrate.odeint(SIR_model, (S0, I0, R0), x, args=(beta, gamma))[:,1]
xdata = Italy_SIR_df.Day_num

ydata = Italy_i

xdata = np.array(xdata, dtype=float)



S0 = 60479998.0

I0 = 2.0

R0 = 0

y = S0, I0, R0



popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)

fitted = fit_odeint(xdata, *popt)
plt.plot(xdata, ydata, 'o')

plt.plot(xdata, fitted)

plt.title("Fit of SIR model for Italy infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
#I am using the new data set with Recovered details included as I am getting gamma and beta values more than 1

#this is the same above data set with more details incorporated like Recovered Count and longutude latitude details



comp_df = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

comp_df.head(5)
comp_df['Province/State'].fillna("None", inplace=True)



#Checking null values

comp_df[comp_df.iloc[:,0:].isnull().any(axis = 1)].iloc[:,0:].head()
comp_df['Date'] = pd.to_datetime(comp_df['Date'])
#covid_pop_sel 

covid_pop_sel.loc[covid_pop_sel['Country.Region']=='Korea, South', 'Country.Region'] = 'South Korea'

comp_df_pop = comp_df.merge(covid_pop_sel, left_on = ['Country/Region','Province/State'], right_on = ['Country.Region','Province.State'], how = 'left')

comp_df_pop.tail(3)
comp_pop_sel = comp_df_pop[['Province/State','Country/Region','Date','Confirmed','Deaths','Recovered','Population']]
comp_pop_sel['Country/Region'][comp_pop_sel['Population'].isnull()].unique()
#comp_df[comp_df['Country/Region'].str.match('Canada')]

#covid_pop_sel[covid_pop_sel['Country.Region'].str.match('Canada')].head(2)
missings_count = {col:comp_pop_sel[col].isnull().sum() for col in comp_pop_sel.columns}

missings = pd.DataFrame.from_dict(missings_count, orient='index')

print(missings.nlargest(30, 0))
#treating null values

comp_pop_sel['Population'] = comp_pop_sel['Population'].fillna(0)
comp_df_Italy = comp_pop_sel[comp_pop_sel['Country/Region']=='Italy']

comp_df_Italy = comp_df_Italy.reset_index(drop = True)

comp_df_Italy.tail(5)
comp_df_Italy['Day_num'] = preprocessing.LabelEncoder().fit_transform(comp_df_Italy.Date)

comp_df_Italy.tail(3)
comp_df_Italy['R'] = comp_df_Italy['Deaths']+comp_df_Italy['Recovered']

comp_df_Italy['I'] = comp_df_Italy['Confirmed']- comp_df_Italy['R']

comp_df_Italy['S'] = comp_df_Italy['Population']-comp_df_Italy['I']-comp_df_Italy['R']

comp_df_Italy.tail(3)
comp_df_Italy_S = comp_df_Italy['S']

comp_df_Italy_I = comp_df_Italy['I']

comp_df_Italy_R = comp_df_Italy['R']



comp_df_Italy_s = np.array(comp_df_Italy_S, dtype=float)

comp_df_Italy_i = np.array(comp_df_Italy_I, dtype=float)

comp_df_Italy_r = np.array(comp_df_Italy_R, dtype=float)



display(comp_df_Italy_s[9],comp_df_Italy_i[9], comp_df_Italy_r[9])
N = 60480000

S0 = 60479998

I0 = 2

R0 = 0



xdata = comp_df_Italy.Day_num

xdata = np.array(xdata, dtype=float)

ydata = comp_df_Italy_i

ydata = np.array(ydata, dtype=float)



popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)

fitted = fit_odeint(xdata, *popt)



plt.plot(xdata, ydata, 'o')

plt.plot(xdata, fitted)

plt.title("Fit of SIR model for Italy infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

df_subm = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
df_train_copy = df_train.copy()

df_test_copy = df_test.copy()
df_train.rename(columns={'Country_Region':'Country'}, inplace=True)

df_train.rename(columns={'Province_State':'State'}, inplace=True)

df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)



df_test.rename(columns={'Country_Region':'Country'}, inplace=True)

df_test.rename(columns={'Province_State':'State'}, inplace=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)
#confirmed cases

y1_df_Train = df_train.iloc[:, -2]

#fatalities

y2_df_Train = df_train.iloc[:, -1]



NOVAL = "NOVAL"

def handlenullstate(State, Country):

    if State == NOVAL: return Country

    return State
#Data Cleansing and enrichment



df_train['State'].fillna(NOVAL, inplace=True)

df_train['State'] = df_train.loc[:, ['State', 'Country']].apply(lambda x : handlenullstate(x['State'], x['Country']), axis=1)



df_train.loc[:, 'Date'] = df_train.Date.dt.strftime("%m%d")

df_train["Date"]  = df_train["Date"].astype(int)



df_test['State'].fillna(NOVAL, inplace=True)

df_test['State'] = df_test.loc[:, ['State', 'Country']].apply(lambda x : handlenullstate(x['State'], x['Country']), axis=1)



df_test.loc[:, 'Date'] = df_test.Date.dt.strftime("%m%d")

df_test["Date"]  = df_test["Date"].astype(int)



df_test.head()
covid_pop_taken = covid_pop[['Province.State','Country.Region','Population']]
covid_pop_taken.rename(columns={'Province.State':'State'}, inplace=True)

covid_pop_taken.rename(columns={'Country.Region':'Country'}, inplace=True)
covid_pop_taken['State'].fillna("None", inplace=True)

NOVAL = 'None'

covid_pop_taken['State'] = covid_pop_taken.loc[:, ['State', 'Country']].apply(lambda x : handlenullstate(x['State'], x['Country']), axis=1)

covid_pop_taken.head(3)
df_train_copy2 = df_train.copy()

df_test_copy2 = df_test.copy()

df_test.head(3)
df_train_co_pop = df_train.merge(covid_pop_taken, left_on = ['Country','State'], right_on = ['Country','State'], how = 'left')

df_test_co_pop = df_test.merge(covid_pop_taken, left_on = ['Country','State'], right_on = ['Country','State'], how = 'left')

df_test_co_pop.head(5)
df_train_co_pop['Country'][df_train_co_pop['Population'].isnull()].unique()
#covid_pop_sel.loc[covid_pop_sel['Country']=='Korea, South', 'Country.Region'] = 'South Korea'
df_train_co_pop[['Population']] = preprocessing.scale(df_train_co_pop[['Population']])

df_test_co_pop[['Population']] = preprocessing.scale(df_test_co_pop[['Population']])
df_train_co_pop.head(3)
df_train = df_train_co_pop 

df_test = df_test_co_pop 
from sklearn import preprocessing



le = preprocessing.LabelEncoder()



df_train.Country = le.fit_transform(df_train.Country)

df_train['State'] = le.fit_transform(df_train['State'])

df_train.Country = le.fit_transform(df_train.Country)



df_test.Country = le.fit_transform(df_test.Country)

df_test['State'] = le.fit_transform(df_test['State'])



df_test.head()
from warnings import filterwarnings

filterwarnings('ignore')



from sklearn import preprocessing



le = preprocessing.LabelEncoder()



from xgboost import XGBRegressor



countries = df_train.Country.unique()
xout = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = df_train.loc[df_train.Country == country, :].State.unique()

    for state in states:

        #trian

        x_train_CS = df_train.loc[(df_train.Country == country) & (df_train.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities','Population']]

        y1_train_CS = x_train_CS.loc[:, 'ConfirmedCases']

        y2_train_CS = x_train_CS.loc[:, 'Fatalities']

        x_train_CS = x_train_CS.loc[:, ['State', 'Country', 'Date','Population']]

        x_train_CS.Country = le.fit_transform(x_train_CS.Country)

        x_train_CS['State'] = le.fit_transform(x_train_CS['State'])

        

        #test

        x_test_CS = df_test.loc[(df_test.Country == country) & (df_test.State == state), ['State', 'Country', 'Date', 'ForecastId','Population']]

        x_test_CS_Id = x_test_CS.loc[:, 'ForecastId']

        x_test_CS = x_test_CS.loc[:, ['State', 'Country', 'Date','Population']]

        x_test_CS.Country = le.fit_transform(x_test_CS.Country)

        x_test_CS['State'] = le.fit_transform(x_test_CS['State'])

        

        xmodel1 = XGBRegressor(n_estimators=1000)

        xmodel1.fit(x_train_CS, y1_train_CS)

        y1_xpred = xmodel1.predict(x_test_CS)

        

        xmodel2 = XGBRegressor(n_estimators=1000)

        xmodel2.fit(x_train_CS, y2_train_CS)

        y2_xpred = xmodel2.predict(x_test_CS)

        

        xdata = pd.DataFrame({'ForecastId': x_test_CS_Id, 'ConfirmedCases': y1_xpred, 'Fatalities': y2_xpred})

        xout = pd.concat([xout, xdata], axis=0)

xout.ForecastId = xout.ForecastId.astype('int')

xout['ConfirmedCases'] = round(xout['ConfirmedCases'],1)

xout['Fatalities'] = round(xout['Fatalities'],1)

display(xout.head())
#xout['ForecastId'] = xout['ForecastId'].apply(int)

xout['ConfirmedCases'] = xout['ConfirmedCases'].apply(int)

xout['Fatalities'] = xout['Fatalities'].apply(int)
xout.dtypes
xout = xout.drop_duplicates()

xout.reindex()
xout.to_csv('submission.csv', index=False)