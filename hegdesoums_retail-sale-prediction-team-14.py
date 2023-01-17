dd=pd.read_csv("../input/Train_Kaggle.csv")

dd
dd=pd.read_csv("../input/Train_Kaggle.csv")

#dd

dd=dd.loc[dd['ProductCategory'] == 'OtherClothing']

#dd

dd1=dd.groupby('Month', as_index=False)['Sales(In ThousandDollars)'].mean()

other=dd1
women

#men

#other
import numpy as np # linear algebra

import pandas as pd

import os

print(os.listdir("../input"))

import calendar

import datetime

from time import strptime

data_e = pd.read_excel('../input/MacroEconomicData.xlsx', 'Sheet1', index_col=None)

data_w = pd.read_excel('../input/WeatherData.xlsx', index_col=None)

#data_xls = pd.read_excel('../input/AttributesDescription.xlsx', index_col=None)

data_h = pd.read_excel('../input/Events_HolidaysData.xlsx')

#data_xls

data_e.to_csv('Economic.csv', encoding='utf-8')

data_w.to_csv('Weather.csv', encoding='utf-8')

data_h.to_csv('Holiday.csv', encoding='utf-8')

de=pd.read_csv("Economic.csv")

dw=pd.read_csv("Weather.csv")

dh=pd.read_csv("Holiday.csv")

de[['Year','Month']]=de['Year-Month'].str.split("-",expand=True,)

de=de.drop('Year-Month',axis=1)

dh[['year','Month','Day']]=dh['MonthDate'].str.split("-",expand=True)

dh=dh.drop('year',axis=1)

dh=dh.drop('MonthDate',axis=1)

dh['Month'] = pd.to_datetime(dh['Month'], format='%m').dt.month_name().str.slice(stop=3)

data_w1 = pd.read_excel('../input/WeatherData.xlsx', sheet_name='2009', ignore_index=True)

data_w2 = pd.read_excel('../input/WeatherData.xlsx', sheet_name='2010', ignore_index=True)

data_w3 = pd.read_excel('../input/WeatherData.xlsx', sheet_name='2011', ignore_index=True)

data_w4 = pd.read_excel('../input/WeatherData.xlsx', sheet_name='2012', ignore_index=True)

data_w5 = pd.read_excel('../input/WeatherData.xlsx', sheet_name='2013', index_col=None)

data_w6 = pd.read_excel('../input/WeatherData.xlsx', sheet_name='2014', index_col=None)

data_w7 = pd.read_excel('../input/WeatherData.xlsx', sheet_name='2015', index_col=None)

data_w8 = pd.read_excel('../input/WeatherData.xlsx', sheet_name='2016', index_col=None)

data_w1['Year'] = data_w1['Year'].astype('str') 

data_w2.Year='2010'

data_w3.Year='2011'

data_w4.Year='2012'

data_w5.Year='2013'

data_w6.Year='2014'

data_w7.Year='2015'

data_w8.Year='2016'



cdf=data_w1

cdf = cdf.append(data_w2)

cdf = cdf.append(data_w3)

cdf = cdf.append(data_w4)

cdf = cdf.append(data_w5)

cdf = cdf.append(data_w6)

cdf = cdf.append(data_w7)

cdf = cdf.append(data_w8)

dw2=cdf.groupby(['Year','Month'], as_index=False).agg(lambda x : x.mean() if x.dtype=='int64' else x.head(1))

dw2['Year'] = dw2['Year'].str.strip()

de['Year'] = de['Year'].str.strip()

dw2['Month'] = dw2['Month'].str.strip()

de['Month'] = de['Month'].str.strip()

data1=pd.merge(de,dw2, on=['Year','Month'])

data1['Month'] = data1['Month'].apply(lambda x:  strptime(x, '%b').tm_mon)

#data1.Month = data1.Month.map(d)

#data1

df1 = pd.read_csv('../input/Train_Kaggle.csv')

df2 = pd.read_csv('../input/Test_Kaggle.csv')

#df1

df1=df1.loc[df1['ProductCategory'] == 'MenClothing']

df2=df2.loc[df2['ProductCategory'] == 'MenClothing']

df = pd.concat([df1,df2])

df['Year'] = df['Year'].astype('str') 

#dd=pd.read_csv("../input/Train_Kaggle.csv")

#dd=dd.loc[dd['ProductCategory'] == 'WomenClothing']

data1=data1.drop(data1[data1['Year']>'2015'].index)

#data1=data1.drop(data1[data1['Year']<'2010'].index)

#dd['Year'] = dd['Year'].astype('str')

#dd.Year.dtype

df['Year'] = df['Year'].str.strip()

data2=pd.merge(data1,df, on=['Year','Month'])

data3=data2.replace('?', '0', regex=False)

data3=data3.replace('T', '0', regex=False)

#data3

holidays = pd.read_excel("../input/Events_HolidaysData.xlsx")

holidays['Month'] = pd.DatetimeIndex(holidays['MonthDate']).month

holidays["Year"] = holidays["Year"].astype(str)

holidays["Month"] = holidays["Month"].astype(str)

holidays["YearMonth"] = holidays["Year"] + "-" + holidays["Month"]

holidayscount = holidays["YearMonth"].value_counts()

holidayscount = pd.DataFrame(data=holidayscount)

holidayscount = holidayscount.rename(columns={'YearMonth':'NoOfHolidays'})

holidayscount['Month'] = pd.DatetimeIndex(holidayscount.index).month

holidayscount['Year']= pd.DatetimeIndex(holidayscount.index).year

holidayscount=holidayscount.drop(holidayscount[holidayscount['Year']>2015].index)

#holidayscount=holidayscount.drop(holidayscount[holidayscount['Year']<2010].index)

holidayscount['Year'] = holidayscount['Year'].astype('str')

data3 = pd.merge(data3,holidayscount, 'left', on = ['Month','Year'])

cols = data3.columns.tolist()

cols = cols[-1:] + cols[:-1]

data3 = data3[cols]

data3=data3.drop("Unnamed: 0",axis=1)

data3=data3.fillna(method='pad')

data3

#dh.set_datetimeIndex('Month').resample('MS').asfreq().fillna(0)

data31=data3

data31=data31.drop('ProductCategory',axis=1)

data31=data31.drop('WeatherEvent',axis=1)

data31=data31.drop('PartyInPower',axis=1)

#data31=data31.drop([60])

#data31 = data31.astype('int')

target_data=data31['Sales(In ThousandDollars)']

#data31=data31.drop('Sales(In ThousandDollars)',axis=1)

#data31

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data31=data31.convert_objects(convert_numeric=True)
def corr_features(df,cols,bar=0.9):

    for c,i in enumerate(cols[:-1]):

        col_set = set(cols)

        for j in cols[c+1:]:

            if i==j:

                continue

           

            score = df[i].corr(df[j])

            

            if score>bar:

                cols = list(col_set-set([j]))

            if score<-bar:

                cols = list(col_set-set([j]))

    return cols
data32=data31.drop(['Sea Level Press.\xa0(hPa) low',

 #'Monthly Real GDP Index (inMillion$)',

 'Temp low (°C)',

 #'Temp avg (°C)',

 #'CommercialBankInterestRateonCreditCardPlans',

 'Dew Point high (°C)',

 'Visibility\xa0(km) high',

 #'Visibility\xa0(km) avg',

 'Change(in%)',

 #'Month',

 #'Humidity\xa0(%) low',

 'Day',

 #'Sea Level Press.\xa0(hPa) avg',

 'Temp high (°C)',

 'Wind\xa0(km/h) low',

 #'yieldperharvested acre',

 #'Mill use  (in  480-lb netweright in million bales)',

 #'Wind\xa0(km/h) avg',

 'Wind\xa0(km/h) high',

 'Visibility\xa0(km) low',

 'AdvertisingExpenses (in Thousand Dollars)',

 'Humidity\xa0(%) high',

 #'Precip.\xa0(mm) sum',

 #'Cotton Monthly Price - US cents per Pound(lbs)',

 #'Exports',

 'Sea Level Press.\xa0(hPa) high',

 #'Average upland planted(million acres)',

 #'Dew Point avg (°C)',

 'Dew Point low (°C)'],axis=1)

 #'Average upland harvested(million acres)',

 #'Finance Rate on Personal Loans at Commercial Banks, 24 Month Loan'],axis=1)
#data31.corr[]

correlations = data31.corr()['Sales(In ThousandDollars)'].sort_values()



# Display correlations

#print('Most Positive Correlations:\n', correlations.tail(15))

#print('\nMost Negative Correlations:\n', correlations.head(15))

correlations
data32=data31.drop(['unemployment rate',

 #'Monthly Real GDP Index (inMillion$)',

 'Temp low (°C)',

 'Temp avg (°C)',

 'CommercialBankInterestRateonCreditCardPlans',

 'Dew Point high (°C)',

 #'Humidity (%) avg',

 'Visibility\xa0(km) avg',

 'Change(in%)',

 #'Month',

 'Humidity\xa0(%) low',

 'Day',

 'Sea Level Press.\xa0(hPa) avg',

 'Sea Level Press.\xa0(hPa) low',

 'Temp high (°C)',

 'Wind\xa0(km/h) low',

 'Wind\xa0(km/h) high',

 'yieldperharvested acre',

 'Mill use  (in  480-lb netweright in million bales)',

 'Wind\xa0(km/h) avg',

 'Production (in  480-lb netweright in million bales)',

 'Visibility\xa0(km) low',

 'Visibility\xa0(km) high',

 'AdvertisingExpenses (in Thousand Dollars)',

 'Humidity\xa0(%) high',

 'Precip.\xa0(mm) sum',

 'Cotton Monthly Price - US cents per Pound(lbs)',

 'Exports',

 'Sea Level Press.\xa0(hPa) high',

 'Average upland planted(million acres)',

 'Dew Point avg (°C)',

 'Dew Point low (°C)',

 'Average upland harvested(million acres)',

 'Finance Rate on Personal Loans at Commercial Banks, 24 Month Loan'],axis=1)

#data32=data32.drop([data32.columns[2],data32.columns[5],data32.columns[4],data32.columns[7]], axis=1)

#data32=data32.drop([data32.columns[2],data32.columns[5],data32.columns[1],data32.columns[7]], axis=1)

#data32=data32.drop([data32.columns[0]], axis=1)

#data32=data32.drop(data32.columns[5], axis=1)

#data32=data32.drop(data32.columns[8], axis=1)

data32=data32.drop(data32.columns[7], axis=1)

#data32=data32.drop(data32.columns[0], axis=1)

data32
#data_other=data32

#data32=data_other

#data_women=data32

#data_men=data32
data32=data32.convert_objects(convert_numeric=True)

data32=data32.fillna(method='pad')

x_train = data32.loc[data32['Year'] < 2015]

y_train = data32.loc[data32['Year'] < 2015, 'Sales(In ThousandDollars)']

x_test = data32.loc[data32['Year'] >= 2015].reset_index(drop=True)

y_test = data32.loc[data32['Year'] >= 2015, 'Sales(In ThousandDollars)'].reset_index(drop=True)

x_test=x_test.drop('Sales(In ThousandDollars)',axis=1)

x_train=x_train.drop('Sales(In ThousandDollars)',axis=1)

#x_test=x_test.drop('Year',axis=1)

#x_train=x_train.drop('Year',axis=1)

#preds = xboost(x_train, y_train, x_test)

#SMAPE(preds, y_test)

#women=preds
preds = xboost(x_train, y_train, x_test)

#SMAPE(preds, y_test)

men=preds

men
from sklearn.ensemble import RandomForestRegressor

m = RandomForestRegressor(n_estimators=100)

m.fit(x_train, y_train)

men=m.predict(x_test)

men
women
x_train = df.loc[df['Year'] < 2014]

y_train = df.loc[df['Year'] < 2014, 'Sales(In ThousandDollars)']

x_test = df.loc[df['Year'] >= 2014].reset_index(drop=True)

y_test = df.loc[df['Year'] >= 2014, 'Sales(In ThousandDollars)'].reset_index(drop=True)
import xgboost as xgb

def xboost(x_train, y_train, x_test):

    """Trains xgboost model and returns Series of predictions for x_test"""

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(x_train.columns))

    dtest = xgb.DMatrix(x_test, feature_names=list(x_test.columns))



    params = {'max_depth':3,

              'eta':0.2,

              'silent':1,

              'subsample':1}

    num_rounds = 1500



    bst = xgb.train(params, dtrain, num_rounds)

    

    return pd.Series(bst.predict(dtest))
def SMAPE (forecast, actual):

    """Returns the Symmetric Mean Absolute Percentage Error between two Series"""

    masked_arr = ~((forecast==0)&(actual==0))

    diff = abs(forecast[masked_arr] - actual[masked_arr])

    avg = (abs(forecast[masked_arr]) + abs(actual[masked_arr]))/2

    

    print('SMAPE Error Score: ' + str(round(sum(diff/avg)/len(forecast) * 100, 2)) + ' %')
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

#data = pandas.read_csv(url, names=names)

scatter_matrix(data31)

plt.show()

 

 

other.shape

#men
import csv

with open("team_14_submission_2.csv", 'w') as csvFile:

    writer = csv.writer(csvFile)

    print(csvFile)

    writer.writerow(['Year','Sales(In ThousandDollars)'])

    count = 0

    for i in range(12):

        print(women[i])

        print(men[i])

        print(other[i])

        writer.writerow([i + 1 + count,women[i]])

        writer.writerow([i + 2 + count,men[i]])

        writer.writerow([i + 3 + count,other[i]])

        count = count + 2
dd1=pd.read_csv("team_14_submission_2.csv")

dd1