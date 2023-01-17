# importing neccesary modules and files
import numpy as np,     plotly.express as px, pandas as pd,    matplotlib.pyplot as plt
import datetime as dt,  plotly,    plotly.io as pio,           plotly.offline as py
from fbprophet.plot import plot_plotly 
import pandas as pd
py.init_notebook_mode()
import os

data_import = pd.read_csv("../input/ai-hack-data/covid_19_data.csv",
                                parse_dates=['ObservationDate'],
                                index_col=['SNo'])
data = data_import.groupby(['Country/Region','ObservationDate']).sum().reset_index()
codes = pd.read_csv("../input/ai-hack-data/codes.csv")
# rename column, then renaming where neccesary
data_import = pd.read_csv("../input/ai-hack-data/covid_19_data.csv")
data = data_import.groupby(['Country/Region','ObservationDate']).sum().reset_index()


Usercountry=input("Enter Country Name to visualise -> ")
choosenCountry = dataGrouppedByCountry[dataGrouppedByCountry['Country'].isin([Usercountry])]


data.rename(columns={"Country/Region":"Country"},inplace=True)
dataGrouppedByCountry=data.groupby(['Country','ObservationDate']).sum().reset_index()
dataGrouppedByCountry['Country'] = np.where(dataGrouppedByCountry['Country']=="Mainland China",'China',dataGrouppedByCountry['Country'])

# merging the data sets so that we get the alpha codes for countries
newdf = pd.merge(dataGrouppedByCountry, codes, how='left', left_on='Country', right_on='Country ')
newdf.drop(['index','Country ','Alpha-2','Numeric'], axis=1, inplace=True)

d_UK = choosenCountry

new_uk = d_UK[['ObservationDate','Deaths']].copy()
new_uk.columns = ['ds', 'y'] # renaming for prophet
from fbprophet import Prophet
m = Prophet()
m.fit(new_uk)
future = m.make_future_dataframe(periods=100)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# Python
display(fig1 = m.plot(forecast))
# Python
display(fig2 = m.plot_components(forecast))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

#Importing the data file
data=pd.read_csv("../input/ai-hack-data/covid_19_data.csv")
datacodes=pd.read_csv("../input/ai-hack-data/codes.csv")
hdidata=pd.read_csv("../input/un-hdi/HDI.csv")

data.rename(columns={"Country/Region":"Country"},inplace=True)
data["DayofYear"] = pd.to_datetime(data["ObservationDate"]).dt.dayofyear
data.drop(['ObservationDate'], axis=1, inplace=True)
LE = LabelEncoder()
data['countrycode'] = LE.fit_transform(data['Country'])
countrycodes=data['countrycode']
dataGrouppedByCountry=data.groupby(['Country','DayofYear']).sum().reset_index()
dataGrouppedByCountry['Country'] = np.where(dataGrouppedByCountry['Country']=="Mainland China", 'China',dataGrouppedByCountry['Country'])
dataGrouppedByCountry.drop('SNo', axis=1, inplace=True)
newdf = pd.merge(dataGrouppedByCountry, datacodes, how='left', left_on='Country', right_on='Country ')
newdf.drop(['index','Country '], axis=1, inplace=True)

dataGroupedByDate=data.groupby(['DayofYear']).sum().reset_index()
dataGroupedByDate.drop('SNo', axis=1, inplace=True)

Usercountry=input("Enter Country Name to visualise -> ")
choosenCountry = dataGrouppedByCountry[dataGrouppedByCountry['Country'].isin([Usercountry])]

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

 
#print Basic info about the data
print(data.info())
print("\n \n ###############Empty Data Check##########")
print(data.isnull().sum())

 


data.rename(columns={"Country/Region":"Country"},inplace=True)
data["DayofYear"] = pd.to_datetime(data["ObservationDate"]).dt.dayofyear
data.drop(['ObservationDate'], axis=1, inplace=True)

 


LE = LabelEncoder()
data['countrycode'] = LE.fit_transform(data['Country'])
countrycodes=data['countrycode']
dataGrouppedByCountry=data.groupby(['Country','DayofYear']).sum().reset_index()

 

dataGrouppedByCountry['Country'] = np.where(dataGrouppedByCountry['Country']=="Mainland China", 'China',dataGrouppedByCountry['Country'])
dataGrouppedByCountry.drop('SNo', axis=1, inplace=True)

dataGroupedByDate=data.groupby(['DayofYear']).sum().reset_index()
dataGroupedByDate.drop('SNo', axis=1, inplace=True)

 

Usercountry=input("Enter Country Name to visualise -> ")
choosenCountry = dataGrouppedByCountry[dataGrouppedByCountry['Country'].isin([Usercountry])]

choosenCountry.drop(['Country','countrycode'], axis=1, inplace=True)
x=choosenCountry.loc[:, choosenCountry.columns != 'Confirmed']
y=choosenCountry['Confirmed']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.7,shuffle=False) 

 

from sklearn.ensemble import RandomForestRegressor
import time
RandomForestRF = RandomForestRegressor(random_state=0, n_estimators=150)
startTime=time.time()
RandomForestRF.fit(X_train, y_train)
predections=RandomForestRF.predict(X_test)
predections=pd.DataFrame(predections)
endTime=time.time()
print("Time Taken to Train -> ",endTime-startTime)
print("Predection Accuracy Score -> ",RandomForestRF.score(X_test, y_test)*100)
