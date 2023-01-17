#Install the package
!pip install yfinance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import yfinance as yf

# find the symbol (i.e., google the instrument + "yahoo finance") to any data series you are interested at 
# e.g., market/sector index ETF for your chosen country and various asset classes (e.g., Comex Gold's symbol is "GC=F")

symbols_list = ["IBGM.MI","DAX","EWG","EWGS","SDEU.L"]
start = dt.datetime(2019,9,1)
end = dt.datetime(2020,8,31)
data = yf.download(symbols_list, start=start, end=end)
data.head()
# filter column adjusted close
df = data['Adj Close']
df.head()
#The daily price of the 5 assets
plt.figure(figsize=(30,10))
df['DAX'].plot()
df['EWG'].plot()
df['EWGS'].plot()
df['IBGM.MI'].plot()
df['SDEU.L'].plot()
plt.ylabel("Daily returns of DAX, EWG, EWGS, IBGM.MI, SDEU.L")
plt.legend()
plt.show()
# pct_change for returns
# first element is NaN, so we remove
df_pct =df.pct_change()[1:]
df_pct.head()

#The percentage change of the assets
plt.figure(figsize=(30,10))
df_pct['DAX'].plot()
df_pct['EWG'].plot()
df_pct['EWGS'].plot()
df_pct['IBGM.MI'].plot()
df_pct['SDEU.L'].plot()
plt.ylabel("Daily returns of DAX, EWG, EWGS, IBGM.MI, SDEU.L")
plt.legend()
plt.show()

#NEED COVID DATA -- IMPORTING......
import statsmodels.api as sm
from statsmodels import regression

X = df_pct["EWG"]
y = df_pct["DAX"]

# Note the difference in argument order
X = sm.add_constant(X)
model = sm.OLS(y.astype(float), X.astype(float), missing='drop').fit()
predictions = model.predict(X.astype(float)) # make the predictions by the model

# Print out the statistics
print(model.summary())

#Import COVID Germany Data (Daily from Jan to Sep)
covid = pd.read_csv('../input/germany-covid19-janseptember/Germany COVID-19.csv')
covid = covid[['Date','Confirmed']]
covid["Date"] = pd.to_datetime(covid["Date"])
covid.tail()
#Import Exchange Rate Data

#Import Power Consumption Data
pw_consumption = pd.read_csv('../input/western-europe-power-consumption/de.csv')
pw_consumption['Date'] = pd.to_datetime(pw_consumption['end'])
pw_consumption['Date'] = pd.to_datetime(pw_consumption['Date']).dt.date
pw_consumption['Date'] = pd.to_datetime(pw_consumption['Date'])
pw_consumption.info()

pw_consumption = pw_consumption[['Date', 'load']]

#Resample data to days
pw_consumption = pw_consumption.resample('D', on = 'Date').sum()
pw_consumption = pw_consumption.sort_values('Date', ascending = False)
pw_consumption = pw_consumption[:700]

pw_consumption.head()
