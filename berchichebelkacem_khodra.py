%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from datetime import datetime
from scipy import stats
import numpy as np
df = pd.read_csv("../input/all_prices.csv", sep=',')
df.head()
df['Date'] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.pivot(index='Date', columns='Product', values='Price')
df = df.infer_objects()
df = df.dropna(thresh=len(df.index)*0.8, axis=1)
df = df.fillna(df.mean())
df.head()
def calc_cpi(df):
    df['CPI'] = df.sum(axis=1)
    df['CPI rebased'] = df['CPI'] * 100 / df['CPI'].iloc[0]
    df['MA 20'] = df['CPI rebased'].rolling(20).mean()
    df['MA 50'] = df['CPI rebased'].rolling(50).mean()
    return df 

df = calc_cpi(df)
    
# Removing outliers using Z-scores. 
# TODO Do it earlier.
z = np.abs(stats.zscore(df))
z = np.nan_to_num(z)
df = df[(z < 3).all(axis=1)]
def plot_time_series(df, title):
    df['CPI rebased'].plot(color='grey', linewidth=1, alpha=0.4)
    df['MA 20'].plot(color='orange', linewidth=2, alpha=1)

    plt.grid()

    plt.axvspan(date2num(datetime(2018,5,16)), date2num(datetime(2018,6,14)), 
               label="Ramadhan 2018",color="green", alpha=0.3)

    plt.axvspan(date2num(datetime(2019,5,5)), date2num(datetime(2019,6,3)), 
               label="Ramadan 2019",color="green", alpha=0.3)

    plt.ylabel('Fruits & vegs, price')
    plt.title(title, size=18)

    plt.show()
    
plot_time_series(df, "Absolute prices")
df = df.iloc[:, :-4]
df = df / df.iloc[0, :]
df = calc_cpi(df)

plot_time_series(df, "Normalized prices")
weights = {
    "Carotte" : 6,
    "Pomme de terre blanche" : 10, 
    "Salade" : 4, 
    "Banane" : 2,
    "Citron" : 4,
    "Pomme (Locale)" : 3
}
df = df.iloc[:, :-4] 

for column in df.columns:
    if column in weights:
        df[column] = df[column] * weights.get(column)
        
df.head()
df = calc_cpi(df)
plot_time_series(df, "Normalized weighted prices")
