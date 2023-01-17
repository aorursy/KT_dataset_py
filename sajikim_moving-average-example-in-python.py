import numpy as np

import pandas as pd
import datetime

import matplotlib.pyplot as plt

%matplotlib inline



def show_ma_chart(df):

    ma1=75

    ma2=100

    

    ma_1 = pd.Series.rolling(df.Price, window=ma1).mean()

    ma_2 = pd.Series.rolling(df.Price, window=ma2).mean()

 

    xdate = [x for x in df.index]

 

    plt.figure(figsize=(15,5))

    plt.style.use('ggplot')

 

    plt.plot(xdate, df.Price, lw=1, color="black",label="Price")

    plt.plot(xdate, ma_1,lw=3,linestyle="dotted", label="Moving Average {} days".format(ma1))

    plt.plot(xdate, ma_2,lw=3,linestyle="dotted", label="Moving Average {} days".format(ma2))

    plt.legend(loc='best')

    plt.xlabel('Date')

    plt.ylabel('Price')

    xmin = df.index[0]

    xmax = df.index[-1]

    plt.title("Cushing_OK_WTI_Spot_Price_FOB ({0} to {1})".format(xmin.date(),xmax.date()))

    plt.ylim(0, 100)

    plt.xlim(xmin, xmax)

    plt.show()
# Prepare dataset

# The U.S. Energy Information Administration (EIA)

df = pd.read_csv("../input/cushing-ok-wti-spot-price-fob/Cushing_OK_WTI_Spot_Price_FOB.csv", header=4, parse_dates=[0],index_col=[0],names=["Date","Price"])

df.sort_index(inplace=True)
# Show Moving Average

show_ma_chart(df['2015-01-01':])

show_ma_chart(df['2019-01-01':])