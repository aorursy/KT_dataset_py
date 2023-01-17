import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
plotblue = '#008fd5'

#  resample data so it smooths out data points
def sp500_data():
    df = pd.read_csv('../input/sp_500_max.csv')
    df['Date'] = pd.to_datetime(df['Date'], errors = 'coerce')
    df = df.copy([['Date', 'Adj Close']])
    df.rename(columns={'Adj Close':'SP500'}, inplace=True)
    df["SP500"] = (df["SP500"]-df["SP500"][0]) / df["SP500"][0] * 100.0
    df = df.reset_index().set_index('Date').resample('1D').mean()
    df = df.resample('M').mean()
    df = df['SP500']
    return df

sp500 = sp500_data()
sp500.dropna(inplace=True)

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))
ax1.set_ylabel('Percent Change')

sp500.plot(color=plotblue, ax=ax1, legend=True, linewidth=2)

plt.show()
