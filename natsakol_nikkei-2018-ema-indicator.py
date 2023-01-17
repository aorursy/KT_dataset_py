import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

nikkei = pd.read_csv('../input/nikkie2018.csv')
nikkei.describe()
nikkei.head()
nikkei.tail()
EMA = pd.DataFrame(index=['Date']) #create new dataframe for EMA
EMA = nikkei[['Date','Close Price']][0:].copy() #copy Date and Close Price from nikkei dataframe
EMA.columns = ['Date', 'Close'] #rename column 'Close Price' to 'Close'
EMA['EMA12'] = EMA['Close'].ewm(12).mean() #12 period moving average
EMA['EMA26'] = EMA['Close'].ewm(26).mean() #26 period moving average
EMA.head()
rows = len(EMA) #count number of row of EMA dataframe

#separate negative and possitve EMA signal
for i in range (0,rows):
    if EMA.loc[i,'EMA12'] < EMA.loc[i,'EMA26']:
        EMA.loc[i,'NEGATIVE'] = EMA.loc[i,'EMA12']
    elif EMA.loc[i,'EMA12'] > EMA.loc[i,'EMA26']: 
        EMA.loc[i,'POSSITIVE'] = EMA.loc[i,'EMA12']
        
for i in range (0,rows):
    if (abs(EMA.loc[i,'EMA26']-EMA.loc[i,'EMA12']))<0.005*(EMA['EMA12'].max()-EMA['EMA12'].min()): # Cross over signal when the difference of EMA less than 0.5% of EMA12 range
        EMA.loc[i,'CROSSOVER']= EMA.loc[i,'EMA26'] 
fig = plt.figure(figsize=(20, 10),dpi=100) # fig size = 20x10 inches/ 100 dots per inch
ax1 = fig.add_subplot(221)

# figure1 ax1
ax1.set_title('NIKKEI PRICE INDEX: JAN-AUG 2018')

x = [dt.datetime.strptime(date,'%Y-%m-%d').date() for date in EMA['Date']] #convert x axis into datetime format
Close = EMA['Close'] 
EMA12 = EMA['EMA12']
EMA26 = EMA['EMA26']
POSSITIVE = EMA['POSSITIVE']
NEGATIVE = EMA['NEGATIVE']

ax1.plot(x, Close,color="gray", linewidth=1.0, linestyle="--",label='Close Price') #plot Close Price against x axis
ax1.plot(x, EMA26,color="black", linewidth=2.0, linestyle="-") #plot EMA26 against x axis
ax1.plot(x, POSSITIVE,color="green", linewidth=3.0, linestyle="-",label='EMA12 > EMA26') #plot EMA26 against x axis
ax1.plot(x, NEGATIVE,color="red", linewidth=3.0, linestyle="-",label='EMA12 < EMA26') #plot EMA26 against x axis
ax1.scatter(x, EMA['CROSSOVER'],color='blue',label='CROSSOVER RISK')
ax1.fill_between(x, EMA26, EMA12, where=EMA12>EMA26, facecolor='lightgreen')
ax1.fill_between(x, EMA26, EMA12, where=EMA12<EMA26, facecolor='lightcoral')
ax1.set_ylim = (20000.0, 25000.0) # Set limits on y axis
ax1.set_ylabel('Price') #set y-axis lebel
ax1.set_xlabel('Date') #set x-axis lebel
ax1.legend(loc='upper right') #set legand

plt.show() 
