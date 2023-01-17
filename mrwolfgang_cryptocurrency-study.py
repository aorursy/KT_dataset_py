
#Important Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")
%matplotlib inline
plt.rcParams['figure.figsize'] = (15, 7)
sns.set(style="whitegrid")
import missingno as msno
#Interactive
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython.display import display, HTML


HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
 $('div.cell.code_cell.rendered.selected div.input').hide();
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" class="btn btn-primary" value="Click here to toggle on/off the raw code."></form>''')


#Reading Data
data = pd.read_csv("../input/all_currencies.csv").iloc[:,1:]
data.head(5)
print("\n Number of rows in our dataset = " + str(data.shape[0]))
# Number of Symbols in our dataset
print("--- Symbols ---")
symbols_name=" , ".join(pd.unique(data['Symbol']).tolist())
print(symbols_name)
print("-------")
print("Number of symbols " + str(len(symbols_name)))
# Checking Missing values in our dataset
print("By Numbers",data.isnull().sum(),"-------","By Percentage",100*data.isnull().sum()/data.shape[0],sep="\n\n")

# By Range (High - Low)
range_per = (100*(data.groupby('Symbol').max()['High'] - data.groupby('Symbol').min()['Low'])/data.groupby('Symbol').min()['Low']).reset_index() 
range_per.rename(columns={0:'Range'},inplace=True)
range_per.sort_values('Range',ascending=False,inplace=True)

#Barplot of top 10 symbols by Range
fig=plt.figure()
bar=sns.barplot(x=range_per.iloc[0:10,0],y=range_per.iloc[0:10,1])
plt.show()

range_per.head(10)

#By Volume
volume_ = data.groupby('Symbol').sum()['Volume'].reset_index().sort_values('Volume',ascending=False)
volume_.head(10)

#Barplot of top 10 symbols by Range
fig=plt.figure()
bar=sns.barplot(x=volume_.iloc[0:10,0],y=volume_.iloc[0:10,1])
plt.show()


#Bitcoin
bitcoin = data[data["Symbol"]=="BTC"]
print("Bitcoin data :-")
bitcoin.head(5)
print("\n\nMissing Data in Bitcoin Data\n")
print("  By Numbers",bitcoin.isnull().sum(),"-------","  By Percentage",100*bitcoin.isnull().sum()/bitcoin.shape[0],sep="\n\n")

from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates

bitcoin['Date_n'] = pd.to_datetime(bitcoin['Date'])
bitcoin["Date_n"] = bitcoin["Date_n"].apply(mdates.date2num)

ohlc= bitcoin[['Date_n', 'Open', 'High', 'Low','Close']].copy()


print("Candlestick Chart of Bitcoin Dataset")
figure,ax=plt.subplots(figsize=(20,6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
candle=candlestick_ohlc(ax,ohlc.values)
plt.show()

print("\nCandlestick Chart of Bitcoin Dataset for last 100 days")
figure,ax=plt.subplots(figsize=(20,6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
candle=candlestick_ohlc(ax,ohlc.tail(100).values)
plt.show()
print("Visualizing Volume of Bitcoin for past 100 days")
figure,ax=plt.subplots(figsize=(20,6))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
vol_bar=sns.barplot(x=bitcoin.tail(100).Date,y=bitcoin.tail(100).Volume)
rot=plt.xticks(rotation=80)
plt.show(figure)
bitcoin['5-day-high']=bitcoin.High.rolling(5).mean()
bitcoin['5-day-low']=bitcoin.Low.rolling(5).mean()
bitcoin['5-day-close']=bitcoin.Close.rolling(5).mean()
bitcoin.tail(10)

bt100=bitcoin.tail(50)
figure,ax=plt.subplots()
rot=plt.xticks(rotation=60)
c=sns.lineplot(x=bt100.Date,y=bt100.Close)
h=sns.lineplot(x=bt100.Date,y=bt100['5-day-high'])
l=sns.lineplot(x=bt100.Date,y=bt100['5-day-low'])
c5=sns.lineplot(x=bt100.Date,y=bt100['5-day-close'])
leg=plt.legend({"Close":"Close","5-day-high":"5-day-high","5-day-low":"5-day-low","5-day-close":"5-day-close"})
plt.show()
bitcoin['55-day-close']=bitcoin.High.rolling(55).mean()
bitcoin['13-day-close']=bitcoin.Low.rolling(13).mean()
bitcoin["55/13 cross"]="Down"
bitcoin["55/13 cross"][bitcoin['13-day-close']>bitcoin['55-day-close']]="Up"
cp=sns.countplot(bitcoin["55/13 cross"])
plt.show()
sym=volume_.head(3)["Symbol"].tolist()
print("  ".join(sym))

ethereum=data[data["Symbol"]=="ETH"].tail(100)
ripple=data[data["Symbol"]=="XRP"].tail(100)
bitcoin=data[data["Symbol"]=="BTC"].tail(100)

ethereum.tail()
ripple.tail()
bitcoin.tail()
corr_df=pd.DataFrame()
corr_df['Date']=bitcoin.Date.tolist()
corr_df['XRP']=ripple.Close.tolist()
corr_df['ETH']=ethereum.Close.tolist()
corr_df['BTC']=bitcoin.Close.tolist()


corr_df.corr()
#scaling all sym prices to 1000 range to plot properly
corr_df['XRP']=(corr_df['XRP']-corr_df['XRP'].min())/(corr_df['XRP'].max()-corr_df['XRP'].min())*1000
corr_df['ETH']=(corr_df['ETH']-corr_df['ETH'].min())/(corr_df['ETH'].max()-corr_df['ETH'].min())*1000
corr_df['BTC']=(corr_df['BTC']-corr_df['BTC'].min())/(corr_df['BTC'].max()-corr_df['BTC'].min())*1000

line_p=corr_df.plot(kind="line",x="Date")