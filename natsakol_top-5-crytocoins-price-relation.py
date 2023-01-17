import pandas as pd 
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from scipy import stats

warnings.filterwarnings("ignore")
raw = pd.read_csv('../input/CryptocoinsHistoricalPrices.csv') #read the raw dataset
clean=raw.dropna(axis=0, inplace=False, thresh=7)  
clean=clean.drop(['Unnamed: 0','Open.','Close..'], axis=1, inplace=False)
clean= clean[clean['Market.Cap']!='-']
clean['Market.Cap'] = clean['Market.Cap'].str.replace(',', '')
clean['Volume'] = clean['Volume'].str.replace(',', '')

cols= [ 'coin','Date','Open','Close','High','Low','Volume','Market.Cap','Delta']
clean = clean[cols]

obj=clean.select_dtypes(include=object).columns.tolist()
obj.remove('coin')
obj.remove('Date')
clean[obj]=clean[obj].convert_objects(convert_numeric=True)

clean['Market.Cap']=clean['Market.Cap'].astype(float)
#select data for 2018
clean['Year'] = clean['Date'].apply(lambda x: x.split('-')[0])
coin2018 = clean[clean['Year']=='2018']
RankMarketCap = clean.groupby('coin').mean().sort_values(by='Market.Cap',ascending=False)
#select top 5 coins by average market capital
TopCoin = RankMarketCap.head(5)
#show top coins and their 5 years volume
TopCoin ['Market.Cap']
Name = list(TopCoin.index)
#Separate and store data of top coin and store in the new dataframe names as the name of the coin
gbl = globals()
for i in Name:
    gbl[i] = coin2018[coin2018['coin']==i][['Date','Open','Close']].copy()
    gbl[i]['%Price Change'] = (gbl[i]['Close']-gbl[i]['Open'])*100/gbl[i]['Open']
    gbl[i].set_index('Date', inplace=True)
#merge dataframe of each coin
Price_Change_Percentage = []
for i in Name:
    temp = gbl[i].rename(index=str, columns={'%Price Change': i})
    temp2 = temp[i]
    Price_Change_Percentage.append(temp2)
Price_Change_Percentage = pd.concat(Price_Change_Percentage, axis=1,sort=False)

#statistical data
Price_Change_Percentage.describe()
fig1 = plt.figure(figsize=(10,8))
ax1 = fig1.add_subplot(111)
ax1 = sns.heatmap(Price_Change_Percentage.corr(),annot=True,cmap="afmhot")
plt.show()
fig2 = plt.figure(figsize=(10,10))
sns.pairplot(data=Price_Change_Percentage,diag_kind="kde",kind="reg")
plt.show()
fig3 = plt.figure(figsize=(10,10))

ax31 = fig3.add_subplot(221)
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(Price_Change_Percentage['BTC'],Price_Change_Percentage['ETH'])
ax31 = sns.regplot(x='BTC', y='ETH', data=Price_Change_Percentage, 
line_kws={'label':"\u0394ETH = {0:.1f} \u0394BTC + {1:.1f}".format(slope1,intercept1)})
ax31 = sns.kdeplot(data=Price_Change_Percentage,vars = ['BTC', 'ETH'],cmap = 'Reds',alpha = 0.35, shade=True, shade_lowest=False)
ax31.legend()

ax32 = fig3.add_subplot(222)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(Price_Change_Percentage['BTC'],Price_Change_Percentage['BCH'])
ax32 = sns.regplot(x='BTC', y='BCH', data=Price_Change_Percentage, 
line_kws={'label':"\u0394BCH = {0:.1f} \u0394BTC + {1:.1f}".format(slope2,intercept2)})
ax32 = sns.kdeplot(data=Price_Change_Percentage,vars = ['BTC', 'BCH'],cmap = 'Reds',alpha = 0.35, shade=True, shade_lowest=False)
ax32.legend()

ax33 = fig3.add_subplot(223)
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(Price_Change_Percentage['BTC'],Price_Change_Percentage['ADA'])
ax33 = sns.regplot(x='BTC', y='ADA', data=Price_Change_Percentage, 
line_kws={'label':"\u0394ADA = {0:.1f} \u0394BTC + {1:.1f}".format(slope3,intercept3)})
ax33 = sns.kdeplot(data=Price_Change_Percentage,vars = ['BTC', 'ADA'],cmap = 'Reds',alpha = 0.35, shade=True, shade_lowest=False)
ax33.legend()

ax34 = fig3.add_subplot(224)
slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(Price_Change_Percentage['BTC'],Price_Change_Percentage['XRP'])
ax34 = sns.regplot(x='BTC', y='XRP', data=Price_Change_Percentage, 
line_kws={'label':"\u0394XRP = {0:.1f} \u0394BTC + {1:.1f}".format(slope4,intercept4)})
ax34 = sns.kdeplot(data=Price_Change_Percentage,vars = ['BTC', 'XRP'],cmap = 'Reds',alpha = 0.35, shade=True, shade_lowest=False)
ax34.legend()

plt.show()