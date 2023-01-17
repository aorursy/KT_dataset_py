#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import plotly.graph_objs as go
import plotly.plotly as py
import datetime as dt
import matplotlib.dates as mdates

data = pd.read_csv("../input/crypto-markets.csv")
data.head()
#set up a column with an index so I can only sample 1/10th of the dataset 
data["row_id"]= range(1, len(data) + 1)
data.head()

#drop columns I'm not using
data_=data.drop(['slug','ranknow','volume','market'], axis=1)
data_.head()
data_.set_index('row_id', inplace=True)
data_.head()
#set date to timestamp format
data_['date'] = pd.to_datetime(data_['date'], format='%Y-%m-%d')
#pick out the currency for two years span (2016>)
date = data_[data_['date'] >= dt.date(2016, 1, 1)]
#show if each one closed up or down each day
date['pos_neg']= date['open']-date['close']
date.head()
#create a binary column - 0 = gain, 1 = loss to have something to predict
date['Up/Down'] = np.where(date['pos_neg']>0, '0', '1')
#create data sets for six crypto currencies
ltcdate = date[date['symbol']=='LTC']
zcashdate = date[date['symbol']=='ZEC']
rippledate = date[date['symbol']=='XRP']
etherdate = date[date['symbol']=='ETH']
mondate = date[date['symbol']=='XMR']
bitdate = date[date['symbol']=='BTC']
#date.head()
zcashdate.head()
#concat the six frames into one
frames=[bitdate,ltcdate,rippledate,etherdate,mondate,zcashdate]
top_six = pd.concat(frames)
top_six.info()
#this chart shows each of the six and the number of times they closed up for the day.  
sns.set_style('whitegrid')
sns.countplot(x='symbol',hue='Up/Down',data=top_six,palette='rainbow')
#a comparison the the up/down ratio at the end of the day for the six.
sns.set(style="whitegrid")

g = sns.factorplot("symbol", "close_ratio", "Up/Down",
                    data=top_six, kind="bar",
                    size=6, palette="muted",
                   legend_out=True)
g.despine(left=True)
g.set_ylabels("Ratio Count")
#This chart above suggests that even though the days with gains outnumbered (in most cases) the losses, the loss ratios
#tended to be about double the gain ratios. 
sixtest=top_six.drop('date',axis=1).drop('symbol',axis=1).drop('name',axis=1)
sixtest.head()
from sklearn.model_selection import train_test_split
X = sixtest[['open', 'high', 'low','close', 'close_ratio','spread']]
y = sixtest['Up/Down']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, 
                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))