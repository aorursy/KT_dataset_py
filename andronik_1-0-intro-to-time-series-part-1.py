# imports libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns

import os
# move to Stocks directory
os.chdir("../input/price-volume-data-for-all-us-stocks-etfs/Stocks/")
# save stock data to stock_df
stock_df = []
csvs = [x for x in os.listdir() if x.endswith(".txt")]
# reading file with size zero throughts an error
csvs = [x for x in csvs if os.path.getsize(x) > 0]
for csv in csvs:
    df = pd.read_csv(csv)
    df['ticker'] = csv.replace('.us.txt','')
    stock_df.append(df)
stock_df = pd.concat(stock_df, ignore_index=True)
stock_df.reset_index(inplace=True, drop=True)
# stock df shape and head
print(stock_df.shape)
stock_df.head()
os.chdir('..')
path_parent = os.path.dirname(os.getcwd())
os.chdir(path_parent)
os.chdir("../input/price-volume-data-for-all-us-stocks-etfs/ETFs/")
# save etf data to etf_df
etf_df = []
csvs = [x for x in os.listdir() if x.endswith(".txt")]
# reading file with size zero throughts an error
csvs = [x for x in csvs if os.path.getsize(x) > 0]
for csv in csvs:
    df = pd.read_csv(csv)
    df['ticker'] = csv.replace('.us.txt','')
    etf_df.append(df)
etf_df = pd.concat(etf_df, ignore_index=True)
etf_df.reset_index(inplace=True, drop=True)
# etf df shape and head
print(etf_df.shape)
etf_df.head()
# Use sci-kit learn to take a random sample of the data so it is more managable
stock_train, stock_test = train_test_split(stock_df, train_size=0.80, test_size=0.20, random_state=42, shuffle=True)
stock_df_large, stock_df_small = train_test_split(stock_train, train_size=0.90, test_size=0.10, random_state=42, shuffle=True)
print("Large df size:", stock_df_large.shape)
print("Small df size: ", stock_df_small.shape)
# % of missing values
stock_df_small.isnull().sum()
# Distribution for Open
stock_df_small.Open[stock_df_small.ticker=='cfg'].plot(kind='hist');
# Distribution for Close
stock_df_small.Close[stock_df_small.ticker=='cfg'].plot(kind='hist');
# $CFG (Citizens Financial Group Inc)
stock_df_small[stock_df_small.ticker=='tsla']

# save to date frame and sort by date
tsla_df = stock_df_small[stock_df_small.ticker=='tsla'].sort_values(by="Date")
tsla_df.head()
# Time Series using Plotly Express
fig = px.line(tsla_df, x='Date',y='Close')
fig.show()
# Time Series using Axes of type date
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Open'],
    name='Open'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['High'],
    name='High'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Low'],
    name='Low'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Close'],
    name='Close'))

fig.update_layout(
    title='Tesla Inc',
    xaxis_title='Date',
    legend_title='$TSLA',
    font=dict(
        family="Courier New, monospace",
        size=18)
)

fig.show()
# Time Series Plot with Custom Date Range
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Open'],
    name='Open'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['High'],
    name='High'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Low'],
    name='Low'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Close'],
    name='Close'))

fig.update_layout(
    xaxis_range=['2016-01-01','2018-01-01'],
    title='Tesla Inc',
    xaxis_title='Date',
    legend_title='$TSLA',
    font=dict(
        family="Courier New, monospace",
        size=18)
)

fig.show()
# Time Series With Range Slider
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Open'],
    name='Open'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['High'],
    name='High'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Low'],
    name='Low'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Close'],
    name='Close'))

fig.update_layout(
    title='Tesla Inc',
    xaxis_title='Date',
    legend_title='$TSLA',
    font=dict(
        family="Courier New, monospace",
        size=18)
)

fig.update_xaxes(rangeslider_visible=True)

fig.show()
# Time Series With Range Slider Button!
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Open'],
    name='Open'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['High'],
    name='High'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Low'],
    name='Low'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Close'],
    name='Close'))

fig.update_layout(
    title='Tesla Inc',
    xaxis_title='Date',
    legend_title='$TSLA',
    font=dict(
        family="Courier New, monospace",
        size=18)
)

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.show()
# Customizing Tick Label Formatting by Zoom Level
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Open'],
    name='Open'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['High'],
    name='High'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Low'],
    name='Low'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Close'],
    name='Close'))

fig.update_layout(
    title='Tesla Inc',
    xaxis_title='Date',
    legend_title='$TSLA',
    font=dict(
        family="Courier New, monospace",
        size=18)
)

fig.update_xaxes(
    rangeslider_visible=True,
    tickformatstops = [
        dict(dtickrange=[None, 1000], value="%H:%M:%S.%L ms"),
        dict(dtickrange=[1000, 60000], value="%H:%M:%S s"),
        dict(dtickrange=[60000, 3600000], value="%H:%M m"),
        dict(dtickrange=[3600000, 86400000], value="%H:%M h"),
        dict(dtickrange=[86400000, 604800000], value="%e. %b d"),
        dict(dtickrange=[604800000, "M1"], value="%e. %b w"),
        dict(dtickrange=["M1", "M12"], value="%b '%y M"),
        dict(dtickrange=["M12", None], value="%Y Y")
    ]
)


fig.show()
# Hiding Weekends and Holidays!
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Open'],
    name='Open'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['High'],
    name='High'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Low'],
    name='Low'))

fig.add_trace(go.Scatter(
    x=tsla_df['Date'],
    y=tsla_df['Close'],
    name='Close'))

fig.update_layout(
    title='Tesla Inc',
    xaxis_title='Date',
    legend_title='$TSLA',
    font=dict(
        family="Courier New, monospace",
        size=18)
)

fig.update_xaxes(
    rangebreaks=[
        dict(bounds=["sat", "mon"]), #hide weekends
        dict(values=["2015-12-25", "2016-01-01"])  # hide Christmas and New Year's
    ]
)

fig.show()
sns.set(style="ticks")

sns.pairplot(tsla_df);
# $TSLA and $AAPL


# two different stocks
tsla = stock_df_small[stock_df_small.ticker=='tsla']
aapl = stock_df_small[stock_df_small.ticker=='aapl']

# frames
frames = [tsla, aapl]

# new dataframe
df = pd.concat(frames)

aapl_tsla_df = df.sort_values(by="Date")
aapl_tsla_df.head()
sns.set(style="ticks")

sns.pairplot(aapl_tsla_df, hue='ticker');
# k-means is a widely used clustering algorithm. 
# It creates ‘k’ similar clusters of data points. 
# Data instances that fall outside of these groups 
# could potentially be marked as anomalies.

from sklearn.cluster import KMeans

# select features form the dataset
data = tsla_df[['Open','Close','High','Low','Volume']]

# set range
n_cluster = range(1, 20)

# 
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(n_cluster, scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show();
from mpl_toolkits.mplot3d import Axes3D


# select features
X = tsla_df[['Open','Close','High','Low','Volume']]
X = X.reset_index(drop=True)

# KMean
km = KMeans(n_clusters=7)
km.fit(X)
km.predict(X)
labels = km.labels_

# plot
fig = px.scatter_3d(X, 
                    x=X.iloc[:,0], 
                    y=X.iloc[:,1], 
                    z=X.iloc[:,2],
                    color='Volume')

# tight layout
fig.update_layout(margin=dict(l=1, r=0, b=0, t=0))

fig.show()
# Now we need to find out the number of components (features) to keep.

from sklearn.preprocessing import StandardScaler


date = tsla_df[['Open','Close','High','Low','Volume']]
X = data.values
X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key = lambda x: x[0], reverse= True)
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

plt.figure(figsize=(10, 5))
plt.bar(range(len(var_exp)), var_exp, alpha=0.3, align='center', label='individual explained variance', color = 'g')
plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show();
tsla_df.head()
# First sort the rows in order. The two bottom rows a the outliers
tsla_df.sort_values(by=['Volume'], ascending=True)
tsla_df = tsla_df[tsla_df.Volume < 32354359]
tsla_df.sort_values(by=['Volume'], ascending=True)
# Data cleaning: removed outliers as determined by KMeans clustering and removed empty column
tsla_df.head()
tsla_df.drop('OpenInt', axis=1, inplace=True)
tsla_df
tsla_df.dtypes
# Feature engeneering
tsla_df['Date'] = pd.to_datetime(tsla_df['Date'],format='%Y-%m-%d')

tsla_df['year']=tsla_df['Date'].dt.year 
tsla_df['month']=tsla_df['Date'].dt.month 
tsla_df['day']=tsla_df['Date'].dt.day

tsla_df['dayofweek_num']=tsla_df['Date'].dt.dayofweek 
tsla_df['dayofweek_name']=tsla_df['Date'].dt.day_name()

tsla_df.head()
# drop multiple data/datetime
tsla_df.drop('Datetime', axis=1, inplace=True)
tsla_df.head()
# Chaikin Money Flow
tsla_df['chaikin_money_flow'] = ((tsla_df['Close']-tsla_df['Low'])-(tsla_df['High']-tsla_df['Close']))/(tsla_df['High']-tsla_df['Low'])


# Money Flow Volume
tsla_df['money_flow_volume'] = tsla_df['Volume']*tsla_df['chaikin_money_flow']

# drop oscillation attempt
tsla_df.drop('oscillator', axis=1, inplace=True)
tsla_df.head()
# categorical encoder
import category_encoders as ce

# initilize encoder
encoder = ce.OneHotEncoder(use_cat_names=True)

# tsla encoded df
tsla_df_encoded = encoder.fit_transform(tsla_df)

tsla_df_encoded.head()
