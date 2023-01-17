import numpy as np 
import pandas as pd 
from itertools import chain
import os
print(os.listdir("../input"))
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()
market_train_df.tail()
market_train_df.shape
market_train_df.columns
market_train_df.info()
market_train_df.describe()
market_train_df.isnull().any()
market_train_df.isnull().sum()*100/market_train_df.shape[0]
import matplotlib.pyplot as plt
percent = (market_train_df.isnull().sum()*100/market_train_df.shape[0]).sort_values(ascending=False)
percent.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Value Percent(%)", fontsize = 20)
plt.title("Total Missing Value by market_obs_df", fontsize = 20)
plt.figure(figsize=(10,10))
market_train_df['returnsClosePrevMktres1'].plot(kind='box')
( market_train_df['returnsClosePrevMktres1']<market_train_df['returnsClosePrevMktres1'].mean()).value_counts()
( market_train_df['returnsClosePrevMktres1']<market_train_df['returnsClosePrevMktres1'].median()).value_counts()
market_train_df['returnsClosePrevMktres1'].fillna(market_train_df['returnsClosePrevMktres1'].mean(),inplace=True)
market_train_df.isnull().any()
market_train_df['returnsOpenPrevMktres1'].plot.box()
market_train_df['returnsClosePrevMktres10'].plot.box()
market_train_df['returnsOpenPrevMktres10'].plot.box()
market_train_df['returnsOpenPrevMktres1'].fillna(market_train_df['returnsOpenPrevMktres1'].median(),inplace=True)
market_train_df['returnsClosePrevMktres10'].fillna(market_train_df['returnsClosePrevMktres10'].median(),inplace=True)
market_train_df['returnsOpenPrevMktres10'].fillna(market_train_df['returnsOpenPrevMktres10'].median(),inplace=True)
market_train_df.isnull().any()

news_train_df.head()
news_train_df.tail()
news_train_df.shape
news_train_df.columns
news_train_df.info()
news_train_df.describe()
news_train_df.isnull().any()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
data = []
for asset in np.random.choice(market_train_df['assetCode'].unique(), 10):
    asset_df = market_train_df[(market_train_df['assetCode'] == asset)]

    data.append(go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Closing prices of 10 random assets",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data=data, layout=layout), filename='basic-line')
market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']
market_train_df.sort_values('price_diff')[:10]
news_train_df = news_train_df.loc[news_train_df['time'] >= '2009-01-01 22:00:00+0000']
market_train_df = market_train_df.loc[market_train_df['time'] >= '2009-01-01 22:00:00+0000']
corr=market_train_df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(corr.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)
plt.show()
corr_with_mkt = market_train_df.corr()["returnsOpenNextMktres10"].sort_values(ascending=False)
plt.figure(figsize=(14,7))
corr_with_mkt.drop("returnsOpenNextMktres10").plot.bar()
plt.show()
corr2=news_train_df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr2,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(corr2.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(corr2.columns)
ax.set_yticklabels(corr2.columns)
plt.show()
corr_with_mkt = news_train_df.corr()["sentimentClass"].sort_values(ascending=False)
plt.figure(figsize=(14,7))
corr_with_mkt.drop("sentimentClass").plot.bar()
plt.show()
market = market_train_df.head(1_500_000)
news = news_train_df.head(3_500_000)
news_cols_agg = {
    'urgency': ['min', 'count'],
    'takeSequence': ['max'],
    'bodySize': ['min', 'max', 'mean'],
    'wordCount': ['min', 'max', 'mean'],
    'sentenceCount':['min', 'max', 'mean'],
    'companyCount': ['min', 'max', 'mean'],
    'relevance': ['min', 'max', 'mean'],
    'sentimentClass': ['min', 'max', 'mean'],
    'sentimentNegative': ['min', 'max', 'mean'],
    'sentimentNeutral': ['min', 'max', 'mean'],
    'sentimentPositive': ['min', 'max', 'mean'],
    'sentimentWordCount': ['min', 'max', 'mean'],
    'noveltyCount12H': ['min', 'max', 'mean'],
    'noveltyCount24H': ['min', 'max', 'mean'],
    'noveltyCount3D': ['min', 'max', 'mean'],
    'noveltyCount5D': ['min', 'max', 'mean'],
    'noveltyCount7D':['min', 'max', 'mean'],
    'volumeCounts12H': ['min', 'max', 'mean'],
    'volumeCounts24H': ['min', 'max', 'mean'],
    'volumeCounts3D': ['min', 'max', 'mean'],
    'volumeCounts5D': ['min', 'max', 'mean'],
    'volumeCounts7D': ['min', 'max', 'mean']
}
def merge_data(news_train_df,market_train_df):
    #rendre les assetCodes dans une liste
    news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'")
    news_train_df.time= news_train_df.time.dt.date
    market_train_df.time= market_train_df.time.dt.date
    #faire sortir élément par élément de la liste
    assetCodes_expanded = list(chain(*news_train_df['assetCodes']))
    #création d'un array ayant les indexes répétés de chaque ligne dupliquée
    assetCodes_index = news_train_df.index.repeat( news_train_df['assetCodes'].apply(len) )
    #Création d'un dataframe contenant les assetCodes et leurs indexes
    df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})
    # Creation d'un news dataframe où les lignes sont répétées
    news_cols = ['time', 'assetCodes'] + sorted(news_cols_agg.keys())
    #je dois ajouter get dummies pour provider à ce niveau là
    news_train_df_expanded = pd.merge(df_assetCodes, news_train_df[news_cols], left_on='level_0',
                                  right_index=True, suffixes=(['','_old']))
     # Free memory
    del news_train_df, df_assetCodes
    news_train_df_aggregated = news_train_df_expanded.groupby(['time', 'assetCode']).agg(news_cols_agg)
     # Free memory
    del news_train_df_expanded
    # Convert to float32 to save memory
    news_train_df_aggregated = news_train_df_aggregated.apply(np.float32)
    # Flat columns
    news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]
    # Join with train
    market_train_df = market_train_df.join(news_train_df_aggregated, on=['time', 'assetCode'])
    # Free memory
    del news_train_df_aggregated
    return market_train_df
data_m=merge_data(news,market)
data_m=data_m.dropna()
data_m.shape 
corr_with_mkt1 = data_m.corr()["returnsOpenNextMktres10"].sort_values(ascending=False)
plt.figure(figsize=(14,7))
corr_with_mkt1.drop("returnsOpenNextMktres10").plot.bar()
plt.show()
Y=data_m.returnsOpenNextMktres10.values 
data_f=data_m.drop(['returnsOpenNextMktres10'],axis=1)
data_final=data_f.drop(['assetName'],axis=1) #On a deja l'assetCode 
print(data_final.dtypes)
from sklearn import preprocessing
del data_final['time']
le = preprocessing.LabelEncoder()
le.fit(data_final['assetCode'])
data_final['assetCode']=le.fit_transform(data_final['assetCode'])
data_final=data_final.astype(float)
print(data_final.dtypes)
X=data_final.values
from sklearn.preprocessing import MinMaxScaler,Imputer
# définir un dictionnaire pour stocker nos rankings 
ranks = {}
# créer une fonction qui stocke le classement des caractéristiques 
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import StandardScaler
X_scaled=StandardScaler().fit_transform(X)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,LogisticRegression)
colnames=data_final.columns
lr = LinearRegression(normalize=True)
lr.fit(X_scaled,Y)
ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)

#  Ridge 
ridge = Ridge(alpha = 7)
ridge.fit(X_scaled,Y)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

#  Lasso
lasso = Lasso(alpha=.05)
lasso.fit(X_scaled, Y)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)
r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print("\t%s" % "\t".join(methods))
for name in colnames:
    print("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods]))))
# mettre la moyenne dans un dataframe Pandas
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Trier le  dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
import seaborn as sns
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
               size=20, aspect=1.9, palette='coolwarm')
features=list(meanplot[meanplot['Mean Ranking']!=0].Feature)
