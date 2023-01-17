import numpy as np

import pandas as pd
nba = pd.read_csv('../input/nba-players-stats/Seasons_Stats.csv')
# I will focus on the 13/14 & 14/15 season for now.



nba = nba[(nba['Year']<=2015) & (nba['Year']>=2014) & (nba['G']>10) & (nba['MP']>200)]
# the data contains the stats total but I want the stats per game, so I'm creating some new variables

nba['PTS'] = (nba['PTS'] / nba['G']).round(1)

nba['ORB'] = (nba['ORB'] / nba['G']).round(1)

nba['DRB'] = (nba['DRB'] / nba['G']).round(1)

nba['TRB'] = (nba['TRB'] / nba['G']).round(1)

nba['AST'] = (nba['AST'] / nba['G']).round(1)

nba['STL'] = (nba['STL'] / nba['G']).round(1)

nba['BLK'] = (nba['BLK'] / nba['G']).round(1)

nba['TOV'] = (nba['TOV'] / nba['G']).round(1)

nba['PF'] = (nba['PF'] / nba['G']).round(1)
# dropping some cols I don't need



nba = nba.drop(columns=['Unnamed: 0','blanl','WS/48','blank2','Tm','Pos'])
#Storing two dfs so I can train my model on the first one and check it on the other one



df = nba[(nba['Year']==2015)]

df2 = nba[(nba['Year']==2014)]



# players get traded during the season, resulting in 3 rows of the same player (stats in first team, second team and total)

# I'm dropping all entries except the stats total



df = df.drop_duplicates(subset=['Player'])

df2 = df2.drop_duplicates(subset=['Player'])
df
df.isnull().sum(axis = 0)
df.dtypes
df.sort_values('WS',ascending=False).head(20)
import matplotlib.pyplot as plt



df_WS_desc= df.sort_values('WS',ascending=False).head(10)



plt.figure(figsize=(10,6))

plt.barh('Player', 'WS',data=df_WS_desc)

plt.xlabel("Player", size=15)

plt.ylabel("WS", size=15)

plt.title("Top 10 WS", size=18)

plt.gca().invert_yaxis()





plt.show()
df_OWS_desc= df.sort_values('OWS',ascending=False).head(10)



plt.figure(figsize=(10,6))

plt.barh('Player', 'OWS',data=df_OWS_desc)

plt.xlabel("Player", size=15)

plt.ylabel("OWS", size=15)

plt.title("Top 10 OWS", size=18)

plt.gca().invert_yaxis()





plt.show()
df_DWS_desc= df.sort_values('DWS',ascending=False).head(10)



plt.figure(figsize=(10,6))

plt.barh('Player', 'DWS',data=df_DWS_desc)

plt.xlabel("Player", size=15)

plt.ylabel("DWS", size=15)

plt.title("Top 10 DWS", size=18)

plt.gca().invert_yaxis()





plt.show()
y = df['WS']

X = df[['PER','TS%','MP','ORB%','DRB%','TRB%','VORP','BPM','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']]





from sklearn.model_selection import train_test_split



num_test = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_test, random_state=100)
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



scale = MinMaxScaler()

model = LinearRegression()



X_train_sc = scale.fit_transform(X_train)

X_test_sc = scale.fit_transform(X_test)
model.fit(X_train_sc, y_train)



model.coef_
predict = model.predict(X_test_sc)



# the accuracy is ok, but the coefficients look odd

print((mean_squared_error(y_test, predict)))
predict[0:10]
y_test[0:10]
from sklearn.metrics import explained_variance_score, r2_score



# I'm taking some more metrics into account, they look fine.

print(explained_variance_score(y_test, predict, multioutput='raw_values'))
print(r2_score(y_test, predict))
s = df[['Player','WS','PER','TS%','MP','ORB%','DRB%','TRB%','VORP','BPM','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']]
s.sort_values('WS',ascending=True).head(20)
s.sort_values('ORB%',ascending=False).head(20)
y2 = df2['WS']

X2 = df2[['PER','TS%','MP','ORB%','DRB%','TRB%','VORP','BPM','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']]



X2_sc = scale.fit_transform(X2)
pred = model.predict(X2_sc)
print((mean_squared_error(y2, pred)))
df2.head(10)
pred[:10]
y2[:10]
print(explained_variance_score(y2, pred, multioutput='raw_values'))