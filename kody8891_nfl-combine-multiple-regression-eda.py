import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

df=pd.read_csv('../input/nfl-combine-data/combine_data_since_2000_PROCESSED_2018-04-26.csv')

df.head()
df['Pick'] = df.Pick.fillna(260)

df
# Predict AV based on factors

# Predict drafted 

# drafted variable

# clustering for positions
df.dtypes
df.isna().sum()
# Height vs Weight

sns.regplot(x='Wt',y='Ht',data=df)
qb=df[df['Pos']=='QB']

activeqb = df[df['AV'] > 4]

from scipy import stats

def r2(Ht, AV):

    return stats.pearsonr(Ht, AV)[0] ** 2

sns.jointplot(x='Ht',y='AV',data=activeqb, kind="reg", stat_func=r2)
# corrplot

plt.figure(figsize=(8,9))

corr = df.corr()

sns.heatmap(corr,square=True,linewidths=.5, cbar_kws={"shrink": .5},cmap="binary")
plt.figure(figsize=(20,9))

sns.violinplot(x='Round',y='AV',data=df,palette="Set3", bw=.2, cut=1, linewidth=1)
df.Pos.value_counts().iloc[:10].plot(kind='barh')
df.Pos.unique()
line = df[(df['Pos'] == 'OT') | (df['Pos'] == 'OG') | (df['Pos']=='EDGE') | (df['Pos'] == 'NT') | (df['Pos'] == 'DT') | (df['Pos'] == 'DE') |(df['Pos'] == 'C')]

oskill= df[(df['Pos'] == 'QB') | (df['Pos'] == 'RB') | (df['Pos']=='WR') | (df['Pos'] == 'TE') | (df['Pos'] == 'FB')]

special=df[(df['Pos'] == 'K') | (df['Pos'] == 'P') | (df['Pos']=='LS')]

db = df[(df['Pos'] == 'SS') | (df['Pos'] == 'FS') | (df['Pos']=='S') | (df['Pos'] == 'CB') | (df['Pos'] == 'DB')]

lb = df[(df['Pos'] == 'OLB') | (df['Pos'] == 'ILB') | (df['Pos']=='LB') | (df['Pos'] == 'EDGE')]
sns.violinplot(x='Pos',y='Ht',data=line,palette="Set3", bw=.2, cut=1, linewidth=1)

plt.title('Lineman Height by Position')

plt.show()

sns.violinplot(x='Pos',y='Wt',data=line,palette="Set3", bw=.2, cut=1, linewidth=1)

plt.title('Lineman Weight by Position')

plt.show()

sns.violinplot(x='Pos',y='Ht',data=oskill,palette="Set3", bw=.2, cut=1, linewidth=1)

plt.title('Offensive Skill Height by Position')

plt.show()

sns.violinplot(x='Pos',y='Wt',data=oskill,palette="Set3", bw=.2, cut=1, linewidth=1)

plt.title('Offensive Skill Weight by Position')

plt.show()

sns.violinplot(x='Pos',y='Ht',data=special,palette="Set3", bw=.2, cut=1, linewidth=1)

plt.title('Special Teamer Height by Position')

plt.show()

sns.violinplot(x='Pos',y='Wt',data=special,palette="Set3", bw=.2, cut=1, linewidth=1)

plt.title('Special Teamer Weight by Position')

plt.show()

sns.violinplot(x='Pos',y='Ht',data=db,palette="Set3", bw=.2, cut=1, linewidth=1)

plt.title('Defensive Back Height by Position')

plt.show()

sns.violinplot(x='Pos',y='Wt',data=db,palette="Set3", bw=.2, cut=1, linewidth=1)

plt.title('Defensive Back Weight by Position')

plt.show()

sns.violinplot(x='Pos',y='Ht',data=lb,palette="Set3", bw=.2, cut=1, linewidth=1)

plt.title('Linebacker Height by Position')

plt.show()

sns.violinplot(x='Pos',y='Wt',data=lb,palette="Set3", bw=.2, cut=1, linewidth=1)

plt.title('Linebacker Weight by Position')

plt.show()
df['Forty']=df.groupby(["Pos"]).Forty.apply(lambda x: x.fillna(x.median()))

df['Vertical']=df.groupby(["Pos"]).Vertical.apply(lambda x: x.fillna(x.median()))

df['BenchReps']=df.groupby(["Pos"]).BenchReps.apply(lambda x: x.fillna(x.median()))

df['BroadJump']=df.groupby(["Pos"]).BroadJump.apply(lambda x: x.fillna(x.median()))

df['Cone']=df.groupby(["Pos"]).Cone.apply(lambda x: x.fillna(x.median()))

df['Shuttle']=df.groupby(["Pos"]).Shuttle.apply(lambda x: x.fillna(x.median()))

df['Pick']= df['Pick'].astype(int)

df=df.dropna(subset=['Cone','Shuttle'])
# linear regression to predict AV. Try Ht,Wt,Forty,Vertical,Bench,Braod,Cone,Shuttle,Pick

X = df[['Ht','Wt','Forty','Vertical','BenchReps','BroadJump','Cone','Shuttle','Pick']]

Y = df[['AV']]

model = LinearRegression()

model.fit(X, Y)

model = LinearRegression().fit(X, Y)

import statsmodels.api as sm    

model = sm.OLS(Y, X).fit()

predictions = model.predict(X)

model.summary()
X = df[['Wt','Vertical','Cone','Pick']]

Y = df[['AV']]

model = LinearRegression()

model.fit(X, Y)

model = LinearRegression().fit(X, Y)

model = sm.OLS(Y, X).fit()

predictions = model.predict(X)

model.summary()
brady=df[df['Player'] =='Tom Brady']

brady.head()
X_predict = brady[['Wt','Vertical','Cone','Pick']]  # put the dates of which you want to predict kwh here

y_predict = model.predict(X_predict)



# Tom Brady actual value over predicted value

brady['AV'] - y_predict