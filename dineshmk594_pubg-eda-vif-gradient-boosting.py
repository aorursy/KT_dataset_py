# Load libraries
import numpy as np
import pandas as pd
from pandas import read_csv
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#Load dataset
data_train=pd.read_csv("../input/train_V2.csv")
train_df=data_train.sample(n=100000)

train_df=train_df.drop(['Id','groupId','matchId'],axis=1)
train_df.info()
plt.title('Correlation B/w Winning % and other Independent Variable')
train_df.corr()['winPlacePerc'].sort_values().plot(kind='barh',figsize=(10,8))
travel_values=train_df[['walkDistance','rideDistance','swimDistance']].sum()
plt.pie(travel_values, explode=[0,0.1,0],
        labels=['Walking','Riding','Swimming'], autopct='%1.1f%%',startangle=90)
train_df[['walkDistance','rideDistance']].hist(bins=15, color='steelblue', 
                                                              edgecolor='black', linewidth=1.0,
                                                              xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 0.9, 0.9)) 
plt.subplots(1,1,figsize=(8,5))
plt.subplot(1,1,1,title='Relationship value[0-1]')
ax1=sns.heatmap(train_df[['walkDistance','rideDistance','swimDistance','winPlacePerc']].corr(),annot=True,center=True)
sns.scatterplot(train_df['walkDistance'],train_df['winPlacePerc'])
plt.subplots(2,2,figsize=(20,16))

plt.subplot(2,2,1)
DBNOS = pd.cut(train_df['DBNOs'], [-1, 0, 2, 5, 10, 60], 
               labels=['0_times','1-2_times', '3-5_times', '6-10_times', '10+_times'])

ax1=sns.boxplot(DBNOS,train_df['winPlacePerc'])

plt.subplot(2,2,2)
Revives = pd.cut(train_df['revives'], [-1, 0, 2, 5, 10, 60], 
               labels=['0_times','1-2_times', '3-5_times', '6-10_times', '10+_times'])

sns.boxplot(Revives,train_df['winPlacePerc'])

plt.subplot(2,2,3)
Heals = pd.cut(train_df['heals'], [-1, 0, 2, 5, 10, 60], 
               labels=['0_times','1-2_times', '3-5_times', '6-10_times', '10+_times'])

sns.boxplot(Heals,train_df['winPlacePerc'])

plt.subplot(2,2,4)
Boosts = pd.cut(train_df['boosts'], [-1, 0, 2, 5, 10, 60], 
               labels=['0_times','1-2_times', '3-5_times', '6-10_times', '10+_times'])

sns.boxplot(Boosts,train_df['winPlacePerc'])


plt.subplots(1,1,figsize=(10,5))
plt.subplot(1,1,1,title='Relationship value range [0-1]')
ax1=sns.heatmap(train_df[['boosts','heals','revives','DBNOs','winPlacePerc']].corr(),annot=True)
plt.subplots(1,2,figsize=(20,5))
plt.subplot(1,2,1,title='Boosts Vs. Heals')
sns.lineplot(train_df['boosts'],train_df['heals'])
plt.subplot(1,2,2,title='Boosts Vs. Winning Percentage(%)')
sns.lineplot(train_df['boosts'],train_df['winPlacePerc'])
train_df[['weaponsAcquired','damageDealt']].hist(bins=15, color='steelblue', 
                                                              edgecolor='black', linewidth=1.0,
                                                              xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 0.9, 0.9)) 
plt.subplots(1,1,figsize=(8,5))

plt.subplot(1,1,1)
Vehicle_destroys = pd.cut(train_df['vehicleDestroys'], [-1, 0, 1, 2, 3], 
               labels=['0_times','1_times', '2_times', '3_times'])

ax1=sns.boxplot(Vehicle_destroys,train_df['winPlacePerc'])
plt.subplots(1,1,figsize=(8,5))
plt.subplot(1,1,1,title='Relationship value range [0-1]')
ax1=sns.heatmap(train_df[['weaponsAcquired','damageDealt','vehicleDestroys','winPlacePerc']].corr(),annot=True)
plt.subplots(1,2,figsize=(20,5))
plt.subplot(1,2,1,title='weaponsAcquired Vs. Damagedealt')
sns.lineplot(train_df['weaponsAcquired'],train_df['damageDealt'])
plt.subplot(1,2,2,title='weaponsAcquired Vs. Winning Percentage(%)')
sns.lineplot(train_df['weaponsAcquired'],train_df['winPlacePerc'])

train_df[['longestKill','killPlace']].hist(bins=15, color='steelblue', 
                                                              edgecolor='black', linewidth=1.0,
                                                              xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 0.9, 0.9)) 
plt.subplots(1,1,figsize=(8,5))

plt.subplot(1,1,1)
Kill_streak = pd.cut(train_df['killStreaks'], [-1, 0, 1, 2, 3,10], 
               labels=['0_times','1_times', '2_times', '3_times','3+times'])

ax1=sns.boxplot(Kill_streak,train_df['winPlacePerc'])
plt.title('Killing Place(position) Vs. Winning percentage')
sns.lineplot(train_df['killPlace'],train_df['winPlacePerc'])
plt.subplots(1,1,figsize=(8,5))
plt.subplot(1,1,1,title='Relationship value range [0-1]')
ax1=sns.heatmap(train_df[['longestKill','killStreaks','killPlace','winPlacePerc']].corr(),annot=True)
plt.title('Killing Place(position) Vs. killstreak')
sns.lineplot(train_df['killPlace'],train_df['killStreaks'])
plt.subplots(1,2,figsize=(20,8))

plt.subplot(1,2,1)
Kills = pd.cut(train_df['kills'], [-1, 0, 2, 5, 10, 60], 
               labels=['0_times','1-2_times', '3-5_times', '6-10_times', '10+_times'])

ax1=sns.boxplot(Kills,train_df['winPlacePerc'])

plt.subplot(1,2,2)
Headshot = pd.cut(train_df['headshotKills'], [-1, 0, 2, 5, 10, 60], 
               labels=['0_times','1-2_times', '3-5_times', '6-10_times', '10+_times'])

sns.boxplot(Headshot,train_df['winPlacePerc'])




X = np.arange(train_df['roadKills'].value_counts().count())[1:]
x=train_df['roadKills'].value_counts().count()-train_df['teamKills'].value_counts().count()
a=list(train_df['teamKills'].value_counts().sort_index()[1:])  # values taken from one and exclues zero.
a.extend(list(np.round(np.zeros(x))))  # Added three zeros because team kills only have 5 unique values
b=list(train_df['roadKills'].value_counts().sort_index()[1:])

ax1=plt.bar(X + 0.00, a, color = 'b', width = 0.25)
ax2=plt.bar(X + 0.25, b, color = 'g', width = 0.25)
plt.legend((ax1[0],ax2[0]),('teamkills','roadkills'))
plt.show()

plt.subplots(1,1,figsize=(8,5))
plt.subplot(1,1,1,title='Relationship value range [0-1]')
ax1=sns.heatmap(train_df[['headshotKills','kills','roadKills','teamKills','winPlacePerc']].corr(),annot=True)
data_match=train_df[['matchType','winPlacePerc']]
data_match=pd.get_dummies(data_match)
plt.title('Match type relationship with winning percentage(%)')
data_match.corr()['winPlacePerc'][1:].sort_values().plot.barh()
plt.subplots(2,2,figsize=(20,16))

plt.subplot(2,2,1)
winpt = pd.cut(train_df['winPoints'], [-1, 0, 200, 500, 1000, 2000,5000], 
               labels=['<0_pts','1-200_pts', '201-500_pts', '501-1000_pts', 
                       '1001-2000_pts','2000+_pts'])

ax1=sns.boxplot(winpt,train_df['winPlacePerc'])

plt.subplot(2,2,2)
rankpt = pd.cut(train_df['rankPoints'], [-1, 0, 200, 500, 1000, 2000,5000,10000], 
               labels=['<0_pts','1-200_pts', '201-500_pts', '501-1000_pts', 
                       '1001-2000_pts','2000-5000_pts','5000+_pts'])


sns.boxplot(rankpt,train_df['winPlacePerc'])

plt.subplot(2,2,3)
killpt = pd.cut(train_df['killPoints'], [-1, 0, 200, 500, 1000, 2000,5000,10000], 
               labels=['<0_pts','1-200_pts', '201-500_pts', '501-1000_pts', 
                       '1001-2000_pts','2000-5000_pts','5000+_pts'])

sns.boxplot(killpt,train_df['winPlacePerc'])

plt.subplot(2,2,4)
numgrp = pd.cut(train_df['numGroups'], [-1, 0, 20, 40, 60, 80,100,120], 
               labels=['0_grp','1-20_grp', '21-40_grp','41-60_grp','61-80_grp','81-100_grp',
                      '101+_grp'])

sns.boxplot(numgrp,train_df['winPlacePerc'])


train_df.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['matchType']=le.fit_transform(train_df['matchType'])

from statsmodels.stats.outliers_influence import variance_inflation_factor
x_features=list(train_df)
data_mat = train_df[x_features].as_matrix()                                                                                                              
vif = [ variance_inflation_factor( data_mat,i) for i in range(data_mat.shape[1]) ]
vif_factors = pd.DataFrame()
vif_factors['column'] = list(x_features)
vif_factors['vif'] = vif     
vif_factors.sort_values(by=['vif'],ascending=False)[0:10]
x_features.remove('maxPlace')
x_features.remove('numGroups')
x_features.remove('winPoints')
x_features.remove('rankPoints')
x_features.remove('killPoints')
x_features.remove('matchDuration')
data_mat = train_df[x_features].as_matrix()                                                                                                              
vif = [ variance_inflation_factor( data_mat,i) for i in range(data_mat.shape[1]) ]
vif_factors = pd.DataFrame()
vif_factors['column'] = list(x_features)
vif_factors['vif'] = vif     
vif_factors.sort_values(by=['vif'],ascending=False)[0:10]
train_df.corr()['kills'].sort_values(ascending=False)[:10]
x_features.remove('winPlacePerc')
x_features.remove('headshotKills')
data_mat = train_df[x_features].as_matrix()                                                                                                              
vif = [ variance_inflation_factor( data_mat,i) for i in range(data_mat.shape[1]) ]
vif_factors = pd.DataFrame()
vif_factors['column'] = list(x_features)
vif_factors['vif'] = vif     
vif_factors.sort_values(by=['vif'],ascending=False)[0:10]
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
data_train=data_train[data_train['winPlacePerc'].isnull()==False]
data_train['matchType']=le.fit_transform(data_train['matchType'])
#For building model i am using full data set.
X=data_train[x_features]
Y=data_train['winPlacePerc']
# Split-out validation dataset
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
model1=GradientBoostingRegressor(learning_rate=0.8)
model1=model1.fit(X_train,Y_train)
print(model1.score(X_train,Y_train))
print(model1.score(X_validation,Y_validation))
test_df=pd.read_csv('../input/test_V2.csv')
test_df['matchType']=le.fit_transform(test_df['matchType'])
X_test=test_df[x_features]
pred=pd.DataFrame(model1.predict(X_test),test_df['Id'])
pred.rename({0:'winPlacePerc'},axis=1).to_csv('sample_submission.csv')
