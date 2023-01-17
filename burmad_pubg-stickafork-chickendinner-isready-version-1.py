import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
#Read the training dataset and display the first 5 rows
train=pd.read_csv('../input/train.csv')
train.head()
train.shape   #shape of the training dataset
train.describe() #Description of the training dataset
train.isna().sum()
train.dtypes    #Datatypes of the attributes
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
fig, axes=plt.subplots(4,4, figsize=(26,20))
sns.distplot(train['kills'], kde=True, ax=axes[0,0])
sns.distplot(train['assists'], kde=True, ax=axes[0,1])
sns.distplot(train['boosts'], kde=True, ax=axes[0,2])
sns.distplot(train['killStreaks'], kde=True, ax=axes[0,3])
sns.distplot(train['revives'], kde=True, ax=axes[1,0])
sns.distplot(train['killPoints'], kde=True, ax=axes[1,1])
sns.distplot(train['killPlace'], kde=True, ax=axes[1,2])
sns.distplot(train['headshotKills'], kde=True, ax=axes[1,3])
sns.distplot(train['killStreaks'], kde=True, ax=axes[2,0])
sns.distplot(train['damageDealt'], kde=True, ax=axes[2,1])
sns.distplot(train['DBNOs'], kde=True, ax=axes[2,2])
sns.distplot(train['longestKill'], kde=True, ax=axes[2,3])
sns.distplot(train['teamKills'], kde=True, ax=axes[3,0])
sns.distplot(train['weaponsAcquired'], kde=True, ax=axes[3,1])
sns.distplot(train['winPoints'], kde=True, ax=axes[3,2])
sns.distplot(train['revives'], kde=True, ax=axes[3,3])

corr=train.corr()   
corr

a=corr[(corr>0.4) | (corr<-0.4)]       
a
f, ax = plt.subplots(figsize = (30,18))

sns.heatmap(data= corr, 
                        mask=np.zeros_like(corr, dtype=np.bool), 
                        cmap=sns.diverging_palette(220, 10, as_cmap=True),
                        square=True, ax=ax,annot=True)
train.plot('winPlacePerc','walkDistance', kind='scatter')
train.plot('winPlacePerc','rideDistance', kind='scatter')
train.plot('winPlacePerc','killPlace', kind='scatter')
train.plot('winPlacePerc','boosts', kind='scatter')
train.plot('winPlacePerc','weaponsAcquired', kind='scatter')
train.plot('winPlacePerc','damageDealt', kind='scatter')
train.plot('winPlacePerc','damageDealt', kind='box')
train.groupby('damageDealt')['winPlacePerc'].mean().plot(kind='barh')
train['Total_Distance']=train.walkDistance+train.rideDistance+train.swimDistance
possible_hackers=train[(train.walkDistance+train.swimDistance+train.rideDistance==0) & (train.winPlacePerc==1)]
possible_hackers.shape
possible_hackers.head()
possible_hackers.weaponsAcquired.unique().mean()
possible_hackers.heals.unique().mean()
possible_hackers.kills.unique().mean()
train.plot('winPlacePerc','Total_Distance', kind='scatter')
scaler=MinMaxScaler()
train_std=pd.DataFrame(scaler.fit_transform(train.iloc[:,0:25]))
train_std.head()
pca=decomposition.PCA(0.95)
train_pca=pd.DataFrame(pca.fit_transform(train_std))
pca.explained_variance_ratio_
pca_df=pd.DataFrame(pca.components_)
pca_df
pca_df.shape
train_pca.head()
final_train_pca=pd.concat([train_pca,train['winPlacePerc']],axis=1)
final_train_pca.head()
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
X=final_train_pca.drop('winPlacePerc', axis=1)    #Dropping the winPlacePerc column
y=final_train_pca['winPlacePerc']
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.4, random_state=0)
regressor= DecisionTreeRegressor()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
df=pd.DataFrame({'Actual': y_test,'Predicted': y_pred})
df
print("Mean Absolute error", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(X_train, y_train)
y_predict=rf.predict(X_test)
df1=pd.DataFrame({'Actual': y_test, 'Pridicted': y_predict})
df1
print('Mean Absolute Error', metrics.mean_absolute_error(y_test, y_predict))
print('Mean Squared Error', metrics.mean_squared_error(y_test, y_predict))
print('Root Mean Squared Error', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

from sklearn.metrics import r2_score
r2_score(y_test, y_predict)
r2_score(y_test, y_pred)
test=pd.read_csv('../input/test.csv')
test.head()
test.isna().sum()
test_std=pd.DataFrame(scaler.fit_transform(test))
test_std.head()
pca=decomposition.PCA(0.95)
test_pca=pd.DataFrame(pca.fit_transform(test_std))
test_pca.head()
y_pred1=rf.predict(test_pca)
y_pred1
final=pd.DataFrame({'Id':test['Id'],'Predicted':y_pred1})
final