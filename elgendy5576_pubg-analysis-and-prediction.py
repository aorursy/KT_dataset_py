# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt 

import seaborn as sns 
df =pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/train_V2.csv")
df.head()
df.describe()
df.isnull().sum()
df.dropna(inplace=True)

df.isnull().sum()[-1] #-1 t to access the last element of the returend pandas series "win place percent" 
df.columns
df['hitnokill']=df['DBNOs']+df['assists']

df['total-distnace']=df['swimDistance']+df['walkDistance']+df['rideDistance']

df['recovery']=df['boosts']+df['heals']
corr=df.corr()

plt.subplots(figsize=(20, 11))

sns.heatmap(corr,annot =True)
df.drop(['recovery','hitnokill'],axis=1,inplace=True)
df.drop(['Id','groupId','matchId','rankPoints','roadKills','vehicleDestroys'],axis=1,inplace=True)
df.info()
plt.figure(figsize=(22,10))

sns.distplot(df.assists,bins=30,kde=False)

plt.show()
df.assists.value_counts()
plt.subplots(figsize=(15,9))

sns.scatterplot(df['winPlacePerc'],df['winPoints'])
plt.subplots(figsize=(15,9))

sns.scatterplot(df['winPlacePerc'],df['walkDistance'])
df.matchType.value_counts().plot(kind='bar',figsize=(10,5))
X=df.drop(['winPlacePerc'],axis=1)

y=df['winPlacePerc']
from sklearn.feature_selection import f_regression

from sklearn.feature_selection import SelectKBest

X.drop('matchType',axis =1 ,inplace =True)
best_feature = SelectKBest(score_func=f_regression,k='all')

fit = best_feature.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score']

featureScores = featureScores.sort_values(by='Score',ascending=False).reset_index(drop=True)



featureScores
X.columns
X= X[featureScores.Feature[:15].values]
from sklearn.preprocessing import StandardScaler

cols = X.columns

scaler = StandardScaler()

X=scaler.fit_transform(X)

X=pd.DataFrame(X,columns=cols)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,random_state=42)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

 

reg = LinearRegression()

model = cross_val_score(reg,X_train,y_train,cv=3,scoring='neg_mean_squared_error')

-model
test=pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/test_V2.csv")
testcp=test.copy()
testcp['total-distnace']=test['swimDistance']+testcp['walkDistance']+testcp['rideDistance']

testcp.columns
X.columns
testcp=testcp[X.columns]
testcp=scaler.fit_transform(testcp)

testcp=pd.DataFrame(testcp,columns=cols)
reg.fit(X_train,y_train)
predictions =reg.predict(testcp)
sub=pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/sample_submission_V2.csv")

sub.head()
sub.winPlacePerc=predictions
sub.head()
sub.to_csv('submission.csv',index=False)
