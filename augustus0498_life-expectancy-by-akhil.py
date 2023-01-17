import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
import os

print(os.listdir("../input"))
df = pd.read_csv('../input/led.csv')
df.head()
df.info()
sns.countplot(df['Status'])
df = df.dropna()
plt.hist(df['Lifeexpectancy'])
sns.barplot(x = df['Status'],y=df['Lifeexpectancy'])
sns.regplot(x=df['AdultMortality'],y=df['Lifeexpectancy'])
sns.regplot(x=df['infantdeaths'],y=df['Lifeexpectancy'])
sns.regplot(x=df['Alcohol'],y=df['Lifeexpectancy'])
sns.regplot(x=df['percentageexpenditure'],y=df['Lifeexpectancy'])
sns.regplot(x=df['HepatitisB'],y=df['Lifeexpectancy'])
sns.regplot(x=df['Measles'],y=df['Lifeexpectancy'])
sns.regplot(x=df['BMI'],y=df['Lifeexpectancy'])
sns.regplot(x=df['under-fivedeaths'],y=df['Lifeexpectancy'])
sns.regplot(x=df['Polio'],y=df['Lifeexpectancy'])
sns.regplot(x=df['Totalexpenditure'],y=df['Lifeexpectancy'])
sns.regplot(x=df['Diphtheria'],y=df['Lifeexpectancy'])
sns.regplot(x=df['HIV/AIDS'],y=df['Lifeexpectancy'])
sns.regplot(x=df['GDP'],y=df['Lifeexpectancy'])
sns.regplot(x=df['Population'],y=df['Lifeexpectancy'])
sns.regplot(data = df[df['Population']<1000000000],x='Population',y='Lifeexpectancy')
sns.regplot(x=df['thinness1-19years'],y=df['Lifeexpectancy'])
sns.regplot(x=df['thinness5-9years'],y=df['Lifeexpectancy'])
sns.regplot(x=df['Incomecompositionofresources'],y=df['Lifeexpectancy'])
sns.regplot(x=df['Schooling'],y=df['Lifeexpectancy'])
plt.figure(figsize=(15,5))

plt.subplot(121)

sns.regplot(x=df['Totalexpenditure'],y=df['Lifeexpectancy'])

plt.subplot(122)

sns.regplot(x=df['percentageexpenditure'],y=df['Lifeexpectancy'])
df.replace(['Developed','Developing'],[1,0],inplace=True)
sns.pairplot(df,hue='Status')
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['Country'] = encoder.fit_transform(df['Country'])
plt.figure(figsize=(15,15))

sns.heatmap(df.corr(),annot=True)
# schooling,Status,Incomecompositionofresources,thinness5-9years,thinness1-19years,GDP,HIV/AIDS,Diptheria,Totalexpenditure,Polio,under-fivedaths,BMI,percentageexpenditure,Alchohol,AdultMortality,Country,infantdeaths
plt.figure(figsize=(10,10))

sns.heatmap(df[['Lifeexpectancy','Status','Schooling','thinness1-19years','GDP','HIV/AIDS','Diphtheria','Totalexpenditure','AdultMortality','Country','infantdeaths']].corr(),annot=True)
features = ['Schooling','thinness1-19years','GDP','HIV/AIDS','Diphtheria','Totalexpenditure','AdultMortality','Country','infantdeaths']
df['Year'].unique()
trdf = df[df['Year']<2014]

tedf = df[df['Year']>=2014]
X_train = trdf[features]

X_test = tedf[features]

y_train = trdf['Lifeexpectancy']

y_test = tedf['Lifeexpectancy']
from sklearn.linear_model import LinearRegression

from sklearn import metrics
lin = LinearRegression()

lin.fit(X_train,y_train)

y_pred = lin.predict(X_test)

np.sqrt(metrics.mean_squared_error(y_pred,y_test))
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,max_depth=4)

rf.fit(X_train,y_train)

y_predr = rf.predict(X_test)

np.sqrt(metrics.mean_squared_error(y_predr,y_test))