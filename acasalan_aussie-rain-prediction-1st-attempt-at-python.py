import pandas as pd

import numpy as np

import matplotlib as plt

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split





import os

print(os.listdir("../input"))
df=pd.read_csv('../input/weatherAUS.csv')
df.shape
df.duplicated(subset='Date', keep=False).sum()
df.dtypes
#First 10 rows

df.head(10)
start=df['Date'].min()

start
end=df['Date'].max()

end
df.describe()
df.isnull().sum()
df=df.dropna(axis=0,how='any')
df.shape
#Number of missing values by predictor

df.isnull().sum()
ct1=pd.crosstab(df.Location, df.RainTomorrow,normalize='index')

ct1
#sorting the percentage of rainy next days by location

ct1.iloc[:,1].sort_values(ascending=False)
ct2=pd.crosstab(df.WindGustDir,df.RainTomorrow,rownames=['Wind Gust Direction'],colnames=['Rain Tomorrow'], normalize='index')

ct2.plot.bar(stacked=True)
#sorting the percentage of rainy next days by Wind Gust

ct2.iloc[:,1].sort_values(ascending=False)
ct3=pd.crosstab(df.RainToday,df.RainTomorrow,rownames=['Rain Today'],colnames=['Rain Tomorrow'], normalize='index')

ct3.plot.bar(stacked=True)
ct3
avg1 = df.groupby(['RainTomorrow'])['MinTemp'].mean()

avg1.plot.bar()

avg1
avg2 = df.groupby(['RainTomorrow'])['MaxTemp'].mean()

avg2.plot.bar()

avg2
avg3 = df.groupby(['RainTomorrow'])['Rainfall'].mean()

avg3.plot.bar()
avg3
avg4 = df.groupby(['RainTomorrow'])['Evaporation'].mean()

avg4.plot.bar()
avg4
avg5 = df.groupby(['RainTomorrow'])['Sunshine'].mean()

avg5.plot.bar()
avg5
avg6 = df.groupby(['RainTomorrow'])['WindGustSpeed'].mean()

avg6.plot.bar()
avg6
avg7 = df.groupby(['RainTomorrow'])['WindSpeed9am'].mean()

avg7.plot.bar()
avg7
avg8 = df.groupby(['RainTomorrow'])['WindSpeed3pm'].mean()

avg8.plot.bar()
avg8
avg9 = df.groupby(['RainTomorrow'])['Humidity9am'].mean()

avg9.plot.bar()
avg9
avg10 = df.groupby(['RainTomorrow'])['Humidity3pm'].mean()

avg10.plot.bar()
avg10
avg11 = df.groupby(['RainTomorrow'])['Pressure9am'].mean()

avg11.plot.bar()
avg11
avg12 = df.groupby(['RainTomorrow'])['Pressure3pm'].mean()

avg12.plot.bar()
avg12
avg13 = df.groupby(['RainTomorrow'])['Cloud9am'].mean()

avg13.plot.bar()
avg13
avg14 = df.groupby(['RainTomorrow'])['Cloud3pm'].mean()

avg14.plot.bar()
avg14
avg15 = df.groupby(['RainTomorrow'])['Temp9am'].mean()

avg15.plot.bar()
avg15
avg16 = df.groupby(['RainTomorrow'])['Temp3pm'].mean()

avg16.plot.bar()
avg16
#Define target and cols (preidctor variables without Risk_MM)

    #I try to do one-hot encoding for the categorical predictors

    #Damn, I miss R. It was so straightforward to deal with categorical predictors there.

cols=['Location','MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 

      'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 

      'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',

      'Cloud3pm','Temp9am','Temp3pm','RainToday'] 

X=df[cols]

y=df['RainTomorrow']
#small x for hot encoded predictors.

x=pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()

a=logreg.fit(X_train, y_train)
y_pred=a.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
logisticRegr = LogisticRegression()
m1=logisticRegr.fit(X_test, y_test)
#get list of column names/ names of predictor variables

col=list(x)
#table of coefficient signs

coef2= pd.DataFrame(m1.coef_, columns=col)

coef2
import statsmodels.api as sm