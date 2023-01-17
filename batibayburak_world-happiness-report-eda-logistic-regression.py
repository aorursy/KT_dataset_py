import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import os

%matplotlib inline 

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/world-happiness/2017.csv", delimiter=',')

df.head()
df.info()
df.describe()
print(df.head())
df[['Happiness.Score', 'Whisker.high', 'Whisker.low', 'Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.',

      'Freedom', 'Generosity', 'Trust..Government.Corruption.' , 'Dystopia.Residual']].hist(figsize=(18,12), bins=50, grid=False);
sns.jointplot(x='Happiness.Score',y='Freedom',data=df,kind='scatter');
sns.pairplot(df);
df.corr()
plt.subplots(figsize=(10,8))

sns.heatmap(df.corr());
plt.subplots(figsize=(10,8))

sns.heatmap(df.corr(),cmap='coolwarm',annot=True);
plt.style.use('dark_background')

df[['Happiness.Score', 'Whisker.high', 'Whisker.low', 'Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.',

      'Freedom', 'Generosity', 'Trust..Government.Corruption.' , 'Dystopia.Residual']].hist(figsize=(20, 15), bins=50, grid=False);
plt.style.use('ggplot')

df[['Happiness.Score', 'Whisker.high', 'Whisker.low', 'Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.',

      'Freedom', 'Generosity', 'Trust..Government.Corruption.' , 'Dystopia.Residual']].hist(figsize=(20, 15), bins=50, grid=False);
plt.style.use('fivethirtyeight')

df.plot.area(alpha=0.4);
cat_feats = ['Country']



final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)

final_data.info
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df.drop('Country',axis=1), 

                                                    df['Country'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

             

             
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report



print(classification_report(y_test,predictions))