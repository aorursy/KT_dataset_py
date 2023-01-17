import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
df=pd.read_csv('../input/data.csv')
df.info()
df.head(10)
df['Subscribers'] = pd.to_numeric(df['Subscribers'], errors='coerce')

df['Video Uploads'] = pd.to_numeric(df['Video Uploads'], errors='coerce')
df.isnull().sum()
df=df.dropna()
output = df.drop_duplicates()

output.groupby('Grade').size()
sns.heatmap(df.corr(),annot=True)

plt.plot()
labels = ['A++', 'A+', 'A', 'A-','B++']

sizes = [10,40,897,941,2722]

#colors

colors = ['#ffdaB9','#66b3ff','#99ff99','#ffcc99','#ff9999']

#explsion

explode = (0.05,0.05,0.05,0.05,0.05)

plt.figure(figsize=(8,8)) 

my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85,explode=explode)

p=plt.gcf()

plt.axis('equal')

p.gca().add_artist(my_circle)

plt.show()
plt.subplots()

sns.regplot(x=df['Subscribers'], y=df["Video views"], fit_reg=True,scatter_kws={"color":"orange"})

plt.show()
sns.regplot(x=df["Video Uploads"], y=df["Video views"], fit_reg=False,scatter_kws={"color":"red"})

plt.show()
sns.regplot(x=df["Video Uploads"], y=df["Subscribers"], fit_reg=False,scatter_kws={"color":"blue"})

plt.show()
sns.lmplot(x='Subscribers', y='Video views', data=df, fit_reg=False, hue='Grade')

plt.show()
df.sort_values(by = ['Subscribers'], ascending = False).head(15).plot.barh(x = 'Channel name', y = 'Subscribers')

plt.show()
df.sort_values(by = ['Video views'], ascending = False).head(15).plot.barh(x = 'Channel name', y = 'Video views')

plt.show()
X = df[['Video Uploads', 'Video views']]

Y = df[['Subscribers']]
X_train, X_test, y_train, y_test =train_test_split(X,Y, test_size = 0.2)
lr=LinearRegression()

lr.fit(X_train.dropna(),y_train.dropna())
pred_train=lr.predict(X_train)

pred_test=lr.predict(X_test)
plt.figure(figsize=(10,8))

plt.scatter(lr.predict(X_train),lr.predict(X_train)-y_train,c='b',s=40,alpha=0.5)

plt.scatter(lr.predict(X_test),lr.predict(X_test)-y_test,c='g',s=40)

plt.hlines(y=0,xmin=0,xmax=100000000)

plt.title('Residual Plots using Training(blue) and Test(green) data')

plt.ylabel('Residuals')
plt.scatter(y_test,pred_test, color = 'green')

plt.xlabel('Y in test set')

plt.ylabel('Predicted Y')