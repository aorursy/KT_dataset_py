import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ggplot import *

from matplotlib import pyplot as plt

import seaborn as sns

sns.set_palette("husl")

from sklearn.model_selection import train_test_split

from sklearn import datasets

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score

%matplotlib inline
df = pd.read_csv("../input/diamonds.csv")
df.head()
df.drop('Unnamed: 0', axis = 1, inplace = True)
df.info()
plt.figure(figsize=[12,12])



# First subplot showing the diamond carat weight distribution

plt.subplot(221)

plt.hist(df['carat'],bins=20,color='lightseagreen')

plt.xlabel('Carat Weight')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond Carat Weight')



# Second subplot showing the diamond depth distribution

plt.subplot(222)

plt.hist(df['depth'],bins=20,color='royalblue')

plt.xlabel('Diamond Depth (%)')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond Depth')



# Third subplot showing the diamond price distribution

plt.subplot(223)

plt.hist(df['price'],bins=20,color='salmon')

plt.xlabel('Price in USD')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond Price')
corr = df.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corr, mask=mask, vmax=1, annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ggplot(aes(x = 'carat', y = 'price', color = 'cut'), data = df) + geom_point(alpha = 0.9)
ggplot(aes(x = 'carat', y = 'price', color = 'color'), data = df) + geom_point(alpha = 0.9)
ggplot(aes(x = 'carat', y = 'price', color = 'clarity'), data = df) + geom_point(alpha = 0.9)
ggplot(aes(x = 'carat', y = 'price', color = 'color', shape = 'cut', group = 'clarity'), data = df) + geom_point(alpha = 1) + facet_wrap( "color") 
ggplot(aes(x = 'carat', y = 'price', color = 'color', shape = 'cut', group = 'cut'), data = df) + geom_point(alpha = 1) + facet_wrap( "cut") 
fig, saxis = plt.subplots(2, 2,figsize=(12,12))



sns.regplot(x = 'carat', y = 'price', data=df, ax = saxis[0,0])

sns.regplot(x = 'x', y = 'price', data=df, ax = saxis[0,1])

sns.regplot(x = 'y', y = 'price', data=df, ax = saxis[1,0])

sns.regplot(x = 'z', y = 'price', data=df, ax = saxis[1,1])
sns.barplot(x = 'cut', y = 'price', order=['Fair','Good','Very Good','Premium','Ideal'], data=df)
sns.barplot(x = 'color', y = 'price', order=['J','I','H','G','F','E','D'], data=df)
sns.barplot(x = 'clarity', y = 'price', order=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], data=df)
# Creating categorical variables for 'cut', 'color', and 'clarity'

df_final = pd.get_dummies(df, columns=["cut", "color", "clarity"])
test_data = df_final.iloc[-round(len(df_final)*.1):].copy()

df_final.drop(df_final.index[-round(len(df_final)*.1):],inplace=True)

test_data.drop('price',1,inplace=True)

print(df_final.shape)

print(test_data.shape)
X = df_final.drop(['price'],1)

y = df_final['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = LinearRegression()

model.fit(X_train,y_train)

model.score(X_test,y_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
model = LinearRegression()

model.fit(X_train,y_train)

model.score(X_test,y_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)
model = LinearRegression()

model.fit(X_train,y_train)

model.score(X_test,y_test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
model = LinearRegression()

model.fit(X_train,y_train)

model.score(X_test,y_test)