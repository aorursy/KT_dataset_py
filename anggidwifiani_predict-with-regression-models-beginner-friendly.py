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
data = pd.read_csv("../input/videogamesales/vgsales.csv")

data.head()
data.info()
data.describe()
data.shape
data.isnull().sum()
data.dropna().shape
#Visualization data to get information

#Import library to plotting data



import matplotlib.pyplot as plt

import seaborn as sns
#Plot 10 publisher that most contribute to global sales. The best one is Electronic Art publisher.

contri=data.groupby('Publisher')['Global_Sales'].count().sort_values(ascending=False).head(10)

my_range=range(0,len(contri.index))

plt.hlines(y=contri.index, xmin=0, xmax=contri.values, color='skyblue')

plt.plot(contri.values, my_range, "o")

plt.xlabel('Global Sales')

plt.title('Top 10 Publisher Contribute Global Sales', pad=20, fontsize=15)

plt.show()
#Plot top 10 games that most selled in global.

games=data.groupby('Name')['Global_Sales'].count().sort_values(ascending = False).head(10)

sns.barplot(x=games.values,y=games.index, palette='YlGn_r')

plt.xlabel('Global Sales')

plt.ylabel('Game Names')

plt.title('Top 10 Games Contribute Global Sales', pad=20, fontsize=15)

plt.show()
#Plot genre games selled in global

import matplotlib

games=data.groupby('Genre')['Global_Sales'].count().head(10)

cmap = matplotlib.cm.PuBu

mini=min(games)

maxi=max(games)

norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)

colors = [cmap(norm(value)) for value in games]

plt.figure(figsize=(9,7))

plt.pie(games, labels=games.index, colors = colors, autopct='%.1f%%')

plt.legend(loc = "upper right", bbox_to_anchor=(0.60, 0.5, 0.60, 0.5))

plt.title('Top 10 Game Genres Contribute Global Sales', pad=20, fontsize=15)

plt.axis('off')

plt.show()
#Find correlation between feature. In this, we get NA_Sales, EU_Sales, JP_Sales, and Other_Sales get much correlation with global sales



cor = data.corr()

cor = pd.DataFrame(cor)
sns.heatmap(cor, vmin=-0.5, cmap="YlGnBu")
#Since region sales high correlate to global sales, we plot it to know where highest region correlation.

data[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum().plot(x="Global_Sales", kind="barh", colormap='gnuplot')

plt.xlabel('Global Sales', fontsize=10)

plt.ylabel('Region Sales', fontsize=10)

plt.title('Region of Global Sales', pad=20, fontsize=15)

plt.show()
#Make cluster data

X = data.iloc[:, 7:11].values

y = data.iloc[:, -1].values
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Hierarchical Clustering Model

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr = lr.fit(X_train, y_train)

y_predlr = lr.predict(X_test)
print("Training Accuracy :", lr.score(X_train, y_train))

print("Testing Accuracy :", lr.score(X_test, y_test))

print(r2_score(y_test,y_predlr))
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X)

lr_2 = LinearRegression()

lr_2.fit(X_poly, y)
print("Accuracy :", lr_2.score(X_poly, y))
from sklearn.svm import SVR

svr = SVR(kernel = 'rbf')

svr.fit(X_train, y_train)

y_predsvr = svr.predict(X_test)
print("Training Accuracy :", svr.score(X_train, y_train))

print("Testing Accuracy :", svr.score(X_test, y_test))

print(r2_score(y_test,y_predsvr))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 0)

dtr.fit(X_train, y_train)

y_preddtr = dtr.predict(X_test)
print("Training Accuracy :", dtr.score(X_train, y_train))

print("Testing Accuracy :", dtr.score(X_test, y_test))

print(r2_score(y_test,y_preddtr))
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 10, random_state = 0)

rfr.fit(X_train, y_train)

y_predrfr = rfr.predict(X_test)
print("Training Accuracy :", rfr.score(X_train, y_train))

print("Testing Accuracy :", rfr.score(X_test, y_test))

print(r2_score(y_test,y_predrfr))
from sklearn.linear_model import Ridge

r = Ridge()

r = r.fit(X_train,y_train)

y_predr = r.predict(X_test)
print("Training Accuracy :", r.score(X_train, y_train))

print("Testing Accuracy :", r.score(X_test, y_test))

print(r2_score(y_test,y_predr))
from sklearn.linear_model import Lasso

l = Lasso()

l = l.fit(X_train,y_train)

y_predl = l.predict(X_test)
print("Training Accuracy :", l.score(X_train, y_train))

print("Testing Accuracy :", l.score(X_test, y_test))

print(r2_score(y_test,y_predl))
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

enet.fit(X_train,y_train)

y_predEnet = enet.predict(X_test)
print("Training Accuracy :", enet.score(X_train, y_train))

print("Testing Accuracy :", enet.score(X_test, y_test))

print(r2_score(y_test,y_predEnet))