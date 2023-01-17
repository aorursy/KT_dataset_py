# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/tourism_dataset.csv")
data.info()
data.head()
data.tail()
x = data['country']
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color = 'white',
                          width = 512,
                          height = 384
                         ).generate(" ".join(x))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()
sns.barplot(x = data.country, y = data.year_2000)
plt.xlabel('countries')
plt.xticks(rotation = 45)
plt.ylabel('visitors(millions)')
plt.title('Visitors of the countries in 2000', color = 'blue', fontsize = 20)
plt.show()
sns.barplot(x = data.country, y = data.year_2016)
plt.xlabel('countries')
plt.xticks(rotation = 45)
plt.ylabel('visitors(millions)')
plt.title('Visitors of the countries in 2016', color = 'blue', fontsize = 20)
plt.show()
labels = data.country
colors = ['grey', 'blue', 'red', 'yellow', 'green', 'brown', 'orange', 'purple', 'cyan', 'pink']
explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sizes = data.year_2016
plt.figure(figsize = (7,7))
plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%')
plt.title("Ratio of Most Visited Countries in 2016", color = 'blue', fontsize = 20)
plt.show()
x = sns.stripplot(x = "country", y = "year_2016", data = data, jitter = True)
plt.xticks(rotation = 45)
plt.show()
data1 = dict(type = 'choropleth', 
           locations = data.country,
           locationmode = 'country names',
           z = data['year_2016'],
           colorbar = {'title':'Visitors (millions)'})
layout = dict(title = 'Most Visited Countries', 
             geo = dict(showframe = False, 
                       projection = {'type': 'natural earth'}))
choromap = go.Figure(data = [data1], layout = layout)
iplot(choromap)
plt.scatter(data.year_2000,data.year_2016) 
plt.xlabel("2000 visitors (millions)")  
plt.ylabel("2016 visitors (millions)")
# linear regression
# sklearn library

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()

x=data.year_2000.values 
y=data.year_2016.values

x=data.year_2000.values.reshape(-1,1)
y=data.year_2016.values.reshape(-1,1)

linear_reg.fit (x,y)
array=np.array([x]).reshape(-1,1) 
plt.scatter(x,y)
y_head=linear_reg.predict(array) 
plt.plot(array,y_head,color="red")  
lr = LinearRegression()

lr.fit(x,y)

#%% predict
y_head = lr.predict(x)

plt.scatter(x,y)
plt.plot(x,y_head,color="red",label ="linear")

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 2)

x_polynomial = polynomial_regression.fit_transform(x)
linear_reg2 = LinearRegression()
linear_reg2.fit(x_polynomial,y)
y_head2 = linear_reg2.predict(x_polynomial)

plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()

polynomial_regression2 = PolynomialFeatures(degree = 4)
x_polynomial = polynomial_regression2.fit_transform(x)

linear_reg3 = LinearRegression()
linear_reg3.fit(x_polynomial,y)

y_head3 = linear_reg3.predict(x_polynomial)

plt.plot(x,y_head3,color= "black",label = "poly")
plt.xlabel("2000 visitors (millions)")
plt.ylabel("2016 visitors (millions)")
plt.legend()
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()   # random sate = 0
tree_reg.fit(x,y)

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)
# %% visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("2000 visitors (millions)")
plt.ylabel("2016 visitors (millions)")
plt.show()
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x,y.ravel())
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

# visualize
plt.scatter(x,y.ravel(),color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("2000 visitors (millions)")
plt.ylabel("2016 visitors (millions)")
plt.show()