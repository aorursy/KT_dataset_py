# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/world-happiness/2019.csv')
data.head()
data.info()
data.describe()
data_mean = sum(data["Score"])/len(data["Score"])
data["happy"] = ["yes"if i>data_mean else "no" for i in data["Score"]] 
colors = ["green" if i=="yes" else "red" for i in data["happy"]]
colors
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'Country or region '],
                           c = colors,
                           figsize = [20,20],
                           diagonal='hist',
                           alpha=0.5,
                           s = 200,
                           marker = '.',
                           edgecolor= "black")

plt.show()
data.columns
# Visualize 
data1 = data.loc[:, data.columns != 'Country or region ']
x = np.array(data1.loc[:,'Score']).reshape(-1,1)
y = np.array(data1.loc[:,'GDP per capita']).reshape(-1,1)
# Scatter
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('Score')
plt.ylabel('GDP')
plt.show()
# LinearRegression

# Score vs GDP

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
prediction = np.linspace(min(x),max(x)).reshape(-1,1)
lr.fit(x,y)
predicted = lr.predict(prediction)
r2 = lr.score(x,y)
print("r^2 score: ",r2)
plt.plot(prediction, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('Score')
plt.ylabel('GDP')
plt.show()
# Score vs Social support 

x = np.array(data1.loc[:,'Score']).reshape(-1,1)
y = np.array(data1.loc[:,'Social support']).reshape(-1,1)

prediction = np.linspace(min(x),max(x)).reshape(-1,1)
lr.fit(x,y)
predicted = lr.predict(prediction)
r2 = lr.score(x,y)
print("r^2 score: ",r2)
plt.plot(prediction, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('Score')
plt.ylabel('Social support')
plt.show()
# Score vs Healthy life expectancy 

x = np.array(data1.loc[:,'Score']).reshape(-1,1)
y = np.array(data1.loc[:,'Healthy life expectancy']).reshape(-1,1)

prediction = np.linspace(min(x),max(x)).reshape(-1,1)
lr.fit(x,y)
predicted = lr.predict(prediction)
r2 = lr.score(x,y)
print("r^2 score: ",r2)
plt.plot(prediction, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('Score')
plt.ylabel('Healthy life expectancy')
plt.show()
# Score vs Overall rank

x = np.array(data1.loc[:,'Score']).reshape(-1,1)
y = np.array(data1.loc[:,'Overall rank']).reshape(-1,1)

prediction = np.linspace(min(x),max(x)).reshape(-1,1)
lr.fit(x,y)
predicted = lr.predict(prediction)
r2 = lr.score(x,y)
print("r^2 score: ",r2)
plt.plot(prediction, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('Score')
plt.ylabel('Overall rank')
plt.show()
data[data['Country or region']=='Cameroon']
x = np.array(data1.loc[:,'GDP per capita']).reshape(-1,1)
y = np.array(data1.loc[:,'Score']).reshape(-1,1)
lr.fit(x,y)
predicted = lr.predict([[0.549]])
print("Cameroon Predicted Score: ", predicted )

x2 = np.array(data1.loc[:,'Score']).reshape(-1,1)
y2 = np.array(data1.loc[:,'Overall rank']).reshape(-1,1)

lr.fit(x2,y2)
predicted2 = lr.predict([[4.61710843]])
print("Cameroon Predicted Overall Rank: ", predicted2 )

data.head()
# Multiple LR

# Score vs GDP & Social Support

x = data.iloc[:,[3,4]].values
y = data.Score.values.reshape(-1,1)

lr = LinearRegression()
lr.fit(x,y)

lr.fit(x,y)
predicted = lr.predict([[0.549,0.91]])
print("Cameroon Predicted Score: ", predicted )

x2 = np.array(data1.loc[:,'Score']).reshape(-1,1)
y2 = np.array(data1.loc[:,'Overall rank']).reshape(-1,1)

lr.fit(x2,y2)
predicted2 = lr.predict([[4.46812163]])
print("Cameroon Predicted Overall Rank: ", predicted2 )
# Score vs GDP & Healthy life expectancy

x = data.iloc[:,[3,5]].values
y = data.Score.values.reshape(-1,1)

lr = LinearRegression()
lr.fit(x,y)

lr.fit(x,y)
predicted = lr.predict([[0.549,0.331]])
print("Cameroon Predicted Score: ", predicted )

x2 = np.array(data1.loc[:,'Score']).reshape(-1,1)
y2 = np.array(data1.loc[:,'Overall rank']).reshape(-1,1)

lr.fit(x2,y2)
predicted2 = lr.predict([[4.23823909]])
print("Cameroon Predicted Overall Rank: ", predicted2 )
data[data['Country or region']=='Cameroon']
# Decision Tree Regression
y = np.array(data1.loc[:,'Score']).reshape(-1,1)
x = np.array(data1.loc[:,'GDP per capita']).reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(x,y)

print("predicted Score: ",dt_reg.predict([[0.549]]))


x = np.array(data1.loc[:,'Score']).reshape(-1,1)
y = np.array(data1.loc[:,'Overall rank']).reshape(-1,1)

dt_reg.fit(x,y)

print("predicted Overall rank: ",dt_reg.predict([[5.044]]))

# Lets look at DTR's R^2 Score 
from sklearn.metrics import r2_score

y = np.array(data1.loc[:,'Score']).reshape(-1,1)
x = np.array(data1.loc[:,'GDP per capita']).reshape(-1,1)
dt_reg.fit(x,y)
y_head = dt_reg.predict(x)
print("r_square score: ", r2_score(y,y_head))

y = np.array(data1.loc[:,'Score']).reshape(-1,1)
x = np.array(data1.loc[:,'GDP per capita']).reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state=42  )
rf.fit(x,y)
print("predicted Score: ",rf.predict([[0.549]]))


x = np.array(data1.loc[:,'Score']).reshape(-1,1)
y = np.array(data1.loc[:,'Overall rank']).reshape(-1,1)

rf.fit(x,y)
print("predicted Score: ",rf.predict([[5.021]]))

# Lets look at DTR's R^2 Score 
from sklearn.metrics import r2_score

y = np.array(data1.loc[:,'Score']).reshape(-1,1)
x = np.array(data1.loc[:,'GDP per capita']).reshape(-1,1)
dt_reg.fit(x,y)
y_head = dt_reg.predict(x)
print("r_square score: ", r2_score(y,y_head))