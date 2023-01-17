# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/world-happiness/2019.csv")
df.head()
df.isna().sum()
df.info()
df.describe().T
df.shape
y = df['Score'].values.reshape(-1, 1)
X = df['GDP per capita'].values.reshape(-1, 1)
y
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:",X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape, "y_test:", y_test.shape)
linreg = LinearRegression() # kurucu metodu çağırıyoruz
linreg.fit(X_train, y_train) # y_train üzerinden X_train'i öğretiyoruz
y_pred = linreg.predict(X_test)
# Daha sonra çıkan denkleme X_test'i koyup, sonuçlara y_pred dedik. 
r2_score(y_test, y_pred)
rms = np.sqrt(mean_squared_error(y_test, y_pred))
rms
r2_score(X_test, y_test) # NOT: BARAN K.
sns.regplot(X_train, y_train)
plt.show()
sns.regplot(y_test, y_pred)
plt.show()
linreg.intercept_
linreg.coef_


X = df.iloc[:, 3]
Y = df.iloc[:, 2]
plt.scatter(X, Y)
plt.show()

model = LinearRegression()
model.fit(X_train, y_train)

plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()

















y = df['Score'].values.reshape(-1, 1)
X = df['GDP per capita'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:",X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape, "y_test:", y_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(prediction)
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, model.predict(X_train), color = 'red')

plt.xlabel('GDP per capita')
plt.ylabel('Score')
plt.show()

plt.scatter(X, y, color = 'blue')
plt.plot(X_train, model.predict(X_train), color = 'red')

plt.xlabel('GDP per capita')
plt.ylabel('Score')
plt.show()



X = df.iloc[:, 3]
Y = df.iloc[:, 2]
plt.scatter(X, Y)
plt.show()

plt.rcParams['figure.figsize'] = (15.0, 15.0)
plt.scatter(X, Y)
plt.show()

# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)

Y_pred = m*X + c

plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()
r2_score(X, Y) 





#df = df.drop(["Country or region", "Overall rank"], axis=1)
y = df[['Score']]
X = df[['GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption']]
y.shape
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("X_train:",X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape, "y_test:", y_test.shape)
multireg = LinearRegression() # kurucu metodu çağırıyoruz
multireg.fit(X_train, y_train) # y_train üzerinden X_train'i öğretiyoruz
y_pred = multireg.predict(X_test)
# Daha sonra çıkan denkleme X_test'i koyup, sonuçlara y_pred dedik. 
r2_score(y_test, y_pred)
rms = np.sqrt(mean_squared_error(y_test, y_pred))
rms
multireg.intercept_
multireg.coef_
conc = np.vstack(y_pred)
y_pred = pd.DataFrame(conc)

y_test=y_test.reset_index(drop=True)

y_pred.columns =['y_pred']
y_test.columns=['y_test']
df2=pd.concat([y_pred, y_test,], axis=1)
df2['Difference']=df2['y_pred']- df2['y_test']
df2['Difference']=df2['Difference']*df2['Difference']
df2['Difference']=np.sqrt((df2['Difference']))
df2['Percentage of Error']=(df2['Difference'])/(0.01*df2['y_test'])

df2.sort_values(by='Percentage of Error',ascending=False)

Z = df[['Score','GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption']]
Z.corr(method ='pearson') 
X = df[['Social support', 'GDP per capita']].values.reshape(-1,2)
Y = df['Score']
x = X[:, 0]
y = X[:, 1]
z = Y
df.sort_values(by='GDP per capita',ascending=False)

x_pred = np.linspace(0, 2, 20)   # range of porosity values
y_pred = np.linspace(0, 2, 20)  # range of brittleness values
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

ols = linear_model.LinearRegression()
model = ols.fit(X, Y)
predicted = model.predict(model_viz)
r2 = model.score(X, Y)
plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('GDP per capita', fontsize=12)
    ax.set_ylabel('Social support', fontsize=12)
    ax.set_zlabel('Score', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')

ax1.text2D(0.2, 0.32, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax1.transAxes, color='grey', alpha=0.5)
ax2.text2D(0.3, 0.42, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax2.transAxes, color='grey', alpha=0.5)
ax3.text2D(0.85, 0.85, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax3.transAxes, color='grey', alpha=0.5)

ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)

fig.suptitle('$R^2 = %.2f$' % r2, fontsize=40)

fig.tight_layout()
X = df[['Generosity', 'Perceptions of corruption']].values.reshape(-1,2)
Y = df['Score']
x = X[:, 0]
y = X[:, 1]
z = Y
x_pred = np.linspace(0, 2, 20)   # range of porosity values
y_pred = np.linspace(0, 2, 20)  # range of brittleness values
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
ols = linear_model.LinearRegression()
model = ols.fit(X, Y)
predicted = model.predict(model_viz)
r2 = model.score(X, Y)
plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Generosity', fontsize=12)
    ax.set_ylabel('Perceptions of corruption', fontsize=12)
    ax.set_zlabel('Score', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')

ax1.text2D(0.2, 0.32, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax1.transAxes, color='grey', alpha=0.5)
ax2.text2D(0.3, 0.42, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax2.transAxes, color='grey', alpha=0.5)
ax3.text2D(0.85, 0.85, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax3.transAxes, color='grey', alpha=0.5)

ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)

fig.suptitle('$R^2 = %.2f$' % r2, fontsize=40)

fig.tight_layout()


