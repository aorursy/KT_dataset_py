import pandas as pd

import numpy as np

import matplotlib.gridspec as gridspec

from matplotlib.ticker import MaxNLocator

from scipy import stats

import matplotlib.pyplot as plt

import math

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, KFold

import plotly.express as px

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/car-purchase-data/Car_Purchasing_Data.csv',encoding='latin-1')
data
data.isnull().sum()
def plot_3chart(df, feature):



    # Creating a customized chart. and giving in figsize and everything.

    fig = plt.figure(constrained_layout=True, figsize=(27, 10))

    # creating a grid of 3 cols and 3 rows.

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)



    # Customizing the histogram grid.

    ax1 = fig.add_subplot(grid[0, :2])

    # Set the title.

    ax1.set_title('Histogram')

    # plot the histogram.

    sns.distplot(df.loc[:, feature],

                 hist=True,

                 kde=True,

                 ax=ax1,

                 color='Red')

    ax1.legend(labels=['Normal', 'Actual'])



    # customizing the QQ_plot.

    ax2 = fig.add_subplot(grid[1, :2])

    # Set the title.

    ax2.set_title('Probability Plot')

    # Plotting the QQ_Plot.

    stats.probplot(df.loc[:, feature].fillna(np.mean(df.loc[:, feature])),

                   plot=ax2)

    ax2.get_lines()[0].set_markerfacecolor('Blue')

    ax2.get_lines()[0].set_markersize(12.0)



    # Customizing the Box Plot.

    ax3 = fig.add_subplot(grid[:, 2])

    # Set title.

    ax3.set_title('Box Plot')

    # Plotting the box plot.

    sns.boxplot(df.loc[:, feature], orient='v', ax=ax3, color='Green')

    ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))



    plt.suptitle(f'{feature}', fontsize=24)
plot_3chart(data, 'Age')

plot_3chart(data,'Annual Salary')

plot_3chart(data,'Net Worth')
plot_3chart(data,'Credit Card Debt')
sns.countplot(data['Gender'])

fig = px.treemap(data, path=['Country'], values='Annual Salary',

                  color='Net Worth', hover_data=['Country'],

                  color_continuous_scale='dense', title='Countries with different annual salaries ')

fig.show()
lable = LabelEncoder()

data.Country = lable.fit_transform(data.Country)
X = data.drop(["Customer Name", 'Country',"Customer e-mail", "Car Purchase Amount",'Credit Card Debt'], axis=1)

y = data["Car Purchase Amount"]
X.head()
sns.set(font_scale=1.1)

correlation_train = data.corr()

mask = np.triu(correlation_train.corr())

plt.figure(figsize=(18, 15))

sns.heatmap(correlation_train,

            annot=True,

            fmt='.1f',

            cmap='coolwarm',

            square=True,

            mask=mask,

            linewidths=1)



plt.show()
y1 = y

y1=y1.values.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler=MinMaxScaler()

#scaler = StandardScaler()

y1 = scaler.fit_transform(y1)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15)
X_train


scores = []

n = 100

model1 = RandomForestRegressor(n_estimators = n)

model1.fit(X_train, y_train)

scores.append(model1.score(X_test, y_test))
y_pred1 = model1.predict(X_test)

RFerror =  mean_absolute_error(y_test, y_pred1)
from sklearn import datasets, ensemble

from sklearn.inspection import permutation_importance

params = {'n_estimators': 1000,

          'max_depth': 4,

          'min_samples_split': 5,

          'learning_rate': 0.01,

          'loss': 'ls'}

reg = ensemble.GradientBoostingRegressor(**params)

reg.fit(X_train, y_train)



GBRerror = mean_absolute_error(y_test, reg.predict(X_test))

regpred = reg.staged_predict(X_test)

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(reg.staged_predict(X_test)):

    test_score[i] = reg.loss_(y_test, y_pred)



fig = plt.figure(figsize=(18, 10))

plt.subplot(1, 1, 1)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')

fig.tight_layout()

plt.show()
from xgboost import XGBRegressor

xgb=XGBRegressor()

from sklearn.model_selection import cross_val_score

cv = 10

performance=cross_val_score(xgb,X,y,cv=cv,scoring="neg_mean_absolute_error",n_jobs=-1)

mae=-performance

xgb.fit(X,y)



y_pred3=xgb.predict(X_test)

print(mae)
XGBerror = mae
print("Mean Absolute Errors by: Random Forest =",RFerror)

print("Gradient Boost = ",GBRerror)

print("XGB regressor = ",XGBerror.mean())
X.shape

X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y1,test_size=0.15)
import tensorflow.keras 

from keras.models import Sequential 

from keras.layers import Dense 



model=Sequential()

model.add(Dense(80,input_dim=4,activation='relu'))

model.add(Dense(40,activation='relu'))

model.add(Dense(1,activation='linear'))

model.compile(optimizer='adam',loss='mean_squared_error')



model.summary()
epochs_hist=model.fit(X_train,y_train,epochs=200,batch_size=50,verbose=1,validation_split=0.2)

y_predict=model.predict(X_test)

y_predict.shape
mae = mean_absolute_error(y_test,y_predict)

mse = mean_squared_error(y_test,y_predict)

print(f'MAE = {mae}')

print(f'RMSE = {mse}')
ANNpredictions = pd.DataFrame(y_predict)

RFpredictions = pd.DataFrame(y_pred1)

XGBpredictions = pd.DataFrame(y_pred3)

GBRprediction = pd.DataFrame(regpred)

Xdata = pd.DataFrame(X)

ydata = pd.DataFrame(y)
Xdata.to_csv('Xdata.csv', index=False)

ydata.to_csv('ydata.csv', index=False)

ANNpredictions.to_csv('Predictedcarprices', index = False)