import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



!pip install ngboost
data = pd.read_csv('../../kaggle/input/nfl-big-data-bowl-2020/train.csv')



data['YardLine100'] = data['YardLine'] #Normalize yardline to 1-99

data.loc[data['FieldPosition'] == data['PossessionTeam'], 'YardLine100'] = 50+  (50-data['YardLine'])

data['Touchdown'] =  data.Yards == data.YardLine100



temp_data = data[data.Touchdown == 0][['YardLine100','Down','Distance','DefendersInTheBox','Yards']].dropna().drop_duplicates()

X = np.array(temp_data[['YardLine100','Down','Distance','DefendersInTheBox']])

y = np.array(temp_data['Yards'])*1.0
from scipy.stats import lognorm

shape, loc, scale = lognorm.fit(y)



fig, ax = plt.subplots()

x_axis = np.linspace(-10,50,100)

ax.hist(y,bins=np.arange(-10, y.max() + 1.5) -0.5,density=True)

ax.plot(x_axis,lognorm.pdf(x_axis, shape, loc, scale))

plt.title('Histogram of rushing yards gain excluding touchdown')

plt.xlabel('rushing yard gain')
from ngboost.ngboost import NGBoost

from ngboost.learners import default_tree_learner

from ngboost.scores import CRPS, MLE

from ngboost.distns import LogNormal, Normal
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
ngb =  NGBoost(n_estimators=100, learning_rate=0.1,

              Dist=LogNormal,

              Base=default_tree_learner,

              natural_gradient=False,

              minibatch_frac=1.0,

              Score=CRPS())

ngb.fit(X_train,y_train-min(y_train)+0.001)
y_preds = ngb.predict(X_test)

y_dists = ngb.pred_dist(X_test)
print('mean of lognormal scale = %f'% y_dists.scale.mean())

print('standard deviation of lognormal scale = %f'%y_dists.scale.std())
from ngboost.evaluation import *

pctles, observed, slope, intercept = calibration_regression(y_dists, y_test-min(y_test)+0.001)

plt.subplot(1, 2, 1)

plot_pit_histogram(pctles, observed, label="CRPS", linestyle = "-")