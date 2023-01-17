import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import norm, skew, kurtosis
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import pylab as p
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data=pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
test_data=pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')
test_data.head()
train_data.head()
test_data.columns
train_data.columns
print('test_data : {}, train_data :{}'.format(test_data.shape, train_data.shape))
test_data.info()
train_data.info()
test_data.describe()
train_data.describe()
train_data['price_range'].value_counts()
size = train_data['price_range'].value_counts()
plt.figure(figsize=(8,6))
plt.style.use('seaborn-paper')
plt.pie(size, labels=[3,2,1,0],shadow=True, autopct='%1.1f%%', colors=['cyan','darkred', 'darkgreen', 'darkblue'])
plt.title('A pie chart showing price range distributions among the data', fontsize=14, color='purple')
plt.show()
print('Kurtosis : {}'.format(kurtosis(train_data['price_range'])))
print('Skew : {}'.format(skew(train_data['price_range'])))
size = train_data['three_g'].value_counts()
plt.figure(figsize=(8,6))
plt.style.use('seaborn-paper')
plt.pie(size, labels=[0,1],shadow=True, autopct='%1.1f%%', colors=['y', 'white'])
plt.title('A pie chart showing three_g distributions among the data', fontsize=14, color='purple')
plt.show()
size = train_data['four_g'].value_counts()
plt.figure(figsize=(8,6))
plt.style.use('seaborn-paper')
plt.pie(size, labels=[0,1],shadow=True, autopct='%1.1f%%', colors=['cyan', 'green'])
plt.title('A pie chart showing four_g distributions among the data', fontsize=14)
plt.show()
plt.figure(figsize=(18,18))
correlation = train_data.corr()
sns.heatmap(correlation,square=True,annot=True,vmax=0.9, color='b')
sns.catplot(x='price_range', y='ram', kind='swarm', data=train_data)
sns.catplot(x='price_range', y='ram', kind='box', data=train_data)
plt.figure(figsize=(10,8))
sns.catplot(x='price_range', y='mobile_wt', kind='box', data=train_data)
plt.title('Distributions between price_range with respect to mobile_wt', color='darkgreen', fontsize=13)
plt.figure(figsize=(10,8))
sns.catplot(x='price_range', y='px_height', kind='box', data=train_data)
plt.title('Distributions between price_range with respect to px_height', color='darkgreen', fontsize=13)
plt.figure(figsize=(10,8))
sns.catplot(x='price_range', y='px_width', kind='box', data=train_data)
plt.title('Distributions between price_range with respect to px_width', color='green', fontsize=13)
plt.figure(figsize=(10,8))
sns.catplot(x='price_range', y='battery_power', kind='box',hue='blue', data=train_data)
plt.title('Distributions between price_range and bluetooth with respect to battery_power', color='darkred', fontsize=13)
#Plotly to try and see some interactive graphs
import plotly.offline as pyo
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
import cufflinks as cf
cf.go_offline()
pyo.init_notebook_mode()
print(__version__)
train_data.iplot(kind='scatter', x='sc_w', y='sc_h', mode='markers', colors='black',size=10)
train_data.iplot(kind='scatter', x='px_width', y='px_height', mode='markers', size=8)
train_data['fc'].iplot(kind='hist', bins=40, xTitle='Mega pixels', yTitle='Frequency', colors='cyan')
train_data['pc'].iplot(kind='hist', bins=40, xTitle='Mega pixels',yTitle='Frequency', colors='darkred')
sns.pointplot(train_data['talk_time'], y=train_data['price_range'], data=train_data)
plt.title('Point plot displaying how price ranges with talk_time', fontsize=13)
test_data['px_area'] = test_data['px_height'] * test_data['px_width']
test_data['phone_area'] = test_data['sc_w'] * test_data['sc_h']
test_data.drop(['px_width', 'px_height', 'sc_w', 'sc_h', 'id'], axis=1, inplace=True)
train_data['px_area'] = train_data['px_height'] * train_data['px_width']
train_data['phone_area'] = train_data['sc_w'] * train_data['sc_h']
train_data.drop(['px_width', 'px_height', 'sc_w', 'sc_h'], axis=1, inplace=True)
train_data
from sklearn.model_selection import train_test_split
X = train_data.drop('price_range', axis=1)
y = train_data['price_range']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=20)
from sklearn.ensemble import RandomForestClassifier
rnd = RandomForestClassifier(max_depth=8, n_estimators=700, random_state=0, n_jobs=-1)
rnd.fit(X_train, y_train)
rnd.score(X_train, y_train)
rnd.score(X_test, y_test)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.001, max_iter=1000, random_state=20)
lasso.fit(X_train, y_train)
lasso.score(X_train, y_train)
lasso.score(X_test, y_test)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1, max_iter=1000, random_state=20)
ridge.fit(X_train, y_train)
ridge.score(X_train, y_train)
ridge.score(X_test, y_test)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.score(X_train, y_train)
lm.score(X_test, y_test)
from lightgbm import LGBMClassifier
reg = LGBMClassifier(learning_rate=0.1, n_estimators=700,  max_depth=8, random_state=0, n_jobs=-1)
reg.fit(X_train, y_train)
reg.score(X_train, y_train)
reg.score(X_test, y_test)
from sklearn.linear_model import LogisticRegression
mod = LogisticRegression(C=0.1, random_state=0, n_jobs=-1, max_iter=100)
mod.fit(X_train, y_train)
mod.score(X_train, y_train)
mod.score(X_test, y_test)
from sklearn.tree import DecisionTreeClassifier
trees = DecisionTreeClassifier(random_state=20, max_depth=5, criterion='entropy')
model = trees.fit(X_train, y_train)
model
trees.score(X_train, y_train)
trees.score(X_test, y_test)
from sklearn.model_selection import GridSearchCV
params = {'max_depth':[5, 1], 'criterion':['entropy', 'gini'], 'random_state':[20,5]}
gridz = GridSearchCV(DecisionTreeClassifier(), param_grid=params, refit=True,verbose=3)
gridz.fit(X_train, y_train)
gridz.best_params_
gridz.best_estimator_
final_results = trees.predict(test_data)
final_results
test_data = pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')
test_data['id']
final = pd.DataFrame({'id':test_data.id, 'price_range': final_results})
final
  