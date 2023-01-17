# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import ignore_warnings
from wordcloud import WordCloud
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
airbnb=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb.shape
airbnb.dtypes
airbnb.info()
airbnb.head()
airbnb.duplicated().sum()
airbnb.drop_duplicates(inplace=True)
airbnb.isnull().sum()
airbnb.drop(['name','id','host_name','last_review'], axis=1, inplace=True)
airbnb.head()
airbnb.fillna({'reviews_per_month':0}, inplace=True)
airbnb.isnull().sum()
airbnb.isnull().sum()
airbnb.dropna(how='any',inplace=True)
airbnb.info() #.info() function is used to get a concise summary of the dataframe
airbnb.describe()
corr = airbnb.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
airbnb.columns
airbnb.head(15)
sns.countplot(airbnb['neighbourhood_group'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Neighbourhood Group')
sns.countplot(airbnb['neighbourhood'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(25,6)
plt.title('Neighbourhood')
#Restaurants delivering Online or not
sns.countplot(airbnb['room_type'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants delivering online or Not')
plt.figure(figsize=(10,10))
ax = sns.boxplot(data=airbnb, x='neighbourhood_group',y='availability_365',palette='plasma')
plt.figure(figsize=(10,6))
sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.neighbourhood_group)
plt.ioff()
plt.figure(figsize=(10,6))
sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.room_type)
plt.ioff()
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080
                         ).generate(" ".join(airbnb.neighbourhood))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('neighbourhood.png')
plt.show()
df = airbnb
class MLP(MLPRegressor):
    @ignore_warnings(category=ConvergenceWarning)
    def __init__(self, sizes=(100,), act='relu', max_i=200):
        super().__init__(hidden_layer_sizes=sizes, activation=act, max_iter=max_i)
class ELM(object):
    def __init__(self, hidden_units=200):
        self._hidden_units = hidden_units
        
    def train(self, X, Y):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        self.random_weights = np.random.randn(X.shape[1], self._hidden_units)
        G = np.tanh(X.dot(self.random_weights))
        self.w_elm = np.linalg.pinv(G).dot(Y)
        
    def predict(self, X):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        G = np.tanh(X.dot(self.random_weights))
        return G.dot(self.w_elm)
df = df.drop(axis=1, labels=[ 'host_id', 'neighbourhood', 'latitude', 'longitude', 'reviews_per_month'])
df.head()
le = LabelEncoder()
df['neighbourhood_group'] = le.fit_transform(df['neighbourhood_group'])
df['room_type'] = le.fit_transform(df['room_type'])
df.head()
price_div = df.price.max()
min_nights_div = df.minimum_nights.max()
n_review_div = df.number_of_reviews.max()
calcu_div = df.calculated_host_listings_count.max()
availa_div = df.availability_365.max()
df.price = df.price / price_div
df.minimum_nights = df.minimum_nights / min_nights_div
df.number_of_reviews = df.number_of_reviews / n_review_div
df.calculated_host_listings_count = df.calculated_host_listings_count / calcu_div
df.availability_365 = df.availability_365 / availa_div
X = df.drop(axis=1, labels=['price'])
Y = df.price
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=32)
class Helper(object):
    def get_elm(self):
        return self.elm
    
    def get_mlp(self):
        return self.mlp
    
    @ignore_warnings(category=ConvergenceWarning)
    def train_both(self, X, Y, n=100):
        self.iterations = list(range(1, n+1))
        self.elm_mse = []
        self.mlp_mse = []
        for i in self.iterations:
            print(f'{i}.. ', end=' ')
            self.elm = ELM(i)
            self.elm.train(X, Y)
            self.elm_mse.append(np.mean((Y - self.elm.predict(X))**2))
            self.mlp = MLP(sizes=(50,), act='tanh', max_i=i)
            self.mlp.fit(X, Y)
            self.mlp_mse.append(np.mean((Y - self.mlp.predict(X))**2))
        return (self.elm_mse, self.mlp_mse)
misc = Helper()
elm_mse, mlp_mse = misc.train_both(X_train, y_train, 100)
plt.figure(figsize=(16, 9))
plt.title("Extreme Learning Machine vs Multilayer Perceptron Performance Comparison")
plt.plot(elm_mse, '+-', label='Extreme Learning Machine')
plt.plot(mlp_mse, '^-', label='Multilayer Perceptron')
plt.legend()
#plt.yscale('log')
plt.show()

elm = misc.get_elm()
mlp = misc.get_mlp()
print(f'ELM MSE: {np.mean((y_test-elm.predict(X_test))**2)}')
print(f'MLP MSE: {np.mean((y_test-mlp.predict(X_test))**2)}')
from sklearn import datasets, linear_model, metrics
reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
reg.fit(X_train, y_train) 
# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(X_test, y_test))) 
# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 
  
## plotting residual errors in training data 
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 
            color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 
import xgboost as xgb
from sklearn.metrics import r2_score
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=1000, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(X_train, y_train)
r2_score(model_xgb.predict(X_test), y_test)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

cb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.05,
                             depth=10,
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
cb_model.fit(X_train, y_train)
r2_score(cb_model.predict(X_test), y_test)
from sklearn.linear_model  import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
BR = BayesianRidge()
BR.fit(X_train, y_train)
r2_score(BR.predict(X_test), y_test)















