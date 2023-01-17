# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/house-price-prediction-challenge'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
%matplotlib inline
from matplotlib import pyplot as plt
import pandas as pd 
import math
import seaborn as sns
#from sklearn.preprocessing import Imputer
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc,mean_squared_error
# Managing Warnings 
import warnings
warnings.filterwarnings('ignore')
#statistics package
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as s
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
data = pd.read_csv('train.csv')
testt = pd.read_csv('test.csv')
print(data.describe())
#print(len(data))
sample = data.iloc[0:7000,:]
print(data.info())
y = sample['TARGET(PRICE_IN_LACS)']


pd.set_option('display.max_columns',None)#####to display all columns
data.describe(include='object')

sample.corr()
#univariate analysis
print(sns.countplot(x = 'UNDER_CONSTRUCTION',data=sample))
print(pd.crosstab(sample['UNDER_CONSTRUCTION'],columns='count'))
print(sns.countplot(x='RERA',data=sample))
print(pd.crosstab(sample['RERA'],columns='count'))
print(sns.countplot(x = 'READY_TO_MOVE',data=sample))
print(pd.crosstab(sample['READY_TO_MOVE'],columns='count'))
#univariate analysis
#betterway to visulaize house price
fig = plt.figure(figsize=(10,7))
fig.add_subplot(2,1,1)
#sns.displot(sample['TARGET(PRICE_IN_LACS)'])
#fig.add_subplot(2,1,2)
#sns.boxplot(sample['TARGET(PRICE_IN_LACS)'])
plt.tight_layout()
from sklearn.metrics import accuracy_score
model = LinearRegression()
col = ['SQUARE_FT','READY_TO_MOVE']
x = sample[col].values
y = sample['TARGET(PRICE_IN_LACS)'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
model.fit(x_train,y_train)
pred = model.predict(x_test)

print(mean_squared_error(y_test,pred))
from sklearn.metrics import accuracy_score
model = LinearRegression()
col = ['SQUARE_FT','READY_TO_MOVE']
x = sample[col].values
y = sample['TARGET(PRICE_IN_LACS)'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
model.fit(x_train,y_train)
pred = model.predict(x_test)

print(mean_squared_error(y_test,pred))
# plot residuals or errors to show that the error distribution is normal
# which further shows the dependent and independent have a linear relationship
error = (y_test-pred)
#sns.displot(error,kde=True)
#on plotting we have found that error is normally distributed
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
x_train = s_scaler.fit_transform(x_train.astype(np.float))
x_test = s_scaler.fit_transform(x_test.astype(np.float))
model.fit(x_train,y_train)
pred = model.predict(x_test)
print(mean_squared_error(y_test,pred))
error = (y_test-pred)
#sns.displot(error,kde=True)
#on plotting we have found that error is normally distributed
col = ['SQUARE_FT','READY_TO_MOVE']
xx = data[col].values
yy = data['TARGET(PRICE_IN_LACS)'].values
print(len(data))
#remove outlier 
#WE CAN SEE FROM BOX PLOT THAT PRICE HAS LOT OF OUTLIER
#WE HAVE TO REMOVE THEM
#LOWER LIMIT 38-1.5IQR  = 38-31 = 7
#UPPER LIMIT 100+1.5IQR = 100+3 = 193

data = data[data['TARGET(PRICE_IN_LACS)']<165.0]
print(len(data))
col = ['SQUARE_FT','READY_TO_MOVE']
xx = data[col].values
yy = data['TARGET(PRICE_IN_LACS)'].values
print(xx.shape,yy.shape)
train_x,test_x,train_y,test_y = train_test_split(xx,yy,test_size=0.4)
model.fit(train_x,train_y)
predd = model.predict(test_x)
mean_squared_error(test_y,predd)
from sklearn.model_selection import KFold
np.set_printoptions(threshold=np.inf)
kf = KFold(n_splits=5)
l=[]
for train_index,test_index in kf.split(xx):
  #print(train_index,test_index)
  xx_train,xx_test=xx[train_index],xx[test_index]
  yy_train,yy_test=yy[train_index],yy[test_index]
  model.fit(xx_train,yy_train)
  predd = model.predict(xx_test)
  errorr = mean_squared_error(yy_test,predd)
  l.append(error)
  #print(sum(l))
  #print(mean_squared_error(yy_test,predd))
#print(l)#6204
col = ['SQUARE_FT','READY_TO_MOVE']
t = testt[col].values
pd.set_option('display.max_columns',None)#####to display all columns
y_pred = model.predict(t)
print(y_pred)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
l=[]
regressor = DecisionTreeRegressor(random_state=0)
for train_index,test_index in kf.split(xx):
  #print(train_index,test_index)
  xx_train,xx_test=xx[train_index],xx[test_index]
  yy_train,yy_test=yy[train_index],yy[test_index]
  regressor.fit(xx_train,yy_train)
  predd = regressor.predict(xx_test)
  errorr = mean_squared_error(yy_test,predd)
  l.append(error)
  #print(sum(l))
  print(mean_squared_error(yy_test,predd))
regressor = DecisionTreeRegressor(random_state=0.0)#6346