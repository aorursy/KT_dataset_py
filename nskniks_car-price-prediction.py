# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/kaggle/input/car-price-prediction/CarPrice_Assignment.csv')
df.head(5)
df.info()
df.describe()
sns.barplot(x ='fuelsystem',y ='price',data= df)
df.fuelsystem.unique()
m = { 'mpfi':0,'2bbl':1, 'mfi':2, '1bbl':3, 'spfi':4, '4bbl':5, 'idi':6, 'spdi':7
    }
df['fuelsystem'] = df['fuelsystem'].map(m)

# these information place a vital role in the price of the car ,which is categorical now: so need to convert to integer
# mapping need to be done
# find which is most relevent
sns.barplot(x ='fueltype',y ='price',data= df)
df.fueltype.unique()
m = { 'gas':1,'diesel':2}
df['fueltype'] = df['fueltype'].map(m)

sns.barplot(x ='enginelocation',y ='price',data= df)
df.enginelocation.unique()
m = { 'front':1,'rear':2}
df['enginelocation'] = df['enginelocation'].map(m)

sns.barplot(x ='aspiration',y ='price',data= df)
df.aspiration.unique()
m = { 'std':1,'turbo':2}
df['aspiration'] = df['aspiration'].map(m)
sns.barplot(x ='drivewheel',y ='price',data= df)
df.drivewheel.unique()
m = { 'rwd':0,'fwd':1,'4wd':2}
df['drivewheel'] = df['drivewheel'].map(m)
sns.barplot(x ='carbody',y ='price',data= df)
df.carbody.unique()
m = { 'convertible':0,'hatchback':1,'sedan':2,'wagon':3,'hardtop':4,}
df['carbody'] = df['carbody'].map(m)
sns.barplot(x ='enginetype',y ='price',data= df)
df.enginetype.unique()

m = { 'dohc':0,'ohcv':1,'ohc':2,'l':3,'rotor':4,'ohcf':5,'dohcv':6}
df['enginetype'] = df['enginetype'].map(m)


sns.barplot(x ='cylindernumber',y ='price',data= df)

df.cylindernumber.unique()

m = { 'four':4,'six':6,'five':5,'three':3,'twelve':12,'two':2,'eight':8}
df['cylindernumber'] = df['cylindernumber'].map(m)
#find the corrrelation among int values
df.drop(['car_ID','symboling'],axis =1).corr()
sns.heatmap(df.drop(['car_ID','symboling'],axis =1).corr())
# so most relevent thing from the above graph: 
#'carlength', 'carwidth', 'curbweight', 'enginetype','cylindernumber', 'enginesize', 'fuelsystem', 'boreratio','horsepower


sns.pairplot(df,x_vars =['wheelbase','enginelocation',
       'carlength', 'carwidth', 'curbweight',
       'cylindernumber', 'enginesize',
       'compressionratio', 'horsepower'],y_vars ='price',kind='reg')
df.info()

df.columns
X = df[['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase',
       'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype',
       'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio','horsepower']]
y=df['price']
from sklearn.model_selection import train_test_split

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,random_state=101)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
from sklearn.linear_model import LinearRegression
lm =LinearRegression()
lm.fit(X_train,y_train)
#intercept:
lm.intercept_
#coefficients
print('coefficients :',lm.coef_)
car_coeff =  pd.DataFrame(lm.coef_,X_test.columns,columns=['co-effi'])

car_coeff
# Train set score
lm.score(X_train,y_train)
# Test set score
lm.score(X_test,y_test)
y_pre = lm.predict(X_test)
plt.scatter(y_test,y_pre)
# prediction score : 
lm.score(X_test,y_pre)
sns.distplot(y_test-y_pre,bins =25)
sns.distplot(y_pre)
from sklearn import metrics

metrics.r2_score(y_test,y_pre)
metrics.explained_variance_score(y_test,y_pre)
print('MAE :',metrics.mean_absolute_error(y_test,y_pre))
print('MSE :',metrics.mean_squared_error(y_test,y_pre))
print('RMSE :',np.sqrt(metrics.mean_squared_error(y_test,y_pre)))

price_table = X_test.copy()
price_table ['Actual'] =y_test
price_table ['predicted'] =y_pre
price_table.head(100)
sns.lmplot('Actual','predicted',data = price_table)


