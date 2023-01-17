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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv(r"../input/onlineretail/OnlineRetail.csv", encoding='cp1252', parse_dates=['InvoiceDate'])
#Basic Information-
df.head()
df.info()
df.describe()
maxdate = df['InvoiceDate'].dt.date.max()
mindate = df['InvoiceDate'].dt.date.min()
customers = df['CustomerID'].nunique()
stock = df['StockCode'].nunique()
quantity = df['Quantity'].sum()

print(f'Transactions timeframe: {mindate} to {maxdate}.')
print(f'Unique customers: {customers}.')
print(f'Unique items sold: {stock}.')
print(f'Quantity sold in period {quantity}')
#two exceptions here negative prices
df=df[df.UnitPrice>=0];
df.drop(df[df['Quantity']<0].index, axis=0, inplace=True)
df['Revenue'] = df['UnitPrice']*df['Quantity']
#Frequency of customer, revenue per customer, revenue per item, revenue per unit price
users = pd.DataFrame(df['CustomerID'].unique())
users.columns = ['CustomerID']
frequency_score = df.groupby('CustomerID')['InvoiceDate'].count().reset_index()
frequency_score.columns = ['CustomerID', 'Frequency']
customer_money=df.groupby('CustomerID')['Revenue'].mean().reset_index()
customer_money.columns = ['CustomerID', 'TotalRev']
usersf = pd.merge(users, frequency_score, on='CustomerID')
usersf.head()
userm=pd.merge(usersf, customer_money, on='CustomerID')
userm.head()
items = pd.DataFrame(df['StockCode'].unique())
items.columns = ['StockCode']
items.head()
item_money = df.groupby('StockCode')['Revenue'].sum().reset_index()
item_money.columns = ['StockCode', 'TotalRev']
items = pd.merge(items, item_money, on='StockCode')
items.head()
prices = pd.DataFrame(df['UnitPrice'].unique())
prices.columns = ['UnitPrice']
prices.head()
price_money = df.groupby('UnitPrice')['Revenue'].sum().reset_index()
price_money .columns = ['UnitPrice', 'TotalRev']
prices = pd.merge(prices, price_money, on='UnitPrice')
prices.head()

#we can see that there are two unit prices which bring in the most revenue
#we have a few price which is bringing in negative revenue
sns.scatterplot(items['TotalRev'],items['StockCode']);
plt.yticks(items['StockCode'], '')
plt.show()
#we can see that some stock codes have generated much much more revenue than others
sns.scatterplot(prices['TotalRev'],prices['UnitPrice']);
plt.show()
#we get 4 customers with extreme values of frequency

sns.scatterplot(userm['Frequency'],userm['CustomerID']);
plt.show()
#most money making customers
sns.scatterplot(userm['TotalRev'],userm['CustomerID']);
plt.show()
sns.scatterplot(userm['TotalRev'],userm['Frequency']);
plt.show()
userm.describe()
#storing outliers of positive interest in respective dataframes
most_freq_customers=userm[userm.Frequency>=1000];
most_money_making_customers=userm[userm.TotalRev>=10000];
#super customers are customers which are both frequent and money making
super_customers=most_money_making_customers[most_money_making_customers.Frequency>=1000]
most_successfull_products=items[items.TotalRev>=50000];
most_succesfull_unit_prices=prices[prices.TotalRev>=100000];
#storing negative values as they are of importance
negative_money_making_customers=userm[userm.TotalRev<=0];
least_successfull_products=items[items.TotalRev<=0];
#Advanced Algorithms
!pip install pyod
!pip install --upgrade pyod
import pandas as pd
import numpy as np

# Import models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
userm.plot.scatter('Frequency','TotalRev')
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
userm[['Frequency','TotalRev']] = scaler.fit_transform(userm[['Frequency','TotalRev']])
userm[['Frequency','TotalRev']].head()
#storing in np array for models
X1 = userm['Frequency'].values.reshape(-1,1)
X2 = userm['TotalRev'].values.reshape(-1,1)

X = np.concatenate((X1,X2),axis=1)
X
random_state = np.random.RandomState(45)
outliers_fraction = 0.005
classifiers = {
        'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
        'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',contamination=outliers_fraction)
}

dict={}
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
def model_printer(X,userm):
  xx , yy = np.meshgrid(np.linspace(0,1 , 200), np.linspace(0, 1, 200))

  for i, (clf_name, clf) in enumerate(classifiers.items()):
      clf.fit(X)
      scores_pred = clf.decision_function(X) * -1
      y_pred = clf.predict(X)
      n_inliers = len(y_pred) - np.count_nonzero(y_pred)
      n_outliers = np.count_nonzero(y_pred == 1)
      plt.figure(figsize=(20, 20))
      
      # copy of dataframe
      dfx = userm
      dfx['outlier'] = y_pred.tolist()
      
      # IX1 - inlier feature 1,  IX2 - inlier feature 2
      IX1 =  np.array(dfx['Frequency'][dfx['outlier'] == 0]).reshape(-1,1)
      IX2 =  np.array(dfx['TotalRev'][dfx['outlier'] == 0]).reshape(-1,1)
      
      # OX1 - outlier feature 1, OX2 - outlier feature 2
      OX1 =  dfx['Frequency'][dfx['outlier'] == 1].values.reshape(-1,1)
      OX2 =  dfx['TotalRev'][dfx['outlier'] == 1].values.reshape(-1,1)
          
      print('OUTLIERS:',n_outliers,'INLIERS:',n_inliers, clf_name)
          
      # threshold value to consider a datapoint inlier or outlier
      threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
          
      # decision function calculates the raw anomaly score for every point
      Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
      Z = Z.reshape(xx.shape)
            
      # fill blue map colormap from minimum anomaly score to threshold value
      plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Greens_r)
          
      # draw red contour line where anomaly score is equal to thresold
      a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
          
      # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
      plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
          
      b = plt.scatter(IX1,IX2, c='white',s=20, edgecolor='k')
      
      c = plt.scatter(OX1,OX2, c='yellow',s=20, edgecolor='k')
        
      plt.axis('tight')  
      
      # loc=2 is used for the top left corner 
      plt.legend(
          [a.collections[0], b,c],
          ['learned decision function', 'inliers','outliers'],
          prop=matplotlib.font_manager.FontProperties(size=20),
          loc=2)
        
      plt.xlim((0, 1))
      plt.ylim((0, 1))
      dict[clf_name]=dfx[dfx.outlier==1]
      plt.title(clf_name)
      print("\n")
      plt.show()
      
import warnings
warnings.filterwarnings("ignore")
model_printer(X,userm)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
prices[['UnitPrice','TotalRev']] = scaler.fit_transform(prices[['UnitPrice','TotalRev']])
prices[['UnitPrice','TotalRev']].head()
#storing in np array for models
X_1 = prices['UnitPrice'].values.reshape(-1,1)
X_2 = prices['TotalRev'].values.reshape(-1,1)

X_ = np.concatenate((X_1,X_2),axis=1)
X_
#will have to write the function again as it was set up for frequency
dict2={}
def model_printer2(X,userm):
  xx , yy = np.meshgrid(np.linspace(0,1 , 200), np.linspace(0, 1, 200))

  for i, (clf_name, clf) in enumerate(classifiers.items()):
      clf.fit(X)
      scores_pred = clf.decision_function(X) * -1
      y_pred = clf.predict(X)
      n_inliers = len(y_pred) - np.count_nonzero(y_pred)
      n_outliers = np.count_nonzero(y_pred == 1)
      plt.figure(figsize=(20, 20))
      
      # copy of dataframe
      dfx = userm
      dfx['outlier'] = y_pred.tolist()
      
      # IX1 - inlier feature 1,  IX2 - inlier feature 2
      IX1 =  np.array(dfx['UnitPrice'][dfx['outlier'] == 0]).reshape(-1,1)
      IX2 =  np.array(dfx['TotalRev'][dfx['outlier'] == 0]).reshape(-1,1)
      
      # OX1 - outlier feature 1, OX2 - outlier feature 2
      OX1 =  dfx['UnitPrice'][dfx['outlier'] == 1].values.reshape(-1,1)
      OX2 =  dfx['TotalRev'][dfx['outlier'] == 1].values.reshape(-1,1)
          
      print('OUTLIERS:',n_outliers,'INLIERS:',n_inliers, clf_name)
          
      # threshold value to consider a datapoint inlier or outlier
      threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
          
      # decision function calculates the raw anomaly score for every point
      Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
      Z = Z.reshape(xx.shape)
            
      # fill blue map colormap from minimum anomaly score to threshold value
      plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Greens_r)
          
      # draw red contour line where anomaly score is equal to thresold
      a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
          
      # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
      plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
          
      b = plt.scatter(IX1,IX2, c='white',s=20, edgecolor='k')
      
      c = plt.scatter(OX1,OX2, c='yellow',s=20, edgecolor='k')
        
      plt.axis('tight')  
      
      # loc=2 is used for the top left corner 
      plt.legend(
          [a.collections[0], b,c],
          ['learned decision function', 'inliers','outliers'],
          prop=matplotlib.font_manager.FontProperties(size=20),
          loc=2)
        
      plt.xlim((0, 1))
      plt.ylim((0, 1))
      plt.title(clf_name)
      print("\n")
      plt.show()
      dict2[clf_name]=dfx[dfx.outlier==1]

  
model_printer2(X_,prices)
#dict has all the outliers of first(userm)
#dict2 has all the outliers of second(prices)
dict.items()
dict2.items()
#Seasonality Report
import pandas as pd
from datetime import datetime, timedelta
dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M')
df3 = pd.read_csv(r"../input/onlineretail/OnlineRetail.csv", encoding='cp1252', parse_dates=['InvoiceDate'],date_parser=dateparse)
df3=df3[df3.Quantity>=0];
df3=df3[df3.UnitPrice>=0]
df3.info()
df3.head()
qt = pd.DataFrame(df3['InvoiceDate'].unique())
qt.columns = ['InvoiceDate']
qt.head()

qt_quan = df3.groupby('InvoiceDate')['Quantity'].sum()
qt_quan.columns = ['InvoiceDate','Quantity']
qtm=pd.merge(qt, qt_quan, on='InvoiceDate')
qtm2=qtm;
qtm.head()
qtm3=qtm2
from datetime import datetime
qtm3.info()
qtm3.index.dtype
qtm4=qtm3.set_index('InvoiceDate')
qtm4.head()
qtm4.plot()
y = qtm4['Quantity'].resample('d').sum()
y
import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(y ,model='additive')
decomposition.plot()
decomposition.seasonal
#fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(15,8))
import pylab
pylab.rcParams['figure.figsize'] = (14, 9)

decomposition.seasonal.plot()
print("Seasonality")