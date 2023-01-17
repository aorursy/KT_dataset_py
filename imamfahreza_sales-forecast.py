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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(rc={'figure.figsize':(11.7,8.27)})
factmarketsales = pd.read_excel('../input/global-market-sales/FactMarketSales.xlsx')
orders = pd.read_excel('../input/global-market-sales/Orders.xlsx')
products = pd.read_excel('../input/global-market-sales/Products.xlsx')
shippings = pd.read_excel('../input/global-market-sales/Shippings.xlsx')
df_all = factmarketsales.merge(orders,on='OrderCode').merge(products,on='ProductId').merge(shippings,on='OrderCode')
df_all.head(5)
df_all['OrderDate']=pd.to_datetime(df_all['OrderDate'])
df_all['ShipDate']=pd.to_datetime(df_all['ShipDate'])
df_all['Date Order'] = [d.date() for d in df_all['OrderDate']]
df_all['Time Order'] = [d.time() for d in df_all['OrderDate']]
df_all['Date Order']=pd.to_datetime(df_all['Date Order'])
df_all.head(5)
df_all.describe().drop(['ProductId','OrderKey','ProductKey','ShipKey_y'],axis=1)
df_all.columns
df_all[['Sales', 'Quantity','Discount', 'Profit', 'Shipping Cost','Date Order']].plot(x='Date Order',kind='line', subplots=True, figsize=(20,15))
plt.show()
df_average_sales_month = df_all.groupby(by=['Date Order'], as_index=False)['Sales'].sum()
df_average_sales = df_average_sales_month.sort_values('Sales', ascending=False)

plt.figure(figsize=(20,5))
plt.plot(df_average_sales_month['Date Order'], df_average_sales_month['Sales'])
plt.show()
# The more remunerative
df_average_sales.head()
# The least remunerative
df_average_sales[::-1].head()
# Top performing type of Sub Category in term of sales
df_top_stores = df_all.groupby(by=['SubCategory'], as_index=False)['Sales'].sum()
df_top_stores.sort_values('Sales', ascending=False).head(5)
# Top performing type of Shipping Region in term of sales
df_top_stores = df_all.groupby(by=['ShippingRegion'], as_index=False)['Sales'].sum()
df_top_stores.sort_values('Sales', ascending=False).head(5)
# Top performing type of Order Priority in term of sales
df_top_stores = df_all.groupby(by=['OrderPriority'], as_index=False)['Sales'].sum()
df_top_stores.sort_values('Sales', ascending=False).head()
#Foreast of total sales volume
ts = df_average_sales_month.set_index('Date Order')
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf

fig, axes = plt.subplots(1,2, figsize=(20,5))
plot_acf(ts, lags=100, ax=axes[0])
plot_pacf(ts, lags=100, ax=axes[1])
plt.show()
from sklearn.linear_model import LinearRegression

def fit_ar_model(ts, orders):
    
    X=np.array([ ts.values[(i-orders)].squeeze() if i >= np.max(orders) else np.array(len(orders) * [np.nan]) for i in range(len(ts))])
    
    mask = ~np.isnan(X[:,:1]).squeeze()
    
    Y= ts.values
    
    lin_reg=LinearRegression()
    
    lin_reg.fit(X[mask],Y[mask])
    
    print(lin_reg.coef_, lin_reg.intercept_)

    print('Score factor: %.2f' % lin_reg.score(X[mask],Y[mask]))
    
    return lin_reg.coef_, lin_reg.intercept_
    
def predict_ar_model(ts, orders, coef, intercept):
    return np.array([np.sum(np.dot(coef, ts.values[(i-orders)].squeeze())) + intercept  if i >= np.max(orders) else np.nan for i in range(len(ts))])
orders=np.array([1,6,52])
coef, intercept = fit_ar_model(ts,orders)
pred=pd.DataFrame(index=ts.index, data=predict_ar_model(ts, orders, coef, intercept))
plt.figure(figsize=(20,5))
plt.plot(ts, 'o')
plt.plot(pred)
plt.show()
diff=(ts['Sales']-pred[0])/ts['Sales']

print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std()))
 
plt.figure(figsize=(20,5))
plt.plot(diff, c='blue')
plt.grid()
plt.show()
#Forecast of the store-wise sales volume
#Develop the forecast model for the phone, which shows the highest sales volume.

df_phone=df_all.where(df_all['SubCategory'] == 'Phones')
df_phone=df_phone.dropna()
df_phone=df_phone.groupby(by=['Date Order'], as_index=False)['Sales'].sum()
df_phone = df_phone.set_index('Date Order')
df_phone.head()
plt.figure(figsize=(20,5))
plt.plot(df_phone.index, df_phone.values)
plt.show()
fig, axes = plt.subplots(1,2, figsize=(20,5))
plot_acf(df_phone.values, lags=100, alpha=0.05, ax=axes[0])
plot_pacf(df_phone.values, lags=100, alpha=0.05, ax=axes[1])
plt.show()
orders=np.array([1,6,29,46,52])
coef, intercept = fit_ar_model(df_phone,orders)
pred=pd.DataFrame(index=df_phone.index, data=predict_ar_model(df_phone, orders, coef, intercept))
plt.figure(figsize=(20,5))
plt.plot(df_phone, 'o')
plt.plot(pred)
plt.show()
diff=(df_phone['Sales']-pred[0])/df_phone['Sales']

print('AR Residuals: avg %.2f, std %.2f' % (diff.mean(), diff.std()))
 
plt.figure(figsize=(20,5))
plt.plot(diff, c='orange')
plt.grid()
plt.show()
dfext=df_all.where( df_all['SubCategory'] == 'Phones')
dfext=dfext.dropna()
dfext=dfext.groupby(by=['Date Order'], as_index=False)[['Sales', 'Quantity','Discount', 'Profit', 'Shipping Cost']].mean()
dfext = dfext.set_index('Date Order')
dfext.head()
dfext.describe()
import seaborn as sns
corr = dfext.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, 
            annot=True, fmt=".3f",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()
corr['Sales'].sort_values(ascending=False)