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
#heatmap of the profit by sub category
df_pivot = df_all.pivot_table(values='Profit',index=['Category','SubCategory'],columns='OrderPriority',aggfunc='sum')
sns.heatmap(df_pivot,annot=True)
#bar cart for the sales and profit for each order priority
by_priority = df_all.groupby(['OrderPriority']).sum()[['Sales','Profit']]
by_priority.plot(kind='bar')
#distribution plot of sales
sns.distplot(df_all['Sales'])
#jointplot between saes and profit
sns.jointplot(data=df_all,x='Profit',y='Sales',kind='reg')
#Profit and sales value by shipping region
by_region = df_all.groupby(['ShippingRegion']).sum()[['Profit','Sales']]
by_region.plot(kind='bar')
#number of sales divided by region for each category
sns.barplot(x="Sales", y="ShippingRegion", hue="Category",data=df_all, palette="coolwarm")
#amount of profit obtained by region for each category
sns.barplot(x="Profit", y="ShippingRegion", hue="Category",data=df_all, palette="coolwarm")
#X and y arrays
X = df_all[['Sales','Shipping Cost','Discount']]
y = df_all['Profit']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=100);
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
