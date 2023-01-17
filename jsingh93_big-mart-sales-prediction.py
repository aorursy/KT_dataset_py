import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df_sales = pd.read_csv('../input/Train.csv')
df_sales.head()
df_sales.info()
df_sales.describe()
plt.figure(figsize=(10,6))
sns.heatmap(df_sales.corr(),annot=True)
sns.distplot(df_sales['Item_Outlet_Sales'])
df_sales.columns
plt.figure(figsize=(20,6))
sns.boxplot(x='Item_Type',y='Item_Weight',data=df_sales)
df_sales['Item_Weight'][df_sales['Item_Weight'].isnull()]=df_sales['Item_Weight'].mean()
df_sales.info()
sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',data= df_sales)
ndf_sale=df_sales[['Item_Weight', 'Item_Visibility', 'Item_MRP','Outlet_Establishment_Year','Item_Outlet_Sales',]]
sns.pairplot(ndf_sale)
from sklearn.cross_validation import train_test_split
X= ndf_sale.drop(['Item_Outlet_Sales'],axis=1)
y=ndf_sale['Item_Outlet_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=105)
from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(X_train,y_train)
print(model.intercept_)
print(model.coef_)
cdf = pd.DataFrame(model.coef_,X.columns,columns=['Coeff'])
cdf
sns.distplot(y_test-predictions)
predictions=model.predict(X_test)
predictions
from sklearn import metrics
metrics.mean_absolute_error(y_test,predictions)
metrics.mean_squared_error(y_test,predictions)
np.sqrt(metrics.mean_squared_error(y_test,predictions))
df_test_sales= pd.read_csv('../input/Test.csv')
df_test_sales.info()
plt.figure(figsize=(20,6))
sns.boxplot(x='Item_Type',y='Item_Weight',data=df_test_sales)
df_test_sales['Item_Weight'][df_test_sales['Item_Weight'].isnull()]= df_test_sales['Item_Weight'].mean()
sns.countplot('Outlet_Size',data=df_test_sales)
sns.pairplot(df_test_sales)
ndf_test_sale= df_test_sales[['Item_Weight', 'Item_Visibility', 'Item_MRP','Outlet_Establishment_Year']]
sns.heatmap(ndf_test_sale.corr(),annot=True)
X= ndf_test_sale
df_test_sales['Item_Outlet_Sales']=model.predict(X)
df_test_sales
data_frame= df_test_sales[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
data_frame.head()
data_frame.to_csv('Big Market Sales.csv',index=0)