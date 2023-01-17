# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt  
import seaborn as sn
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Importing the dataset 
data = pd.read_csv('../input/1000-companies/1000_Companies.csv')
data.head()
data.info()
data.describe(include = 'all')
data.State.value_counts()
data.columns = ['RDSpend', 'Administration', 'MarketingSpend', 'State', 'Profit']
##PLOTS
## Lets Check the Profit vs Each attr
data.plot(x='RDSpend', y='Profit',style = 'o')  
plt.title('RDSpend vs Profit')  
plt.xlabel('RDSpend')  
plt.ylabel('Profit')  
plt.show()
##From the above graphical interpretation it can be seen that the Profit increases mostly with increase in RDSpend
##However when the RDSpend is 100000-150000 the Profit is highest

## Can be verified below :
x = data[(data.RDSpend < 100000)]
y = data[(data.RDSpend > 100000)]
print("Maximum Profit when RDSpend is below 100000 is :" , x.Profit.max())
print("Maximum Profit when RDSpend is above 100000 is :" , y.Profit.max())
x.shape
y.shape
## Also it can be seen that the dataset contains 60% data with RDSpend below 100000 and Profit less than 141585.52
data.plot(x='Administration', y='Profit', style='o')  
plt.title('Administration vs Profit')  
plt.xlabel('Administration')  
plt.ylabel('Profit')  
plt.show()
## Above Graph shows uneven data distribution
data.plot(x='MarketingSpend', y='Profit', style='or')  
plt.title('MarketingSpend vs Profit')  
plt.xlabel('MarketingSpend')  
plt.ylabel('Profit')  
plt.show()        
## It can be seen that the Profit is highest when  MarketingSpend is close to 300000
z = data.Profit.max()
data.MarketingSpend[(data.Profit == z)]
##In which states do people have more profit
filter_state = pd.DataFrame(data.groupby(["State"])["Profit"].sum()).reset_index()
sn.barplot(y = 'Profit', x = 'State',data = filter_state, edgecolor = 'w')
plt.show()
##Profit is high for California then for New York and then for Florida
## Re-checking below
data.State.value_counts()
## Checking the distribution of categorical and continuous vars
## Individually checking the distribution for each var
## For continuos var we plot displot from seaborn library
sn.distplot(data.RDSpend,rug = True)
sn.distplot(data.Administration,rug = True)
sn.distplot(data.MarketingSpend,rug = True)
## For catgeorical we plot bar plot
data['State'].value_counts().plot(kind='bar')
 ##Let’s check the profit and once we plot it we can observe that the Average Profit is Between Nearly 100000 and 200000.
plt.figure(figsize=(15,10))
plt.tight_layout()
sn.distplot(data['Profit'])
data.Profit.mean()
## Encoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data.State = le.fit_transform(data.State)
data.State = data.State.astype('category')
data.State.value_counts()
data.isnull().sum().sum()
data.corr()
sn.heatmap(data.corr(), annot=True)
data_x = data.iloc[:,0:4]
data_x.columns
data_x.shape
data_y = data.iloc[:,4]
data_y
data_y.shape
data_x_train,data_x_test,data_y_train,data_y_test = train_test_split(data_x,data_y,test_size = 0.2,random_state = 101)
data_x_train.shape
data_y_train.shape
data_x_test.shape
data_y_test.shape
lr = LinearRegression()
lr.fit(data_x_train,data_y_train) ## Training the algorithm
#To retrieve the intercept:
print(lr.intercept_)
#For retrieving the slope:
print(lr.coef_)
pred_val = lr.predict(data_x_test)
compare = pd.DataFrame({'Actual': data_y_test, 'Predicted': pred_val})
compare

df1 = compare.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(data_y_test, pred_val))  
print('Mean Squared Error:', metrics.mean_squared_error(data_y_test, pred_val))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(data_y_test, pred_val)))
data.corr()
data1 = pd.read_csv('../input/1000-companies/1000_Companies.csv')
data1.head()
data2 = data1.iloc[:,[0,1,3,4]] ## Removing MarketingSpend
data2.head()
data2.State.value_counts()
data2.State = le.fit_transform(data2.State)
data2.State.value_counts()
data2.State.describe()
data2.State = data2.State.astype('category')
data2.State.describe()
data2.head()
data2_x = data2.iloc[:,0:3]
data2_x.head()
data2_y = data2.iloc[:,3]
data2_y.head()
data2_x_train,data2_x_test,data2_y_train,data2_y_test = train_test_split(data2_x,data2_y,test_size = 0.2, random_state = 101)
data2_x_train.shape
data2_y_train.shape
data2_x_test.shape
data2_y_test.shape
lr1 = LinearRegression()
lr1.fit(data2_x_train,data2_y_train)
pred_val1 = lr1.predict(data2_x_test)
#To retrieve the intercept:
print(lr1.intercept_)
#For retrieving the slope:
print(lr1.coef_)
compare1 = pd.DataFrame({'Actual': data2_y_test, 'Predicted': pred_val1})
compare1
compare2 = compare1.head(25)
compare2.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(data2_y_test, pred_val1))  
print('Mean Squared Error:', metrics.mean_squared_error(data2_y_test, pred_val1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(data2_y_test, pred_val1)))