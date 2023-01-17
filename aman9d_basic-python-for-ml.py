import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt   #data visulisation
import seaborn as sns       #data visulisation

from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression    #linear regression
from sklearn.metrics.regression import mean_squared_error  #error matrics
from sklearn.metrics import mean_absolute_error 


%matplotlib inline
sns.set(color_codes = True)
your_working_path = 'C:/'
#train_data = pd.read_csv(your_working_path + 'file name')
train_data = pd.read_csv('../input/train.csv')
#test_data = pd.read_csv(your_working_path + 'file name')
test_data = pd.read_csv('../input/test.csv')
train_data.head()
test_data.head()

test_data.head(2)
train_data.describe()   #discribe the statistics of data
#counting null value
train_data.isnull().sum()
#train_data.isnull().mean()
#replace the data
#train_data.replace('junk', np.nan, inplace= True)  #np.nun -numpy null value
len(train_data)
train_data.shape
print(train_data.columns)  # show the columns for dataset
print(test_data.count())   #Count the uniqe entries
#droping the feature
train1 = train_data.drop(['Id'], axis=1)
#train_data.dropna(thresh= 'thrshold', axis=1, inplace=True)
train1.head()
plt.plot(train_data['SalePrice'])
plt.show()
train_data['SalePrice'].plot.hist(alpha=0.5)
plt.show()
sns.distplot(train_data['SalePrice'])
print('Skewness: %f'  % train_data['SalePrice'].skew() )
print('Kurtosis: %f' % train_data['SalePrice'].kurt())
temp = pd.concat([train_data['SalePrice'], train_data['GrLivArea']], axis=1)
    #plt.title(var)
print(temp.head())
temp.plot.scatter(x='GrLivArea', y='SalePrice')  #scatter plot
    #plt.show()
    #sns.boxplot(x=var, y=train['SalePrice'], data=temp)  #box plot seaborn
plt.show()
#defin a function to take diffrent input as var and show the relationship with Saleprice
#Correlation matrix
cor_train = train_data.corr()  #Compute pairwise correlation of columns, excluding NA/null values
print(cor_train.shape)
print(train_data.shape)
print(train_data.count())
print(cor_train.count())
plt.subplots(figsize=(12, 9))  #zooming the plot
sns.heatmap(cor_train, vmax=1, square=True)  #heat map
cols_train = cor_train.nlargest(10, 'SalePrice')['SalePrice'].index  #assiing 10 largest matching clomn name 
cm = np.corrcoef(train_data[cols_train].values.T)
cols_train
print(cm.shape)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols_train.values, xticklabels=cols_train.values)
#Scatter plots between 'SalePrice' and correlated variables
sns.set()
corr_variables = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data[corr_variables], size=4)  #plosting the multiple plots same as plt subplot
plt.show()
#spliting the data
train_d, Validation_d = train_test_split(train_data,train_size=0.8, random_state = 0)
train_d_F = train_d[corr_variables]
train_d_F = train_d_F.drop('SalePrice', axis=1)
Validation_d_F = Validation_d[corr_variables]
Validation_d_F = Validation_d_F.drop('SalePrice', axis=1)
print(train_data.shape)
print(train_d.shape)
print(train_d_F.shape)
print(Validation_d.shape)
print(Validation_d_F.shape)
model = LinearRegression()
#model.fit(train_d['GrLivArea'],train_d['SalePrice'])
model.fit(X=train_d_F, y= train_d['SalePrice'])
print(model.coef_)
print('\n', model.intercept_)
model.predict(Validation_d_F)
model.score(Validation_d_F, Validation_d['SalePrice'])
mean_squared_error(Validation_d['SalePrice'],model.predict(Validation_d_F) )