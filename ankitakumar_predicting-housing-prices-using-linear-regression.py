# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
#import scikit libraries
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
#Read the csv file using pandas dataframe
df = pd.read_csv('../input/kc_house_data.csv')
#printing the first 5 rows of dataframe
df.head()
#summary statistics of the dataframe
df.describe()
df.corr()
#This shows how price is positively correlated to other features
#Create the linear regression model
model=linear_model.LinearRegression()
#split data into training data and testing data
train_data,test_data=train_test_split(df,train_size=0.8,random_state=3)
X_train=np.array(train_data['sqft_living'],dtype=pd.Series).reshape(-1,1)
y_train=np.array(train_data['price'],dtype=pd.Series).reshape(-1,1)
#fit the training data
model.fit(X_train,y_train)
#prepare the testing data
X_test=np.array(test_data['sqft_living'],dtype=pd.Series).reshape(-1,1)
y_test=np.array(test_data['price'],dtype=pd.Series).reshape(-1,1)
print('Average price of the Test Data is %.3f'%(y_test.mean()))
#predict y_test
prediction=model.predict(X_test)
print('Mean Squared error is %.3f'%(np.sqrt(metrics.mean_squared_error(y_test,prediction))))
print('Intercept is %f'%(model.intercept_))
print('Coefficient is %f'%(model.coef_))
#s=model.score(X_train,y_train)

print('R-squared score of the training set is %s'%model.score(X_test,y_test))
print('R-squared score of the testing set is %f'%model.score(X_test,y_test))
#representing the data in graph
plt.figure(figsize=(6.5,5))
plt.scatter(X_test,y_test,color='darkgreen',label="Data",alpha=.1)
plt.xlabel("Space in sqft",fontsize=15)
plt.ylabel("Price is $",fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
sns.set(style='white',font_scale=1)
#list all the features of the dataframe
list(df.columns.values)
features=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']
mask=np.zeros_like(df[features].corr(),dtype=np.bool) #shows whether the features are correlated or not in the form of an array
mask[np.triu_indices_from(mask)]=True
#converts the upper triangular matrux to true from false
f,ax=plt.subplots(figsize=(16,12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(df[features].corr(), linewidths=0.25, vmax=1.0, square=True, cmap="BuGn_r", linecolor='w', annot=True, 
            mask=mask, cbar_kws={"shrink": .75})

f,axes=plt.subplots(1,2,figsize=(15,5))
#1,2 means 1 row 2 columnswill be filled with boxes of dimension 15,5

sns.boxplot(x=train_data['bedrooms'],y=train_data['price'],ax=axes[0])
sns.boxplot(x=train_data['floors'],y=train_data['price'],ax=axes[1])
axes[0].set(xlabel='Bedrooms',ylabel='Price')
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='Floors',ylabel='Price')


#creating another plot to compare bathrooms vs price of house
f,axe=plt.subplots(1,1,figsize=(12.18,5))
sns.boxplot(x=train_data['bathrooms'],y=train_data['price'],ax=axe)
axe.set(xlabel='Bathrooms/Bedrooms',ylabel='Price')

fig=plt.figure(figsize=(19,12.5))
ax=fig.add_subplot(2,2,1, projection="3d")
ax.scatter(train_data['floors'],train_data['bedrooms'],train_data['bathrooms'],c="darkgreen",alpha=.5)
ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nBathrooms / Bedrooms')
ax.set(ylim=[0,12])

ax=fig.add_subplot(2,2,2, projection="3d")
ax.scatter(train_data['floors'],train_data['bedrooms'],train_data['sqft_living'],c="darkgreen",alpha=.5)
ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nsqft Living')
ax.set(ylim=[0,12])

ax=fig.add_subplot(2,2,3, projection="3d")
ax.scatter(train_data['sqft_living'],train_data['sqft_lot'],train_data['bathrooms'],c="darkgreen",alpha=.5)
ax.set(xlabel='\n sqft Living',ylabel='\nsqft Lot',zlabel='\nBathrooms / Bedrooms')
ax.set(ylim=[0,250000])

ax=fig.add_subplot(2,2,4, projection="3d")
ax.scatter(train_data['sqft_living'],train_data['sqft_lot'],train_data['bedrooms'],c="darkgreen",alpha=.5)
ax.set(xlabel='\n sqft Living',ylabel='\nsqft Lot',zlabel='Bedrooms')
ax.set(ylim=[0,250000])
