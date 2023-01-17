import numpy as np

import pandas as pd 
myData=pd.read_csv('../input/autompg-dataset/auto-mpg.csv')

myData.head(10)
## clone the datasets

cmyData=myData.copy()
## Remove car name feature

cmyData=cmyData.drop(columns='car name',axis=1)

cmyData.head()
## Replace 1 as American,2 as Europian,3 as Japanese

cmyData['origin']=cmyData['origin'].replace({1:'American',2:'Europian',3:'Japanese'})

cmyData.head()
## create 3 simple true or false columns for origin feature

cmyData=pd.get_dummies(cmyData,columns=['origin'])

cmyData
## Verify the presence of Null value

cmyData[cmyData.isna().any(axis=1)]
cmyData.dtypes
## Verify the presence of any value other than numeric

hpIsdigit=pd.DataFrame(cmyData['horsepower'].str.isdigit())

cmyData[hpIsdigit['horsepower']==False]
## Replace these ? mark to NaN value

cmyData=cmyData.replace('?',np.nan)
## Replace NaN value with its median value and convert the dtypes

cmyData=cmyData.fillna(cmyData.median())

cmyData['horsepower']=cmyData['horsepower'].astype('float64')
import matplotlib.pyplot as plt

import seaborn as sns

sns.pairplot(cmyData)
ax=cmyData.boxplot(figsize=(20,15))

ax.set_xticklabels(ax.get_xticklabels(),rotation=45 )

ax.tick_params(axis='both', which='major', labelsize=13)

plt.show()
cmyData=cmyData.drop(columns='origin_Japanese')

X=cmyData.drop(['mpg'],axis=1)

y=cmyData['mpg']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LinearRegression

regression_model=LinearRegression()

regression_model.fit(X_train,y_train)
## Get score on training data

regression_model.score(X_train,y_train)
#out of sample score (R^2)

regression_model.score(X_test,y_test)
from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=2,interaction_only=True)

X_train2=poly.fit_transform(X_train)

X_test2=poly.fit_transform(X_test)
#Out off sample (testing) R^2 is our measure of sucess and does improve



## Fit polynomial

regression_model.fit(X_train2,y_train)



#In sample (training) R^2 will always improve with the number of variables!

print(regression_model.score(X_train2,y_train))



#Out off sample (testing) R^2 is our measure of sucess and does improve

print(regression_model.score(X_test2,y_test))
# but this improves as the cost of 29 extra variables!

print(X_train.shape)

print(X_train2.shape)