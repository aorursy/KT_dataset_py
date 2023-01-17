import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from pylab import rcParams

import seaborn  as sb
%matplotlib inline

rcParams['figure.figsize']=10,6

plt.style.use('seaborn-whitegrid')
data = pd.read_csv('../input/BlackFriday.csv')

data.head() #overview of our dataset

data.info()
data_gender_purchase =data[['User_ID','Gender','Purchase']]

data_gender_purchase=data_gender_purchase.groupby(['User_ID','Gender']).sum()

data_gender_purchase.reset_index(inplace=True)

data_gender_purchase.head()
sb.countplot(data_gender_purchase["Gender"])
print (data_gender_purchase.groupby('Gender')['Purchase'].sum())

print (data_gender_purchase.groupby('Gender')['Purchase'].mean())
plt.subplot(1,2,1)

plt.pie(data_gender_purchase.groupby('Gender')['Purchase'].sum())

plt.legend(['Female','Male'],loc='best')

plt.title('Total Purchase Ammount')



plt.subplot(1,2,2)

plt.pie(data_gender_purchase.groupby('Gender')['Purchase'].mean())

plt.legend(['Female','Male'],loc='best')

plt.title('Average Purchase Ammount')
data_City_Category_purchase =data[['User_ID','City_Category','Purchase']]

data_City_Category_purchase=data_City_Category_purchase.groupby(['User_ID','City_Category']).sum()

data_City_Category_purchase.reset_index(inplace=True)

data_City_Category_purchase.head()
sb.countplot(data_City_Category_purchase["City_Category"])
print (data_City_Category_purchase.groupby('City_Category')['Purchase'].sum())

print (data_City_Category_purchase.groupby('City_Category')['Purchase'].mean())
plt.subplot(1,2,1)

plt.pie(data_City_Category_purchase.groupby('City_Category')['Purchase'].sum())

plt.legend(['A','B','C'],loc='best')

plt.title('Total Purchase Ammount')



plt.subplot(1,2,2)

plt.pie(data_City_Category_purchase.groupby('City_Category')['Purchase'].mean())

plt.legend(['A','B','C'],loc='best')

plt.title('Average Purchase Ammount')
data_City_Years_purchase =data[['User_ID','Stay_In_Current_City_Years','Purchase']]

data_City_Years_purchase=data_City_Years_purchase.groupby(['User_ID','Stay_In_Current_City_Years']).sum()

data_City_Years_purchase.reset_index(inplace=True)

data_City_Years_purchase.head()
sb.countplot(data_City_Years_purchase["Stay_In_Current_City_Years"])
plt.subplot(1,2,1)

plt.pie(data_City_Category_purchase.groupby('City_Category')['Purchase'].sum())

plt.legend(['A','B','C'],loc='best')

plt.title('Total Purchase Ammount')



plt.subplot(1,2,2)

plt.pie(data_City_Category_purchase.groupby('City_Category')['Purchase'].mean())

plt.legend(['A','B','C'],loc='best')

plt.title('Average Purchase Ammount')
# converting product id in to int

data['Product_ID']=data['Product_ID'].apply(lambda x:x[1:-2])

data['Product_ID']=(data['Product_ID']).astype(int)

data.head()
data_Product_ID_purchase =data[['Product_ID','Purchase']]

data_Product_ID_purchase=data_Product_ID_purchase.groupby(['Product_ID']).mean()

data_Product_ID_purchase.reset_index(inplace=True)

data_Product_ID_purchase.head(10)
plt.bar(x='Product_ID',height='Purchase',data=data_Product_ID_purchase)

plt.title('Mean Purchase for every product ID')
data.isnull().sum()
# First handling Categorical variables

data = pd.get_dummies(data, columns = ['Gender','Age','City_Category','Stay_In_Current_City_Years'],drop_first = True)

data.drop(['User_ID'],axis=1,inplace=True)
data.head()
plt.subplot(1,2,1)

plt.boxplot(data[data['Product_Category_2'].isnull()==False]['Product_Category_2'],showmeans=True)

plt.title('Before Filling missing values')



plt.subplot(1,2,2)

data['Product_Category_2']=data['Product_Category_2'].fillna(data['Product_Category_2'].median())

plt.boxplot(data['Product_Category_2'],showmeans=True)

plt.title('After Filling missing values')
sb.distplot(data[data['Product_Category_3'].isnull()==False]['Product_Category_3'])

plt.show()
plt.subplot(1,2,1)

plt.boxplot(data[data['Product_Category_3'].isnull()==False]['Product_Category_3'],showmeans=True)

plt.title('Before Filling missing values')



plt.subplot(1,2,2)

data['Product_Category_3']=data['Product_Category_3'].fillna(method='bfill')

data['Product_Category_3']=data['Product_Category_3'].fillna(method='ffill')

data.boxplot(column='Product_Category_3',showmeans=True)

plt.title('After Filling missing values')
data.isnull().sum()
sb.distplot(data['Purchase'])

plt.show()
sb.distplot(data['Occupation'])

plt.show()
#Using Random Forest Algo 

import sklearn

from sklearn.model_selection import  train_test_split 

from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import mean_squared_error
Y=data['Purchase']

X=data.drop(columns=['Purchase'])

# split the train and test dataset where test set is 30% of dataset

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3)
model= RandomForestRegressor(max_depth=15) 

model=model.fit(xtrain,ytrain) 
model.score(xtest,ytest)
ypred=model.predict(xtest)
rmse = np.sqrt(mean_squared_error(ytest, ypred))

rmse