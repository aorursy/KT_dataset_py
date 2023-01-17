import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Data Preprocessing

housedata = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")
data = housedata.drop(['id','date'], axis = 1)
housedata.info()
#print(housedata['price'].describe())

plt.figure(figsize = (10,5))
sns.distplot(housedata['price'])

plt.figure(figsize = (20,10))
house_info = ['price' ,'bedrooms','bathrooms', 'sqft_living','sqft_lot', 'floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long', 'sqft_living15','sqft_lot15']
mycorr = housedata[house_info].corr()  
sns.heatmap(mycorr, square=False, linewidths=.5, annot=True) 

print(mycorr["price"].sort_values(ascending=False))


#Train & test data

x = data.drop('price',axis=1)
y = data['price']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=21)


#Model training
def Linear(X_train, X_test, y_train, y_test):    
    from sklearn.linear_model import LinearRegression 
    from sklearn import metrics
    lr = LinearRegression()
    #Model testing
    lr.fit(X_train, y_train)
    lr_pre = lr.predict(X_test)
    #Evaluation
    mse = metrics.mean_squared_error(y_test,lr_pre)
    return (mse/10000)

print('LR mse: ',Linear(X_train, X_test, y_train, y_test))
