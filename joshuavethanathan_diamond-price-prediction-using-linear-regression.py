# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

data=pd.read_csv('/kaggle/input/diamonds/diamonds.csv',index_col=0)

data.head()
plt.figure(figsize=(20,6))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(data.corr(), annot=True,cmap='RdYlGn',square=True) 
sns.pairplot(data)

plt.grid()

plt.show()
data.info()


print(data['x'].mean())

print(data['x'].median())

print(data['y'].mean())

print(data['y'].median())

print(data['z'].mean())

print(data['z'].median())
data['z']=data['z'].replace(0,data['z'].mean())

data['x']=data['x'].replace(0,data['x'].mean())

data['y']=data['y'].replace(0,data['y'].mean())
num_features=['carat','depth','table','price','x','y','z']

for feature in num_features:

    print(feature)

    ax=sns.distplot(data[feature])

    plt.show() 
num_features=['carat','depth','table','price','x','y','z']

for feature in num_features:

    data[feature]=np.log(data[feature])

    #ax=sns.distplot(data[feature])

    #plt.show() 

    

data.head()
print(data['cut'].value_counts())

print(data['color'].value_counts())

print(data['clarity'].value_counts())
categorical_feat=['cut','color','clarity']

for feature in categorical_feat:

    le=LabelEncoder();

    le.fit(data[feature])

    print(le.classes_)

    data[feature]=le.transform(data[feature])

data.head()

feature_scale=[feature for feature in data.columns if feature not in ['price']]

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(data[feature_scale])

data[feature_scale]=scaler.transform(data[feature_scale])

data.head()
y=data['price']

x=data.drop(['price'],axis=1)

x.head()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm

print(regressor.intercept_)

print(regressor.coef_)
#### Predcit the results using the created model ####

y_pred=regressor.predict(X_test)

data_result=pd.DataFrame({'Actual':y_test,'Predict':y_pred})

data_result.head(10)
print("accuracy: "+ str(regressor.score(X_test,y_test)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(y_test,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(y_test,y_pred)))

R2 = r2_score(y_test,y_pred)

print('R Squared: {}'.format(R2))

n=X_test.shape[0]

p=X_test.shape[1] - 1



adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))
import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std



model1=sm.OLS(y_train,X_train)



result=model1.fit()



y_pred_ols=result.predict(X_test)
data_result=pd.DataFrame({'Actual':y_pred_ols,'Predict':y_pred})

data_result.tail()
print("accuracy: "+ str(regressor.score(X_test,y_test)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(y_pred_ols,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(y_pred_ols,y_pred)))

R2 = r2_score(y_pred_ols,y_pred)

print('R Squared: {}'.format(R2))

n=X_test.shape[0]

p=X_test.shape[1] - 1



adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))