# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
housing=pd.read_excel('/kaggle/input/housing.xlsx')
housing.head(n=10)
housing.describe()
housing.info()#Checking the type of data
housing.shape #Checking dimensions of dataset
housing.hist(edgecolor='black',bins=20,color='darkblue')
fig=plt.gcf()
fig.set_size_inches(13,9)
sns.set_style('darkgrid')
plt.show()
housing.isnull().sum() #207 null values in total_bedrooms(column).before that lets check for outliers in the dataset..
#Using a box plot
plt.figure(figsize=(18,6))
sns.boxplot(data=housing,y="total_bedrooms",orient="h", palette="Set3",linewidth=2.5)
sns.set_style("whitegrid")
plt.show()

X=housing.iloc[:,0:9].values
Y=housing.iloc[:,9].values
type(X)
type(Y)
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:,[4]])
X[:,[4]]=imputer.fit_transform(X[:,[4]])
df1=pd.DataFrame(X).values
df1.isnull().sum()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_df1=LabelEncoder()
df1[:,8]=labelencoder_df1.fit_transform(df1[:,8])
df2=pd.DataFrame(df1)
onehotencoder=OneHotEncoder(categories="auto")
df2=onehotencoder.fit_transform(df2).toarray()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#Standardizing the data
#from sklearn.preprocessing import StandardScaler
#sc_df2=StandardScaler()
#X_train=sc_df2.fit_transform(X_train)
#X_test=sc_df2.fit_transform(X_test)
#Performing the linear Regression

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
regressor.coef_ 
regressor.intercept_
regressor.score(X_train,Y_train)
predicted=regressor.predict(X_test)
expected=Y_test
regressor.score(X_train,Y_train)
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(expected,predicted))
plt.scatter(expected,predicted,color='blue')
plt.plot([0,600000],[-0,600000],'--k',color='red')
plt.title('California Housing Price Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Expected Price')
sns.set_style('darkgrid')
plt.show()
