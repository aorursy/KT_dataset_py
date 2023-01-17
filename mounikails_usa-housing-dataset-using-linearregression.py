import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as smf

from scipy.stats import shapiro,levene

from sklearn.model_selection import train_test_split
data=pd.read_csv("../input/USA_Housing.csv")

data.head()

# drop the Address column

data = data.drop('Address', axis=1)

#checking the head of the data

data.head()
#describing the data

data.describe()
data.info()
#checking the shape of the data

data.shape
data.isnull().sum()
data.corr()
sns.heatmap(data.corr(),annot=True)
sns.pairplot(data,diag_kind='kde')

plt.show()
data.columns
# renaming the columns

data.rename(columns={'Avg. Area Income':'Area_Income','Avg. Area House Age':'Area_House_Age','Avg. Area Number of Rooms':'Area_Number_of_Rooms','Avg. Area Number of Bedrooms':'Area_Number_of_Bedrooms','Area Population':'Area_Population'},inplace=True)
data.columns
X = data[data.columns[0:-1]]

Y = data["Price"]

X.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_std =  sc.fit_transform(X)
model=smf.ols('Price~Area_Income+Area_House_Age+Area_Number_of_Rooms+Area_Number_of_Bedrooms+Area_Population',data).fit()

model.summary()
#refitting  the model

model=smf.ols('Price~Area_Income+Area_House_Age+Area_Number_of_Rooms+Area_Population',data).fit()

model.summary()
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X=data.drop(['Area_Number_of_Bedrooms','Price'],axis=1)

Y=data['Price']

Xtrain,Xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0)

linreg=LinearRegression()

linreg.fit(X_train,Y_train)
ypred=linreg.predict(X_test)

ypred
from sklearn.model_selection import KFold

import sklearn.metrics as metrics

from sklearn.metrics import roc_curve,auc
X.head()
Y.head()
kf=KFold(n_splits=5,shuffle=True,random_state=2)

root=[]

lst=[]

for train,test in kf.split(X,Y):

    linreg=LinearRegression()

    X_train,X_test=X.iloc[train,:],X.iloc[test,:]

    Y_train,Y_test=Y.iloc[train],Y.iloc[test]

    linreg.fit(X_train,Y_train)

    ypred=linreg.predict(X_test)

    root.append(np.sqrt(metrics.mean_squared_error(Y_test,ypred)))

    lst.append(linreg.score(X_train,Y_train))

    

print('Cross Validation Mean rmse is %1.2f'%np.mean(root))

print('Cross Validation Variance of rmse is %1.5f'%np.var(root,ddof=1))

print('Cross Validation Mean R square is %1.2f'%np.mean(lst))

print('Cross Validation Variance of R square is %1.5f'%np.var(lst,ddof=1))