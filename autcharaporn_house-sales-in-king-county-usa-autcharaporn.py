import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns #for data visualization

import matplotlib.pyplot as plt  #for data visualization
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')

df.head()
df.shape
df.info()
X = df.iloc[:, 0:-1]

y = df.iloc[:, -1]
X.drop(['id', 'date'], axis=1, inplace=True)

X.head()
#Splitting the dataset into the Training set and Test set



from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)
#Fitting Multiple Linear Regression to the Training Set



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
#Predicting the Test Set Results



y_pred = regressor.predict(X_test)
#Building the optimal model using Backward Elimination



import statsmodels.api as sm

X = np.append(arr=np.ones((21613,1)).astype(int), values = X, axis=1)
print(X)
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]] #Initialize the metrix

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
#cut x15 becuase high P>|t|



X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
#cut x13 becuase high P>|t|



X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,18]] 

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
#cut x14 becuase high P>|t|



X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,16,17,18]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
#cut x12 becuase high P>|t|



X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,16,17,18]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
#cut x8 becuase high P>|t|



X_opt = X[:,[0,1,2,3,4,5,6,7,9,10,11,16,17,18]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
#cut x11 becuase high P>|t|



X_opt = X[:,[0,1,2,3,4,5,6,7,9,10,16,17,18]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
#cut x10 becuase high P>|t|



X_opt = X[:,[0,1,2,3,4,5,6,7,9,16,17,18]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
#cut x9 becuase high P>|t|



X_opt = X[:,[0,1,2,3,4,5,6,7,16,17,18]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
print("Are There Missing Data? :",df.isnull().any().any())

print(df.isnull().sum())
print("\n\nPrice in Dataset:\n")

print("There are {} different values\n".format(len(df.price.unique())))

print(df.price.unique())
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)

plt.show()
dataframe=pd.pivot_table(df, index = 'condition', values=["view","bedrooms"])



#to normalize

dataframe["view"]=dataframe["view"]/max(dataframe["view"])

dataframe["bedrooms"]=dataframe["bedrooms"]/max(dataframe["bedrooms"])

sns.jointplot(dataframe.bedrooms,dataframe.view,kind="kde",height=5,space=0)

plt.savefig('graph.png')

plt.show()
#Linear regression with marginal distributions



g = sns.jointplot("view","bedrooms", data=dataframe,height=5,kind="reg",ratio=3, color="purple")
sns.pairplot(df[['view','bedrooms']],height=5)

plt.show()