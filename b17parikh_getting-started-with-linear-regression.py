import pandas as pd

from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression

from sklearn.model_selection import train_test_split

import statsmodels.api as sm 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df =pd.read_csv('../input/admission-prediction/Admission_Prediction.csv')

df.head()
df.describe(include='all')
df.isna().sum()
df['University Rating'] = df['University Rating'].fillna(df['University Rating'].mode()[0])

df['TOEFL Score'] = df['TOEFL Score'].fillna(df['TOEFL Score'].mean())

df['GRE Score']  = df['GRE Score'].fillna(df['GRE Score'].mean())
df= df.drop(columns = ['Serial No.'])

df.head()
df
plt.figure(figsize=(20,25), facecolor='white')

plotnumber = 1



for column in df:

    if plotnumber<=16 :

        ax = plt.subplot(4,4,plotnumber)

        sns.distplot(df[column])

        plt.xlabel(column,fontsize=20)

        #plt.ylabel('Salary',fontsize=20)

    plotnumber+=1

plt.tight_layout()
y = df['Chance of Admit']

X =df.drop(columns = ['Chance of Admit'])
plt.figure(figsize=(20,30), facecolor='white')

plotnumber = 1



for column in X:

    if plotnumber<=15 :

        ax = plt.subplot(5,3,plotnumber)

        plt.scatter(X[column],y)

        plt.xlabel(column,fontsize=20)

        plt.ylabel('Chance of Admit',fontsize=20)

    plotnumber+=1

plt.tight_layout()
scaler =StandardScaler()



X_scaled = scaler.fit_transform(X)
from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = X_scaled



# we create a new data frame which will include all the VIFs

# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)

# we do not include categorical values for mulitcollinearity as they do not provide much information as numerical ones do

vif = pd.DataFrame()



# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 

vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]

# Finally, I like to include names so it is easier to explore the result

vif["Features"] = X.columns

vif
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=355)
regression = LinearRegression()



regression.fit(x_train,y_train)
regression.score(x_train,y_train)
def adj_r2(x,y):

    r2 = regression.score(x,y)

    n = x.shape[0]

    p = x.shape[1]

    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    return adjusted_r2
adj_r2(x_train,y_train)
regression.score(x_test,y_test)
adj_r2(x_test,y_test)
lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)

lasscv.fit(x_train, y_train)
alpha = lasscv.alpha_

alpha
lasso_reg = Lasso(alpha)

lasso_reg.fit(x_train, y_train)
lasso_reg.score(x_test, y_test)
import numpy as np

alphas = np.random.uniform(low=0, high=10, size=(50,))

ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)

ridgecv.fit(x_train, y_train)
ridgecv.alpha_
ridge_model = Ridge(alpha=ridgecv.alpha_)

ridge_model.fit(x_train, y_train)

ridge_model.score(x_test, y_test)
elasticCV = ElasticNetCV(alphas = None, cv =10)



elasticCV.fit(x_train, y_train)

elasticCV.alpha_
elasticnet_reg = ElasticNet(alpha = elasticCV.alpha_,l1_ratio=0.5)

elasticnet_reg.fit(x_train, y_train)

elasticnet_reg.score(x_test, y_test)