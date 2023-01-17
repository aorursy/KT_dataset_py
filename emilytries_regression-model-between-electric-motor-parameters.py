# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.preprocessing import Imputer

from sklearn.model_selection import KFold

from sklearn import linear_model

from sklearn.metrics import make_scorer



from sklearn import svm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



from sklearn import neighbors

from math import sqrt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# read data

df = pd.read_csv('../input/pmsm_temperature_data.csv', 

                 usecols=[0,1,2,3,4,5,6,7,8,9,10,11])

df.head(10)
df.info()
df.describe().T
df.isnull().values.any()
# Count the number of NaNs each column has.

nans=pd.isnull(df).sum()

nans[nans>0]
# Count the column types

df.dtypes.value_counts()
df.corr()
import seaborn as sns

sns.jointplot(x='i_d', y='motor_speed', data=df, kind='reg')
#correlation map

f,ax=plt.subplots(figsize=(12,12))

corr=df.corr()



sns.heatmap(corr, annot=True, linewidths=.5, fmt='.2f', 

            mask= np.zeros_like(corr,dtype=np.bool), 

            cmap=sns.diverging_palette(100,200,as_cmap=True), 

            square=True, ax=ax)



plt.show()
import statsmodels.api as sm

#Defining dependet and independent variable

X = df['i_d']

X=sm.add_constant(X)



y = df['motor_speed']



lm=sm.OLS(y,X)

model=lm.fit()



model.summary()
model.params
print("f_pvalue:", "%.4f" % model.f_pvalue)
model.mse_model #mean squared error is too much. It is not good.
model.rsquared #Not bad
model.rsquared_adj #Not bad
model.fittedvalues[0:5] #Predicted values
y[0:5] #Real values
#Model equation

print("Motor speed = " + 

      str("%.3f" % model.params[0]) + ' + i_d' + "*" + 

      str("%.3f" % model.params[1]))
#Model Visualization 

g=sns.regplot(df['i_d'] , df['motor_speed'], 

              ci=None, scatter_kws={'color': 'r', 's':9})

g.set_title('Model equation: motor_speed = -0.002 + i_d * -0.725')

g.set_ylabel('Motor_speed')

g.set_xlabel('i_d');
from sklearn.metrics import r2_score,mean_squared_error



mse=mean_squared_error(y, model.fittedvalues)

rmse=np.sqrt(mse)

rmse
k_t=pd.DataFrame({'Real_values':y[0:50], 

                  'Predicted_values' :model.fittedvalues[0:50]})

k_t['error']=k_t['Real_values']-k_t['Predicted_values']

k_t.head()
model.resid[0:10] #It is easy way to learn residuals.
plt.plot(model.resid);
X=df.drop("motor_speed", axis=1)

y=df["motor_speed"]
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)



training=df.copy()
lm=sm.OLS(y_train, X_train)



model=lm.fit()

model.summary() #All coefficients are significant for the model by looking at the p-value. ( P>|t| )
#Root Mean Squared Error for Train

rmse1=np.sqrt(mean_squared_error(y_train,model.predict(X_train)))

rmse1
#Root Mean Squared Error for Test

rmse2=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))

rmse2
#Model Tuning for Multiple Linear Regression

model = LinearRegression().fit(X_train,y_train)

cross_val_score1=cross_val_score(model, X_train, y_train, cv=10, scoring='r2').mean() #verified score value for train model

print('Verified R2 value for Training model: ' + str(cross_val_score1))



cross_val_score2=cross_val_score(model, X_test, y_test, cv=10, scoring='r2').mean() #verified score value for test model

print('Verified R2 value for Testing Model: ' + str(cross_val_score2))
RMSE1=np.sqrt(-cross_val_score(model, X_train, y_train, cv=10, 

                               scoring='neg_mean_squared_error')).mean() #verified RMSE score value for train model

print('Verified RMSE value for Training model: ' + str(RMSE1))



RMSE2=np.sqrt(-cross_val_score(model, X_test, y_test, cv=10, 

                               scoring='neg_mean_squared_error')).mean() #verified RMSE score value for test model

print('Verified RMSE value for Testing Model: ' + str(RMSE2))
#Visualizing for Multiple Linear Regression y values



import seaborn as sns

ax1 = sns.distplot(y_train, hist=False, color="r", label="Actual Value")

sns.distplot(y_test, hist=False, color="b", label="Fitted Values" , ax=ax1);
from sklearn.decomposition import PCA

from sklearn.preprocessing import scale



pca=PCA()

X_reduced_train=pca.fit_transform(scale(X_train))
explained_variance_ratio=np.cumsum(np.round(pca.explained_variance_ratio_ , decimals=4)* 100)[0:20]
plt.bar(x=range(1, len(explained_variance_ratio)+1), height=explained_variance_ratio)

plt.ylabel('percentange of explained variance')

plt.xlabel('principal component')

plt.title('bar plot')

plt.show()

# 7 component is enough for model.
lm=LinearRegression()

pcr_model=lm.fit(X_reduced_train,y_train)

print('Intercept: ' + str(pcr_model.intercept_))

print('Coefficients: ' + str(pcr_model.coef_))
#Prediction

y_pred=pcr_model.predict(X_reduced_train)

np.sqrt(mean_squared_error(y_train,y_pred))
df['motor_speed'].mean()
#R squared

r2_score(y_train,y_pred)
# Prediction For testing error 

pca2=PCA()



X_reduced_test=pca2.fit_transform(scale(X_test))

pcr_model2=lm.fit(X_test,y_test)



y_pred=pcr_model2.predict(X_reduced_test)



print('RMSE for test model : ' +str(np.sqrt(mean_squared_error(y_test,y_pred))))
#Model Tuning for PCR



lm=LinearRegression()

pcr_model=lm.fit(X_reduced_train[:,0:10],y_train)

y_pred=pcr_model.predict(X_reduced_test[:,0:10])



from sklearn import model_selection



cv_10=model_selection.KFold(n_splits=10,

                           shuffle=True,

                           random_state=1)
lm=LinearRegression()

RMSE=[]



for i in np.arange(1,X_reduced_train.shape[1] + 1):

    score=np.sqrt(-1*model_selection.cross_val_score(lm,

                                                    X_reduced_train[:,:i],

                                                    y_train.ravel(),

                                                    cv=cv_10,

                                                    scoring='neg_mean_squared_error').mean())

    RMSE.append(score)
plt.plot(RMSE)

plt.xlabel('# of Components')

plt.ylabel('RMSE')

plt.title('PCR Model Tuning for Motor_Speed Prediction'); 
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import RandomForestRegressor



from warnings import filterwarnings

filterwarnings('ignore')
knn_model=KNeighborsRegressor().fit(X_train, y_train)

y_pred=knn_model.predict(X_test)
y_pred.shape
#Model Tuning (learning best n_neighbors hyperparameter)

knn_params={'n_neighbors' : np.arange(1,5,1)}



knn=KNeighborsRegressor()

knn_cv_model=GridSearchCV(knn, knn_params, cv=5)



knn_cv_model.fit(X_train,y_train)

knn_cv_model.best_params_["n_neighbors"]
# Train error values from n=1 up n=2

RMSE=[]

RMSE_CV=[]

for k in range(2):

    k=k+1

    knn_model=KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)

    y_pred=knn_model.predict(X_train)

    rmse=np.sqrt(mean_squared_error(y_train,y_pred))

    rmse_cv=np.sqrt(-1*cross_val_score(knn_model,X_train,y_train,cv=2,

                                       scoring='neg_mean_squared_error').mean())



    RMSE.append(rmse)

    RMSE_CV.append(rmse_cv)



    print("RMSE value: ", rmse, 'for k= ',k,

          "RMSE values with applying Cross Validation: ", rmse_cv)
#Model Tuning according to best parametre for KNN Regression

knn_tuned=KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])

knn_tuned.fit(X_train,y_train)

np.sqrt(mean_squared_error(y_test,knn_tuned.predict(X_test)))
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score,mean_squared_error
quad = PolynomialFeatures (degree = 2)

x_quad = quad.fit_transform(X_train)



X_train,X_test,y_train,y_test = train_test_split(x_quad,y_train, random_state = 0)



plr = LinearRegression().fit(X_train,y_train)



Y_train_pred = plr.predict(X_train)

Y_test_pred = plr.predict(X_test)



print('Polynomial Linear Regression:' ,plr.score(X_test,y_test))
#Plotting Residual in Linear Regression 



from sklearn import linear_model,metrics

#Create linear regression object

reg=linear_model.LinearRegression()



#train the model using the train data sets

reg.fit(X_train,y_train)



#regression coefficients

print("Coefficients: \n", reg.coef_)



#Variance score

print("Variance score: {}".format(reg.score(X_test,y_test)))



plt.style.use('fivethirtyeight')



#plotting residual errors in training data

plt.scatter(reg.predict(X_train),reg.predict(X_train)-y_train, 

            color="green", s=10, label="train data")



#plotting residual errors in test data

plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test, 

            color="blue", s=10, label="test data")



#plot line for zero residual error

plt.hlines(y=0,xmin=-2, xmax=2, linewidth=2)



#plot legend

plt.legend(loc='upper right')



#plot title

plt.title("residual error")



plt.show()