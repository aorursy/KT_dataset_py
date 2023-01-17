# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics,model_selection,preprocessing

from sklearn.linear_model import LinearRegression

df = pd.read_csv("/kaggle/input/usa-housing/USA_Housing.csv")



df.drop("Address",axis=1,inplace=True)

display(df.head())

display(df.shape)
fig,ax = plt.subplots(figsize=(10,8))

sns.heatmap(df.corr(),annot=True,ax=ax)



y = df["Price"]

X = df.drop("Price",axis=1)



features = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']



X = X[features]



X_train, X_test, y_train,y_test = model_selection.train_test_split(X,y,random_state=11,test_size=0.2)





lr = LinearRegression()

lr.fit(X_train,y_train)



accuracy = lr.score(X_test,y_test)



print("Accuray: "+str(accuracy*100))





sns.pairplot(df)

plt.show()
y_predict = lr.predict(X_test)



sns.distplot(y_predict-y_test)  #error distribution



#approximately random gaussian noise
df_coef = pd.DataFrame({"coeffiecients":lr.coef_},index = features)

df_coef
rmse = metrics.mean_squared_error(y_test,y_predict,squared=False)

display(rmse)



r2 = metrics.r2_score(y_test,y_predict)





#Ridge Regression



from sklearn.linear_model import Ridge



ridge_reg = Ridge(alpha=1, solver="cholesky",fit_intercept=True)

ridge_reg.fit(X_train, y_train)

ridge_reg.score(X_test,y_test)











#Lasso regression



from sklearn.linear_model import Lasso



lasso_reg = Lasso(alpha=0.001)

lasso_reg.fit(X, y)

lasso_reg.score(X_test,y_test)
#Elastic net



from sklearn.linear_model import ElasticNet

elastic_reg = ElasticNet(alpha=0.1,l1_ratio=0.7)      #liratio is r (coeff for l1)

elastic_reg.fit(X_train,y_train)

elastic_reg.score(X_test,y_test)

from sklearn.preprocessing import PolynomialFeatures



poly_reg = PolynomialFeatures(degree=2)   #choose degree of polynomial here

X_poly = poly_reg.fit_transform(X)



X_train, X_test, y_train, y_test = model_selection.train_test_split(X_poly, y, test_size=0.2, random_state=101)



lin_reg = LinearRegression(normalize=True)

lin_reg.fit(X_train,y_train)

lin_reg.score(X_test,y_test)




