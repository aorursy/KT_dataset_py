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

from sklearn.linear_model import LinearRegression 

from sklearn import preprocessing, model_selection



df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv",parse_dates=True)



df.drop(["id","date"],axis=1,inplace=True)

display(df.head(3))



df.shape
df.info()

#Dataset is clean of null values
#checking correlations

fig,ax = plt.subplots(figsize=(25, 15))

sns.heatmap(df.corr(),annot=True,ax=ax)
#preparing X,y

y =df["price"]

X = df.drop("price",axis=1)

features = list(X.columns)



display(X.shape,y.shape)

display(X.head(3))

display(y.head(3))





#optional but makes gradient descent faster



#X = preprocessing.scale(X)

#y = preprocessing.scale(y)



#train test split



X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=11)

#lr is our linear regression classifier



lr = LinearRegression(fit_intercept=True, n_jobs=1, normalize=False)



lr.fit(X_train , y_train)

Accuracy = lr.score(X_test,y_test)

print(str(Accuracy*100) +" %" )



pd.DataFrame({"Theta values":lr.coef_},index = features )
y_predict = lr.predict(X_test)



sns.scatterplot(y_predict,y_test)





plt.hist(y_predict-y_test,bins=60)

plt.show()


from sklearn import metrics

RMSE = metrics.mean_squared_error(y_predict,y_test,squared=False)

MSE = RMSE**2

MAE = metrics.mean_absolute_error(y_predict,y_test)

print( "Accuracy: " + str(accuracy*100) +" %" )

print("MSE : "+str(MSE))

print("RMSE : "+str(RMSE))

print("MAE : "+str(MAE))
#ELASTIC NET r=1 essentialy ridge



from sklearn.linear_model import ElasticNet

from sklearn import metrics



elastic_reg = ElasticNet(alpha=0.1,l1_ratio=1,random_state=11)

elastic_reg.fit(X_train,y_train)

Accuracy = elastic_reg.score(X_test,y_test)









print( "Accuracy: " + str(Accuracy*100) +" %" )

#polynomial conversion + plain linear regression - ----- Most accurate



from sklearn import preprocessing

poly_features = preprocessing.PolynomialFeatures(degree=2)

X_poly = poly_features.fit_transform(X)



#resplitting data after polynomial conversion

X_train,X_test,y_train,y_test = model_selection.train_test_split(X_poly,y,test_size=0.2,random_state=11)



lr = LinearRegression(fit_intercept=True, n_jobs=1, normalize=False)



lr.fit(X_train , y_train)

Accuracy = lr.score(X_test,y_test)



print( "Accuracy: " + str(Accuracy*100) +" %" )

#polynomial conversion + Elasticnet (Regularized linear regression)



from sklearn import preprocessing

poly_features = preprocessing.PolynomialFeatures(degree=2)

X_poly = poly_features.fit_transform(X)



#resplitting data after polynomial conversion

X_train,X_test,y_train,y_test = model_selection.train_test_split(X_poly,y,test_size=0.2,random_state=11)



elastic_reg = ElasticNet(alpha=0.1,l1_ratio=0.5,random_state=11,max_iter=2000)

elastic_reg.fit(X_train,y_train)

Accuracy=elastic_reg.score(X_test,y_test)





print( "Accuracy: " + str(Accuracy*100) +" %" )
