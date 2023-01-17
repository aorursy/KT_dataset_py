# Numerical libraries
import numpy as np

# to handle data in form of rows and columns 
import pandas as pd

from sklearn import preprocessing

# Import Linear Regression machine learning library
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score


#importing ploting libraries
import matplotlib.pyplot as plt

#importing seaborn for statistical plots
import seaborn as sns
#read the data and store it in data frame
df = pd.read_csv('/kaggle/input/autompg-dataset/auto-mpg.csv')
df
#the car name feature is not helping so we can drop it.
df = df.drop('car name',axis=1)
#just drop the Nan records
df.dropna()
#replace special character or junk data
df = df.replace('?',np.nan)
#replace the Nan value with the Mean value
df = df.apply(lambda x: x.fillna(x.median()),axis=0)
#Separate independent and dependent variables

# Copy all the predictor variables into X dataframe. Since 'mpg' is dependent variable drop it
X = df.drop('mpg',axis=1)

# Copy the 'mpg' column alone into the y dataframe. This is the dependent variable
y = df[['mpg']]
# scale all the columns of df. This will produce a numpy array
X_scaled = preprocessing.scale(X)
X_scaled = pd.DataFrame(X_scaled,columns=X.columns)

y_scaled = preprocessing.scale(y)
y_scaled = pd.DataFrame(y_scaled,columns=y.columns)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,test_size=0.30,random_state=1)
reg_model = LinearRegression()
reg_model.fit(X_train,y_train)

for idx,col_name in enumerate(X_train.columns):
    print('Coefficient for {} is {}'.format(col_name,reg_model.coef_[0][idx]))
intercept = reg_model.intercept_[0]
print('The intercept of the model is {}'.format(intercept))
ridge = Ridge(alpha=0.3)
ridge.fit(X_train,y_train)

print('Ridge model',(ridge.coef_))
lasso = Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
print('Lasso coefficeient :',lasso.coef_)
print("Linear Regression Training score is {}".format(reg_model.score(X_train,y_train)))
print("Linear Regression Training score is {}".format(reg_model.score(X_test,y_test)))
print("Redge Training model score is {}".format(ridge.score(X_train,y_train)))
print("Redge Test model score is {}".format(ridge.score(X_test,y_test)))
print("Lasso Training model score is {}".format(lasso.score(X_train,y_train)))
print("Lasso Test model score is {}".format(lasso.score(X_test,y_test)))
#import the required library
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2,interaction_only=True)

X_poly = poly.fit_transform(X_scaled)
X_poly.shape
#create train and test model using new X dataframe    

X_poly = poly.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.30, random_state=1)
X_test.shape
reg_model.fit(X_train,y_train)
print(reg_model.coef_[0])
ridge = Ridge(alpha=0.3)
ridge.fit(X_train,y_train)
print("Ridge model coefficient is {}".format(ridge.coef_))
print("Train model accuracy :" ,
      ridge.score(X_train,y_train))
print("Test model accuracy :" ,
      ridge.score(X_test,y_test))
lasso = Lasso(alpha=0.3)
lasso.fit(X_train,y_train)
print("Lasso model coefficient is {}".format(lasso.coef_))
print("Train model accuracy :" ,
      lasso.score(X_train,y_train))
print("Test model accuracy :" ,
      lasso.score(X_test,y_test))
