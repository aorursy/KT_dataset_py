import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv('../input/autompg-dataset/auto-mpg.csv')

data.head()
#Describe the dataset

data.describe(include = 'all').transpose()
#Car name column can be dropped as this column cannot be fit into model

data.drop('car name', axis = 1, inplace = True)



#Origin column can be renamed based on metadata available

data['origin'] = data['origin'].replace({1:'america',2:'europe',3:'asia'})



#Convert all categorical variable in Origin into binary column -> Using One Hot Encoding

data = pd.get_dummies(data, columns = ['origin'])



data.head()
#Horsepower column should only contain digits but here it contains text too, so first we'll check how many rows have text in them



temp = pd.DataFrame(data['horsepower'].str.isdigit())

temp[temp['horsepower'] == False]
data.iloc[126]
#Replace '?' with NaN

data = data.replace('?', np.nan)

data[data.isnull().any(axis = 1)]
data.median()
#instead of dropping the rows, lets replace the missing values with median value.

data = data.apply(lambda x: x.fillna(x.median()), axis = 0)

data.info()
#horsepower column should have it's datatype changed from object to float

data['horsepower'] = data['horsepower'].astype('float64')

data.head()
sns.pairplot(data, diag_kind = 'kde') #kde helps to plot kernel density estimation instead of default histogram
plt.figure(figsize = (10,8))

sns.heatmap(data.corr(), annot = True)
#Prepare X, y with 'mpg' column being the target column



X = data.drop('mpg', axis = 1)

y = data[['mpg']]
#Let us break the X and y dataframes into training set and test set. For this we will use

#Sklearn package's data splitting function which is based on random function

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)



# Invoke LinearRegression function to find the best fit model on training data

from sklearn.linear_model import LinearRegression

reg_model = LinearRegression()

reg_model.fit(X_train,y_train)
# Let us explore the coefficients for each of the independent attributes



for idx, col_name in enumerate(X_train.columns):

    print("The coefficient for {} is {}".format(col_name, reg_model.coef_[0][idx]))

    

# Model intercept

intercept = reg_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))
# Model score - R2 or coeff of determinant

# R^2=1–RSS / TSS



reg_model.score(X_test, y_test)
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model



poly = PolynomialFeatures(degree=2, interaction_only=True)

X_train_p = poly.fit_transform(X_train)



X_test_p = poly.fit_transform(X_test)



poly_clf = linear_model.LinearRegression()



poly_clf.fit(X_train_p, y_train)



y_pred = poly_clf.predict(X_test_p)



#print(y_pred)



print(poly_clf.score(X_test_p, y_test))

#Shape of orginal X, X_train & X_train_p



print(X.shape)

print(X_train.shape)

print(X_train_p.shape)