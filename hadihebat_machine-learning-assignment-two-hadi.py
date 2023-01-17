# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")
#2. We make the shape (to know the number of rows and columns)





print("The train shape: ",train.shape)

print("The test shape: ", test.shape) # We pay attention that the dependent variable (sales prices) is not included, as we said before.
#3. Let’s take a look at the first data:

train.head()
#4. We describe the variable to be estimated, which will be our dependent variable (Y)



train.SalePrice.describe()
#4.prepare the data.

#a. A measure of the disparity or form (Skewness) of the variable to be estimated (SalesPrice).

#do this step to see if we need to do some kind of log conversion. As a “golden rule”,

#if we see that the Dependent variable presents asymmetry both to the left and to the right,

#will convert it to logarithmic.

print("Skew is: ", train.SalePrice.skew())

plt.style.use(style="ggplot")

plt.rcParams["figure.figsize"]=(10,6)

plt.hist(train.SalePrice,color="blue")

plt.show() #Positive asymmetry or to the right.
#b.convert the Y dependent variable to logarithmic, because it is asymmetric.



target=np.log(train.SalePrice)

print ("The value of the Y is :", target.skew())

plt.hist(target,color="blue")

plt.show()
#c. We take the variables that are numeric and store them in a Data Frame.

numeric_features=train.select_dtypes(include=[np.number])
#d. We measure the correlation between the columns.

numeric_features.dtypes
#5.We measure the correlation with the dependent variable SalesPrice:



corr=numeric_features.corr()

print (corr.SalePrice.sort_values(ascending=False)[:5],"\n") # Cogemos las cinco con mayor corrlación con SalesPrice

print (corr.SalePrice.sort_values(ascending=True)[:5]) # We take the five with the lowest correlation to SalesPrice
# Inquiring into the most correlated variable:



train.OverallQual.unique() #unique values of the variable
# We created a pivot table to investigate the relationship between that variable and SalePrice.



quality_pivot=train.pivot_table(index="OverallQual",values="SalePrice",aggfunc=np.median)

print(quality_pivot)
#We show it on the screen 



quality_pivot.plot(kind="bar",color="blue")

plt.xticks(rotation=0)

plt.show()
#We look at the next most correlated variable, GrLivArea. 

plt.scatter(x=train["GrLivArea"],y=target) 

plt.show()
# We watched GarageArea:



plt.scatter(x=train["GarageArea"], y =target)

plt.show()
#5.we eliminate Outliers, Null and Missing Values:

#a.We eliminate OUTLIERS. With the vector method, 

#  you get a series of pandas with index and True and then you pass a Panda DataFrame

train=train[train.GarageArea <1200]

print (train)

plt.scatter(x=train.GarageArea,y=np.log(train.SalePrice)) 

plt.show()
#b.NULL and MISSING VALUES

#Locating the null values:



null=pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

print (null)
print ("Unique values are:", train.MiscFeature.unique())

data=train.select_dtypes(include=[np.number]).interpolate().dropna()
#We check if all columns have 0 null values:



print (sum(data.isnull().sum())) # If the result is zero then there is no value that is 0
#a. We look at the categorical variables.



categoricals=train.select_dtypes(exclude=[np.number])

categoricals.describe()
#7.We build the model.

#a. We split the dependent v. (Y) and the independent v. (X):



y=np.log(train.SalePrice)

x=data.drop(["SalePrice","Id"], axis=1)
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=.33)
#c. Modelling.



lr=linear_model.LinearRegression()
#4. Model fitting. With the four parameters of X train, X_test, y_train, y_test



model=lr.fit(X_train,y_train)
#8.Model evaluation.



#a. Model performance evaluation.



print ("R² is: \n", model.score(X_test,y_test))
#b. Prediction:



predictions=model.predict(X_test)# Returns a list of predictions given a set of predictors
#c. Calculation of mean_squared_error (rmse). Measures the distance between our and actual values.



print ("RMSE is: \n", mean_squared_error(y_test,predictions))

actual_values=y_test

plt.scatter(predictions,actual_values,alpha=.75,color="b") 

plt.xlabel("Predicted Price") 

plt.ylabel("Actual Price")

plt.title("Linear Regression Model") 

plt.show()
submission=pd.DataFrame()

submission["ID"]=test.Id

feats=test.select_dtypes(include=[np.number]).drop(["Id"],axis=1).interpolate()

predictions=model.predict(feats)
final_predictions=np.exp(predictions)

print ("original predictions: \n",predictions[:5], "\n")

print ("Final predictions: \n", final_predictions[:5])

submission["SalePrice"]=final_predictions

#We modify the type of data of the first column to be integrated.



submission["ID"]=submission.ID.astype("int64")

print (submission)

#convert a CSV

submission.to_csv("submission1.csv",index=False)