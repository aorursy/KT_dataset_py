# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#loading the initial packages that are commonly used before starting the python model

import pandas as pd  # for dataframe and data structure operations

import numpy as np  # for Array or matrix based operations

import matplotlib.pyplot as plt # Basic Math Calculations and plots or Visualization

%matplotlib inline
# Load the required input of train data

houseprice=pd.read_csv("../input/train.csv")
houseprice.shape # gives you the count of Number of Observations and Number of Variables
# make some data analysis

houseprice.head() # give you the top 5 observations details
houseprice.tail() # gives you the last 5 observations details
houseprice.describe() # displays the statistic details or descriptive statistics of each variable
houseprice.describe().transpose() # displays the statistic details of each variable in other way
houseprice.info() # gives you information of each variable data type
# Checking the Skewness for Variables

houseprice.skew()
# Checking the Kurtosis for Variables

houseprice.kurt()
# We know that the dependent variable is "SalePrice"

houseprice.SalePrice.describe()
#Histogram

houseprice.SalePrice.plot(kind="hist")
#boxplot

houseprice.SalePrice.plot(kind="box")
#boxplot

houseprice.SalePrice.plot(kind="box", vert=False)
# Density Curve

houseprice.SalePrice.plot(kind="density")
# The initial process for the data cleaning is to check null values or missing values

houseprice.isnull().sum()
# As you are unable to identify all the variables, divide the variables based on their data types

housepricenum=houseprice.select_dtypes(include=[np.number])

housepricecat=houseprice.select_dtypes(include=[object])
housepricenum.columns
housepricecat.columns
#now check missing values for all the numerical variables

housepricenum.isnull().sum()
housepricecat.isnull().sum()
# We can see that there are more % of missing values for Some variable

#So we can remaove them but instead of removing we are using them to our model by imputing later

# we Seperate those varialbes with more missing values to a sperate data frame

nonecols=housepricecat[["Alley","Fence","PoolQC","MiscFeature","FireplaceQu"]]
nonecols.shape
nonecols.isnull().sum()
# As we created a seperate dataframe for the above variables we have to remove them from housepricecat dataframe

# create a new dataframe

housepricecat1=housepricecat.drop(["Alley","Fence","PoolQC","MiscFeature","FireplaceQu"],axis=1)
# This is because the mentioned variables are categorical but are denoted as numeric

housepriceval=housepricenum[["MSSubClass","OverallCond","OverallQual","YearBuilt","YearRemodAdd","GarageYrBlt","MoSold","YrSold"]]
housepriceval.head()
# As we are creating a new dataframe for above variables, we have to remove those variables from data frame

# with numericl variables

housepricenum.shape
# we are make changes to existing df, so it is better to create a new df for changed data

housepricenum1=housepricenum.drop(["MSSubClass","OverallCond","OverallQual","YearBuilt","YearRemodAdd","GarageYrBlt","MoSold","YrSold"],axis=1)
housepricenum1.shape
# Check the missing values for all the data frames

housepricenum1.isnull().sum()
housepricecat1.isnull().sum()
housepriceval.isnull().sum()
nonecols.isnull().sum()
#Now its time to impute the null Values

# In our case we will impute the missing value for continuous or numeric with Mean.

for col in housepricenum1:

    housepricenum1[col].fillna(housepricenum1[col].mean(),inplace=True)
housepricenum1.isnull().sum()
# imputing the categorical varibles with repeated value

for col in housepricecat1:

    housepricecat1[col].fillna(housepricecat1[col].value_counts().idxmax(),inplace=True)
housepricecat1.isnull().sum()
for col in housepriceval:

    housepriceval[col].fillna(housepriceval[col].value_counts().idxmax(),inplace=True)
housepriceval.isnull().sum()
for col in nonecols:

    nonecols[col]=nonecols[col].fillna(value="No")
nonecols.isnull().sum()
# for transforming the data we have to use the lable Encoder

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
housepriceval1=housepriceval.apply(le.fit_transform)

housepricecols=nonecols.apply(le.fit_transform)

housepricecat2=housepricecat1.apply(le.fit_transform)
# Now combain all the dataframes into a single data frame

housepricetrain=pd.concat([housepricenum1,housepriceval1,housepricecols,housepricecat2],axis=1)

#the above is the final dataframe after data cleaning process is completed.

# This is the final train dataframe that we have to use for building the model
# Load the required input of train data

houseprice2=pd.read_csv("../input/test.csv")
houseprice2.shape
# As you are unable to identify all the variables, divide the variables of test data based on their data types

houseprice2num=houseprice2.select_dtypes(include=[np.number])

houseprice2cat=houseprice2.select_dtypes(include=[object])

nonecols2=houseprice2cat[["Alley","Fence","PoolQC","MiscFeature","FireplaceQu"]]

houseprice2cat1=houseprice2cat.drop(["Alley","Fence","PoolQC","MiscFeature","FireplaceQu"],axis=1)

houseprice2val=houseprice2num[["MSSubClass","OverallCond","OverallQual","YearBuilt","YearRemodAdd","GarageYrBlt","MoSold","YrSold"]]

houseprice2num1=houseprice2num.drop(["MSSubClass","OverallCond","OverallQual","YearBuilt","YearRemodAdd","GarageYrBlt","MoSold","YrSold"],axis=1)

#Now its time to impute the null Values

# In our case we will impute the missing value for continuous or numeric with Mean.

for col in houseprice2num1:

    houseprice2num1[col].fillna(houseprice2num1[col].mean(),inplace=True)
# imputing the categorical varibles with repeated value

for col in houseprice2cat1:

    houseprice2cat1[col].fillna(houseprice2cat1[col].value_counts().idxmax(),inplace=True)
for col in houseprice2val:

    houseprice2val[col].fillna(houseprice2val[col].value_counts().idxmax(),inplace=True)
for col in nonecols2:

    nonecols2[col]=nonecols2[col].fillna(value="No")
# for transforming the data we have to use the lable Encoder

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

houseprice2val1=houseprice2val.apply(le.fit_transform)

houseprice2cols=nonecols2.apply(le.fit_transform)

houseprice2cat2=houseprice2cat1.apply(le.fit_transform)

#the above is the final test dataframe after data cleaning process is completed.

# This is the final test  dataframe that we have to use for predictions
housepricetest=pd.concat([houseprice2num1,houseprice2val1,houseprice2cols,houseprice2cat2],axis=1)
# As Id variable doesnt show impact we are removing it

housepricetest=housepricetest.drop(["Id"],axis=1)
# Training Phase using train data

# First build the models using the train data

# Divide the Dependent (y) and Independent Variables (X)

y=np.log(housepricetrain.SalePrice)

# As SalePrice is not following a good normal distribution we are using log transformation

X=housepricetrain.drop(["Id","SalePrice"],axis=1) # As Id is just the identification number,SalePrice is y"
#package for linear model

from sklearn.linear_model import LinearRegression 
LinearReg=LinearRegression()
# fitting of the linear model

LinearReg.fit(X,y)
# for finding the accuracy

LinearReg.score(X,y)
# for cross validating the accuracy

from sklearn.model_selection import cross_val_score

cvs=cross_val_score(LinearReg,X,y,cv=10)
print("Accuracy: %0.2f (%0.2f)"%(cvs.mean(),cvs.std()))
# Prediction on train data

# As we used log transformation now we have go with exp transformation for SalePrice back to its original

LinearReg_pred=np.exp(LinearReg.predict(X))

LinearReg_pred
LRresid=housepricetrain.SalePrice-LinearReg_pred # for Residuals (Errors)
np.sqrt(np.mean((LRresid)**2)) # Root Mean Square Error
Linear_regtest=np.exp(LinearReg.predict(housepricetest))
Linear_regtest=pd.DataFrame(Linear_regtest)

Linear_regtest
#Import the package of Random Forest

from sklearn.ensemble import RandomForestRegressor

RF_reg=RandomForestRegressor()
#Fit the Model

RF_reg.fit(X,y)
# Model Accuracy

RF_reg.score(X,y)
# Predict of train

RF_regpredtrain=np.exp(RF_reg.predict(X))

RF_regpredtrain
# Predict on test data

RF_regpredtest=np.exp(RF_reg.predict(housepricetest))

RF_regpredtest
RF_regpredtest=pd.DataFrame(RF_regpredtest)

RF_regpredtest