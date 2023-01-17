#importing libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#using seaborn for styling

sns.set()
#Reading csv values and converting it into DataFrames

Train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

Test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
#finding the number of row and column in Test dataset

Test.shape
#finding the number of row and column in Test dataset

Train.shape
#Understanding the structure of test dataset

Train.head(5)
#Exploring the name of categorical variables in the train dataset

Train.select_dtypes( include = "object").columns
#Finding out the number of categorical Variables in train dataset

Train.select_dtypes(include = "object").columns.shape
#Exploring the name of Numerical variables in the dataset

Train.select_dtypes( exclude ="object").columns
#Finding out the number of categorical Variables in train dataset

Train.select_dtypes( exclude ="object").columns.shape
#Here is a shortcut just use info method to find out non -null count,Dtype

Test.info()
#using describe to know descriptive stats

#round function is used to round upto 2 decimals

#transpose is used for interchanging rows into columns or vice versa

Train.describe().transpose().round(2)
#By default describe only show numerical attributes ,show we need to use include 

#top shows the most occured instance in that column , Freq is its frequency

Train.describe(include = "object").transpose()
#Exploring the SalePrice which is our dependent variable which needs to be predicted

Train["SalePrice"].describe()
#Plotting density plot with distplot

#kde is kernel density estimationn,a method to estimate the density plot

sns.distplot(Train["SalePrice"],hist = True ,rug = True,

            kde = True )

plt.ylabel("Density")


print('Skewness: %f' % Train["SalePrice"].skew())

print("Kurtosis : %f" %Train["SalePrice"].kurtosis())
#scatter plot grlivarea/saleprice

plt.scatter( y = Train["SalePrice"],x = Train["GrLivArea"]

          ,alpha = 0.5, color = "Red")

plt.ylabel("SalePrice")

plt.xlabel("GrlivArea")

plt.title("Scatter Plot")
#scatter plot totalbsmtsf/saleprice

plt.scatter( y= Train["SalePrice"] , 

            x = Train["TotalBsmtSF"], alpha = 0.5)

plt.ylabel("SalePrice")

plt.xlabel("TotalBsmtSF")

#box plot overallqual/saleprice

fig = sns.boxplot(y = Train["SalePrice"] , x =

           Train["OverallQual"]

        ,width=0.5, fliersize=5)
#SalePrice/YearBuilt boxplot

f,ax = plt.subplots(figsize = (15,8))

fig = sns.boxplot(y =  Train["SalePrice"], x=Train["YearBuilt"]

           ,width = 0.5)

x = plt.xticks(rotation = 90)

y = plt.yticks(rotation = 90)

plt.savefig("sample.pdf")

#Drawing the correlation matrix using heatmap

#Correlation measures how much a variable is correlated to another variable

corrmat = Train.corr()

plt.subplots(figsize = (12,9))

sns.heatmap(corrmat,cmap="YlGnBu",vmax = 0.8, square= 1)
#saleprice correlation matrix

#10(in nlargest()func )is the number of variables for heatmap

cols = corrmat.nlargest(10,"SalePrice")["SalePrice"].index

cm = np.corrcoef(Train[cols].values.T)

ax = sns.heatmap(cm ,square = True,cbar = 1, xticklabels= cols.values,

           yticklabels = cols.values , annot = True

           , fmt = ".2f" , annot_kws = { "size": 9})

plt.show()

#Pair plot for top 10 correlated variables

cols = ["SalePrice" , "OverallQual","GrLivArea",

        "GarageCars" , "TotalBsmtSF" , "FullBath" ,

       "YearBuilt"]

sns.pairplot(Train[cols])

plt.savefig("pairplot.pdf")

plt.show()
#missing data of Train

total = Train.isnull().sum().sort_values( ascending = False)

percent = (Train.isnull().sum()/Train.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total , percent],axis = 1 , keys = ["Total","Percent"])

missing_data.head(20)
#missing data of Test

total1 = Test.isnull().sum().sort_values( ascending = False)

percent1 = (Test.isnull().sum()/Test.isnull().count()).sort_values(ascending = False)

missing_data1 = pd.concat([total1, percent1],axis = 1 , keys = ["Total1","Percent1"])

missing_data1[missing_data1["Total1"]>0]

#Deleting all the columns in train dataset which have more than one  missing value



Train =  Train.drop(missing_data[missing_data["Total"]>1].index, axis = 1)
#Deleting all the columns in test dataset which we have deleted in the train dataset

missing_data[missing_data["Total"]>1].index

Test = Test.drop(labels = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage',

       'GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual',

       'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual',

       'MasVnrArea', 'MasVnrType'] , axis = 1)
#Dealing with the missing numerical and categorical data of test dataset separately.

#Filling the missing value of categorical data with the mode of that columns.



Test['MSZoning']=Test['MSZoning'].fillna(Test['MSZoning'].mode()[0])

Test['Functional']=Test['Functional'].fillna(Test['Functional'].mode()[0])

Test['Utilities']=Test['Utilities'].fillna(Test['Utilities'].mode()[0])

Test['Exterior2nd']=Test['Exterior2nd'].fillna(Test['Exterior2nd'].mode()[0])

Test['KitchenQual']=Test['KitchenQual'].fillna(Test['KitchenQual'].mode()[0])

Test['SaleType']=Test['SaleType'].fillna(Test['SaleType'].mode()[0])

Test['Exterior1st']=Test['Exterior1st'].fillna(Test['Exterior1st'].mode()[0])



#Filling the missing numerical value with its mean

for num_col in ['BsmtHalfBath', 'BsmtFullBath', 'GarageCars', 'GarageArea','BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2']:

    Test[num_col]= Test[num_col].fillna(Test[num_col].mean())

#Standardization of SalePrice 

from sklearn.preprocessing import StandardScaler

std = StandardScaler()

saleprice_scaled = std.fit_transform(Train["SalePrice"].values.reshape(-1,1))

#To know the variation in the SalePrice

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10] # Ten Lowest saleprice of house

high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]# Ten Largest saleprice of house

#Finding the outliers in the GrLivArea

fig = plt.scatter(x = Train["GrLivArea"] , y = Train["SalePrice"])

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")
#Finding out the index of the outliers

Train["GrLivArea"].sort_values(ascending = False)[:2]
#Deleting the outliers

Train = Train.drop( index = 1298 , axis = 0)

Train = Train.drop(index = 523,axis = 0)



#Now that's good ,not outliers

plt.scatter(x = Train["GrLivArea"] , y = Train["SalePrice"])

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")
#Finding out the outliers in TotalBsmtSf

plt.scatter(x = Train["TotalBsmtSF"] ,

            y = Train["SalePrice"])

plt.xlabel("TotalBsmtSf")

plt.ylabel("SalePrice")
#Creating a new column "HasBsmt" in the train dataset which hold binary variable (1 0r 0) on the basis that the house have basement or not.

#Creating a series having the same index as train dataset

Train["HasBsmt"] = pd.Series(len(Train["TotalBsmtSF"]),index = Train.index)

#Filling all the rows of "HasBsmt" column with zero

Train["HasBsmt"] = 0

#filling  the rows of "HasBsmt" with 1 where "TotalBsmt" is greater than zero.

Train.loc[Train["TotalBsmtSF"]>0,"HasBsmt"] = 1



#Doing all the same thing with Test Dataset

Test["HasBsmt"] = pd.Series(len(Test["TotalBsmtSF"]),index = Test.index)

Test["HasBsmt"] = 0

Test.loc[Test["TotalBsmtSF"]>0,"HasBsmt"] = 1

#Plotting the density plot of SalePrice 

from scipy import stats

from scipy.stats import norm

fig = sns.distplot(Train["SalePrice"],fit = norm)

plt.ylabel("Density")

plt.show()

#Plotting the probability plot

res = stats.probplot(Train["SalePrice"], plot = plt)

#Logrithmic Transformation of SalePrice

Train["SalePrice"] = np.log(Train["SalePrice"])

f = stats.probplot(Train["SalePrice"] , plot = plt)
#After neccesary transformation , we can say that SalePrice is normally distributed.

y = sns.distplot(Train["SalePrice"] , fit = norm)
#Prob plot and density plot of GrLivArea

f= sns.distplot(Train["GrLivArea"], fit = norm)

fig = plt.figure()

x= stats.probplot(Train["GrLivArea"],plot = plt)
#Logirthmic Transformation on Both Train and Test Dataset

Train["GrLivArea"] = np.log(Train["GrLivArea"])

Test["GrLivArea"] = np.log(Test["GrLivArea"])
#After Transformation

fig = sns.distplot(Train["GrLivArea"], fit = norm)

fig = plt.figure()

f = stats.probplot(Train["GrLivArea"], plot = plt)
#Doing the same with TotalBsmtSF

sns.distplot(Train["TotalBsmtSF"],fit = norm)

fig = plt.figure()

r = stats.probplot(Train["TotalBsmtSF"],plot = plt)
#Doing the logarithmic Transformation for all the non zeros valuse in the TotalBsmtSF column for both the 

Train.loc[Train["HasBsmt"]==1,'TotalBsmtSF'] = np.log(Train["TotalBsmtSF"])

Test.loc[Test["HasBsmt"]==1,'TotalBsmtSF'] = np.log(Test["TotalBsmtSF"])

sns.distplot(Train[Train["HasBsmt"]>0]["TotalBsmtSF"], fit = norm)

fig = plt.figure()

f = stats.probplot(Train[Train["TotalBsmtSF"]>0]["TotalBsmtSF"],plot = plt)
#Concating both the dataset into single DataSet

final_df=pd.concat([Train,Test],axis=0)
final_df
final_df.shape
#Finding all the name of categorical varibles"

Train.select_dtypes(include = object).columns
#Storing all the categorical variables in the "columns" list

columns  = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',

       'PavedDrive', 'SaleType', 'SaleCondition']
#category_onehot_multcols is function that takes list of categorical columns and covert it into dummy variables

def category_onehot_multcols(multcolumns):

    df_final=final_df

    i=0

    for fields in multcolumns:

        

        print(fields)

        df1=pd.get_dummies(final_df[fields],drop_first=True)

        

        final_df.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:

            

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([final_df,df_final],axis=1)

        

    return df_final
#Coverting all the categorical columns into dummy variables 

final_df=category_onehot_multcols(columns)
final_df.shape
#Removing the duplicates

final_df = final_df.loc[:,~final_df.columns.duplicated()]

final_df.shape
#Splitting the final_df into df_Train and df_Test.

#As we know that in our "submission.csv" ,house id starts from 1458.

df_Train=final_df.iloc[:1458,:]

df_Test=final_df.iloc[1458:,:]
#Deleting the SalePrice From df_test

df_Test.drop(["SalePrice"],axis=1,inplace = True)
#Fom training of our Linear Regression model

X_train = df_Train.drop(["SalePrice"],axis = 1)

y_train = df_Train["SalePrice"]
from sklearn.linear_model import LinearRegression

#Creating instance of LinearRegression()

lr = LinearRegression()

#Using fit method to obtain linear Regression

lr.fit(X_train,y_train)

#Using linear regression to predict the the HousePrices of test dataset

y_pred = lr.predict(df_Test)

#Taking exponential of y_pred

predictions =np.exp(y_pred)
#Making dataframe "submit" according to the given submission file.

tid = df_Test["Id"]

submit = pd.DataFrame({"Id" : tid, "SalePrice" : predictions})

submit.head()
#Coverting the DataFrame submit into csv file.

submit.to_csv("house_price_prediction.csv",index = False)