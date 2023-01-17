# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
test_data_copy = test_data
train_data_copy = train_data

#Determining the number of null values for each variable
temp_na = pd.DataFrame()
temp_na["variable"] = train_data.isnull().apply(sum).index
temp_na["nulls"] = train_data.isnull().apply(sum).values
temp_na = temp_na.sort_values(by="nulls",ascending=False)
#temp_na

#Out of the main explanatory variables, GarageFinish and BsmtQual seem to have the highest 
#number of null values
'''
Looking for variables that have no or little data
'''

#Running a loop to find out which variables have relatively higher number of null values as these
#don't provide us with much information. Also, since a lot of these values were not provided by 
#customers it is a safe assumption that these are not high in priority as factors that would 
#affect price

for i in train_data.columns:
    print (i,sum(train_data[i].isnull())," out of ",train_data.shape[0])  

#From the above data variables Alley, FireplaceQu, PoolQC, Fence, MiscFeature seem to have 
#a high number of null values
#Dropping the above determined variables from the train and test data sets
train_data = train_data.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1)
test_data = test_data.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1)

#Dropping all rows with null values
train_data = train_data.dropna()
test_data = test_data.dropna()
'''
Initial Correlation Matrix
'''
#train_data.corr().style.background_gradient()
'''
Exploratory Data Analysis
'''
'''
Univariate analysis
'''
'''
Analyzing the target variable - Sale Price
'''
plt.figure()
plt.hist(train_data["SalePrice"],bins=20,rwidth=0.8,align="mid",range=(0,1000000))
plt.xlabel("Sale Price")
plt.ylabel("Frequency")

#From the plot below, it seems like the distribution for Sale Price is right skewed and unimodal
#Since we have a large sample size, this data can be analyzed using a normal distribution

#Summary statistics
price_mean = np.mean(train_data["SalePrice"])
price_median = np.median(train_data["SalePrice"])
print ("Sale Price summary statistics")
print ("Length ",len(train_data["SalePrice"]))
print ("Min ",min(train_data["SalePrice"]))
print ("Max ",max(train_data["SalePrice"]))
print ("Mean ",price_mean)
print ("Median ",price_median)

#Since the distribution is right skewed, mean (180921.1958) is higher than the median (163000)
'''
Analyzing other variables
'''
'''
plt.figure()
plt.bar(Counter(train_data["MSSubClass"]).keys(),Counter(train_data["MSSubClass"]).values(),width=4,align="center")
plt.xlabel("Dwelling type")
plt.ylabel("Frequency")

#From the bar plot of dwelling type, it seems like type 20 and 60 have a relatively high frequency
#indicating that these are the most popular types of dwelling

plt.figure()
plt.bar(Counter(train_data["MSZoning"]).keys(),Counter(train_data["MSZoning"]).values(),width=0.8,align="center")
plt.xlabel("Zoning class")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["LotFrontage"],bins=32,rwidth=0.8,align="left",range=(min(train_data["LotFrontage"]),max(train_data["LotFrontage"])+10))
plt.xlabel("Lot frontage")
plt.ylabel("Frequency")
print (min(train_data["LotFrontage"]))
print (max(train_data["LotFrontage"]))

plt.figure()
plt.hist(train_data["LotArea"],bins=100,rwidth=0.8,align="left",range=(min(train_data["LotArea"]),max(train_data["LotArea"])+1))
plt.xlabel("Lot area")
plt.ylabel("Frequency")
'''
'''
Univariate (contd)
'''
'''
plt.figure()
plt.bar(Counter(train_data["Street"]).keys(),Counter(train_data["Street"]).values(),width=0.8,align="center")
plt.xlabel("Street type")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["LotShape"]).keys(),Counter(train_data["LotShape"]).values(),width=0.8,align="center")
plt.xlabel("Lot shape")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["LandContour"]).keys(),Counter(train_data["LandContour"]).values(),width=0.8,align="center")
plt.xlabel("Land Contour")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["Utilities"]).keys(),Counter(train_data["Utilities"]).values(),width=0.8,align="center")
plt.xlabel("Utilities")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["LotConfig"]).keys(),Counter(train_data["LotConfig"]).values(),width=0.8,align="center")
plt.xlabel("Lot Config")
plt.ylabel("Frequency")
'''
'''
Univariate (contd)
'''
'''
plt.figure()
plt.bar(Counter(train_data["LandSlope"]).keys(),Counter(train_data["LandSlope"]).values(),width=0.8,align="center")
plt.xlabel("Lot Slope")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["Neighborhood"]).keys(),Counter(train_data["Neighborhood"]).values(),width=0.8,align="center")
plt.xlabel("Neighborhood")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["Condition1"]).keys(),Counter(train_data["Condition1"]).values(),width=0.8,align="center")
plt.xlabel("Condition1")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["Condition2"]).keys(),Counter(train_data["Condition2"]).values(),width=0.8,align="center")
plt.xlabel("Condition2")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["BldgType"]).keys(),Counter(train_data["BldgType"]).values(),width=0.8,align="center")
plt.xlabel("Building Type")
plt.ylabel("Frequency")
'''

'''
Univariate (contd)
'''
'''
plt.figure()
plt.bar(Counter(train_data["HouseStyle"]).keys(),Counter(train_data["HouseStyle"]).values(),width=0.8,align="center")
plt.xlabel("House Style")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["OverallQual"]).keys(),Counter(train_data["OverallQual"]).values(),width=0.8,align="center")
plt.xlabel("OverallQual")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["OverallCond"]).keys(),Counter(train_data["OverallCond"]).values(),width=0.8,align="center")
plt.xlabel("Overall Condition")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["YearBuilt"],bins=20,rwidth=0.8,align="left",range=(1850,2050))
plt.xlabel("Year Built")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["YearRemodAdd"],bins=20,rwidth=0.8,align="left",range=(1850,2050))
plt.xlabel("Year Remodelled")
plt.ylabel("Frequency")
'''
'''
Univariate (contd)
'''
'''
plt.figure()
plt.bar(Counter(train_data["HouseStyle"]).keys(),Counter(train_data["HouseStyle"]).values(),width=0.8,align="center")
plt.xlabel("House Style")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["RoofStyle"]).keys(),Counter(train_data["RoofStyle"]).values(),width=0.8,align="center")
plt.xlabel("Roof Style")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["RoofMatl"]).keys(),Counter(train_data["RoofMatl"]).values(),width=0.8,align="center")
plt.xlabel("Roof material")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["Exterior1st"]).keys(),Counter(train_data["Exterior1st"]).values(),width=0.8,align="center")
plt.xlabel("Exterior")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["Exterior2nd"]).keys(),Counter(train_data["Exterior2nd"]).values(),width=0.8,align="center")
plt.xlabel("Exterior")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["MasVnrType"]).keys(),Counter(train_data["MasVnrType"]).values(),width=0.8,align="center")
plt.xlabel("MasVnrType")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["MasVnrArea"])
plt.xlabel("MasVnr area")
plt.ylabel("Frequency")
'''
'''
Univariate (contd)
'''
'''
plt.figure()
plt.bar(Counter(train_data["ExterQual"]).keys(),Counter(train_data["ExterQual"]).values(),width=0.8,align="center")
plt.xlabel("External Qual")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["ExterCond"]).keys(),Counter(train_data["ExterCond"]).values(),width=0.8,align="center")
plt.xlabel("External Condition")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["Foundation"]).keys(),Counter(train_data["Foundation"]).values(),width=0.8,align="center")
plt.xlabel("Foundation")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["BsmtQual"]).keys(),Counter(train_data["BsmtQual"]).values(),width=0.8,align="center")
plt.xlabel("Basement Qual")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["BsmtCond"]).keys(),Counter(train_data["BsmtCond"]).values(),width=0.8,align="center")
plt.xlabel("Basement condition")
plt.ylabel("Frequency")
'''
'''
Univariate (contd)
'''

plt.figure()
plt.bar(Counter(train_data["ExterQual"]).keys(),Counter(train_data["ExterQual"]).values(),width=0.8,align="center")
plt.xlabel("External Qual")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["BsmtExposure"]).keys(),Counter(train_data["BsmtExposure"]).values(),width=0.8,align="center")
plt.xlabel("Basement Exposure")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["BsmtFinType1"]).keys(),Counter(train_data["BsmtFinType1"]).values(),width=0.8,align="center")
plt.xlabel("Basement finish")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["BsmtFinSF1"])
plt.xlabel("Basement area")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["BsmtFinType2"]).keys(),Counter(train_data["BsmtFinType2"]).values(),width=0.8,align="center")
plt.xlabel("Basement finish")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["BsmtFinSF2"])
plt.xlabel("Basement area")
plt.ylabel("Frequency")
plt.figure()
plt.hist(train_data["BsmtUnfSF"])
plt.xlabel("Unfinished basement area")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["TotalBsmtSF"])
plt.xlabel("Basement area")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["Heating"]).keys(),Counter(train_data["Heating"]).values(),width=0.8,align="center")
plt.xlabel("Heating")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["HeatingQC"]).keys(),Counter(train_data["HeatingQC"]).values(),width=0.8,align="center")
plt.xlabel("Heating QC")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["CentralAir"]).keys(),Counter(train_data["CentralAir"]).values(),width=0.8,align="center")
plt.xlabel("Central AC")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["Electrical"]).keys(),Counter(train_data["Electrical"]).values(),width=0.8,align="center")
plt.xlabel("Electrical")
plt.ylabel("Frequency")
plt.figure()
plt.hist(train_data["1stFlrSF"])
plt.xlabel("First floor square feet")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["2ndFlrSF"])
plt.xlabel("Second floor square feet")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["LowQualFinSF"])
plt.xlabel("Low quality finish SF")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["GrLivArea"])
plt.xlabel("Above grade living area")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["BsmtFullBath"])
plt.xlabel("Basement full bathrooms")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["BsmtHalfBath"])
plt.xlabel("Basement half bathrooms")
plt.ylabel("Frequency")
plt.figure()
plt.hist(train_data["FullBath"],range=(0,5),bins=10)
plt.xlabel("Full bath above grade")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["HalfBath"],range=(0,5),bins=10)
plt.xlabel("First floor square feet")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["BedroomAbvGr"],range=(0,10),bins=10)
plt.xlabel("Bedrooms above grade")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["KitchenAbvGr"])
plt.xlabel("Kitchens above grade")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["KitchenQual"]).keys(),Counter(train_data["KitchenQual"]).values(),width=0.8,align="center")
plt.xlabel("Kitchen quality")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["TotRmsAbvGrd"],range=(0,10),bins=10)
plt.xlabel("Total rooms above grade")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["Functional"]).keys(),Counter(train_data["Functional"]).values(),width=0.8,align="center")
plt.xlabel("Functionality")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["Fireplaces"],range=(0,10),bins=10)
plt.xlabel("Fireplaces")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["GarageType"]).keys(),Counter(train_data["GarageType"]).values(),width=0.8,align="center")
plt.xlabel("Garage Type")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["GarageYrBlt"])
plt.xlabel("Garage Year Built")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["GarageFinish"]).keys(),Counter(train_data["GarageFinish"]).values(),width=0.8,align="center")
plt.xlabel("Garage Finish")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["GarageCars"],range=(0,10),bins=10)
plt.xlabel("Garage capacity")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["GarageArea"])
plt.xlabel("Garage area")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["GarageQual"]).keys(),Counter(train_data["GarageQual"]).values(),width=0.8,align="center")
plt.xlabel("Garage Qual")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["GarageCond"]).keys(),Counter(train_data["GarageCond"]).values(),width=0.8,align="center")
plt.xlabel("Garage Condition")
plt.ylabel("Frequency")
plt.figure()
plt.bar(Counter(train_data["PavedDrive"]).keys(),Counter(train_data["PavedDrive"]).values(),width=0.8,align="center")
plt.xlabel("Paved Drive")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["WoodDeckSF"])
plt.xlabel("Deck SF")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["OpenPorchSF"])
plt.xlabel("Open Porch SF")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["EnclosedPorch"])
plt.xlabel("Enclosed Porched area")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["3SsnPorch"])
plt.xlabel("Three season porch")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["ScreenPorch"])
plt.xlabel("Screen porch")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["PoolArea"])
plt.xlabel("Pool area")
plt.ylabel("Frequency")

plt.figure()
plt.hist(train_data["MoSold"])
plt.xlabel("Month Sold")
plt.ylabel("Frequency")
plt.figure()
plt.hist(train_data["YrSold"],range=(2005,2015),bins=20)
plt.xlabel("Year sold")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["SaleType"]).keys(),Counter(train_data["SaleType"]).values(),width=0.8,align="center")
plt.xlabel("Sale Type")
plt.ylabel("Frequency")

plt.figure()
plt.bar(Counter(train_data["SaleCondition"]).keys(),Counter(train_data["SaleCondition"]).values(),width=0.8,align="center")
plt.xlabel("Sale Condition")
plt.ylabel("Frequency")
'''
Removing variables from univariate section
'''
#Since the MSZoning variable shows RL as the predominant zoning class, this variable offers
#little information
train_data = train_data.drop("MSZoning",axis=1)
test_data = test_data.drop("MSZoning",axis=1)
#Removing variable Street and Land Contour since most of the houses have paved streets
train_data = train_data.drop(["LotFrontage","Street","LandContour","Utilities","LandSlope","Condition1","Condition2"],axis=1)
test_data = test_data.drop(["LotFrontage","Street","LandContour","Utilities","LandSlope","Condition1","Condition2"],axis=1)
train_data = train_data.drop(["BldgType","RoofStyle","RoofMatl","BsmtCond","BsmtFinType2","BsmtFinSF2","BsmtUnfSF"],axis=1)
test_data = test_data.drop(["BldgType","RoofStyle","RoofMatl","BsmtCond","BsmtFinType2","BsmtFinSF2","BsmtUnfSF"],axis=1)
train_data = train_data.drop(["Heating","CentralAir","Electrical","LowQualFinSF","BsmtHalfBath","SaleCondition"],axis=1)
test_data = test_data.drop(["Heating","CentralAir","Electrical","LowQualFinSF","BsmtHalfBath","SaleCondition"],axis=1)
train_data = train_data.drop(["KitchenAbvGr","Functional","GarageQual","GarageCond","PavedDrive","OpenPorchSF"],axis=1)
test_data = test_data.drop(["KitchenAbvGr","Functional","GarageQual","GarageCond","PavedDrive","OpenPorchSF"],axis=1)
train_data = train_data.drop(["EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","SaleType"],axis=1)
test_data = test_data.drop(["EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","SaleType"],axis=1)

#Too many unique discrete values
train_data = train_data.drop(["Neighborhood","HeatingQC"],axis=1)
test_data = test_data.drop(["Neighborhood","HeatingQC"],axis=1)

#Removing YearRemodAdd since it is slightly collinear with YearBuilt
train_data = train_data.drop("YearRemodAdd",axis=1)
test_data = test_data.drop("YearRemodAdd",axis=1)

print (train_data.columns)
'''
Label Encoding Categorical Variables
'''
#This needs to be done for easier computing of correlation between SalePrice and 
#categorical variables
from sklearn.preprocessing import LabelEncoder

#Selecting columns that contain only string values
categorical = train_data.select_dtypes(include="object").columns
        
#Running a loop across the categorical variables to label encode
for i in categorical:
    le = LabelEncoder().fit(train_data[i].append(test_data[i]))
    train_data[i] = le.transform(train_data[i])
    test_data[i] = le.transform(test_data[i])
'''
Feature Selection Experiments (optional)
'''
'''
x_temp = train_data.drop("SalePrice",axis=1)
y_temp = train_data["SalePrice"]

print (feature_selection.f_regression(x_temp,y_temp)[1])
print (x_temp.columns)

feat_sel = pd.DataFrame()
feat_sel["features"] = x_temp.columns
feat_sel["p value"] = feature_selection.f_regression(x_temp,y_temp)[1]
print (feat_sel.sort_values(by="p value",ascending=True))
'''
'''
Multivariate Analysis
'''
'''
Correlation Matrix - only for numerical data types, strings have to be label encoded
'''
corr_matrix = train_data.corr(method="pearson")
print (corr_matrix.shape)
corr_matrix.style.background_gradient()

train_id = train_data["Id"]
test_id = test_data_copy["Id"]

#'''
#Dropping numerical variables based on correlation matrix results
#'''
#Dropping variables which have a low correlation with the target variable SalePrice
#We do this by simply selecting the variables that have a relatively high correlation (>0.5)
train_data = train_data.loc[:,["SalePrice","OverallQual","YearBuilt","ExterQual","BsmtQual","TotalBsmtSF","1stFlrSF","FullBath","KitchenQual","TotRmsAbvGrd","GarageYrBlt","GarageFinish","GarageArea","GrLivArea","GarageCars"]]
test_data = test_data.loc[:,["SalePrice","OverallQual","YearBuilt","ExterQual","BsmtQual","TotalBsmtSF","1stFlrSF","FullBath","KitchenQual","TotRmsAbvGrd","GarageYrBlt","GarageFinish","GarageArea","GrLivArea","GarageCars"]]
#Visualizing correlation matrix again
corr_matrix = train_data.corr(method="pearson")
print (corr_matrix.shape)
corr_matrix.style.background_gradient()

'''
From the resulting matrix the variables that show a relatively high collinearity can be observed
'''

#YrBuilt and GarageYrBlt - Since SalePrice has slightl higher correlation with YrBuilt, 
#GarageYrBlt can be removed from the regression

#TotRmsAbvGrd and GrLivArea - SalePrice has a higher correlation with GrLiveArea, 
#hence removing TotsRmsAbvGrade

#GarageArea and GarageCars - Removing GarageArea since GarageCars has a higher correlation
#with SalePrice

#TotalBsmtSF and 1stFlrSF - Removing TotalBsmtSF due to collinearity

train_data = train_data.drop(["GarageYrBlt","TotalBsmtSF","TotRmsAbvGrd","GarageArea"],axis=1)
test_data = test_data.drop(["GarageYrBlt","TotalBsmtSF","TotRmsAbvGrd","GarageArea"],axis=1)
corr_matrix = train_data.corr(method="pearson")
print (corr_matrix.shape)
corr_matrix.style.background_gradient()

'''
Bivariate Analysis
'''
#Sale Price vs Quality
plt.figure()
plt.scatter(train_data["OverallQual"],train_data["SalePrice"])
plt.xlabel("Quality")
plt.ylabel("Price")
#There is a moderate positive correlation between price and overall quality

#SalePrice vs YearBuilt
plt.figure()
plt.scatter(train_data["YearBuilt"],train_data["SalePrice"])
plt.xlabel("Year built")
plt.ylabel("Price")
#Seems to be weakly correlated with a positive trend. Not too linear

#SalePrice vs ExterQual
plt.figure()
plt.scatter(train_data["ExterQual"],train_data["SalePrice"])
plt.xlabel("External Quality")
plt.ylabel("Price")
#Weak negative correlation

#SalePrice vs BsmtQual
plt.figure()
plt.scatter(train_data["BsmtQual"],train_data["SalePrice"])
plt.xlabel("Basement Quality")
plt.ylabel("Price")
#Weak, negative correlation

#SalePrice vs 1stFlrSF
plt.figure()
plt.scatter(train_data["1stFlrSF"],train_data["SalePrice"])
plt.xlabel("1st Floor Area (sq ft)")
plt.ylabel("Price")
#Moderate, positive and slightly linear correlation

#SalePrice vs FullBath
plt.figure()
plt.scatter(train_data["FullBath"],train_data["SalePrice"])
plt.xlabel("Full Bath")
plt.ylabel("Price")
#Moderate, positive correlation. Although, this variable is not too useful since houses with 
#2 or more bathrooms are generally going to be more expensive than houses with 1 bathroom
#Therefore, this comparison does not give us the best information. Perhaps we can drop the variable

#SalePrice vs KitchenQual
plt.figure()
plt.scatter(train_data["KitchenQual"],train_data["SalePrice"])
plt.xlabel("Kitchen Quality")
plt.ylabel("Price")

#SalePrice vs GarageFinish
plt.figure()
plt.scatter(train_data["GarageFinish"],train_data["SalePrice"])
plt.xlabel("Garage Finish")
plt.ylabel("Price")

#SalePrice vs GrLivArea
plt.figure()
plt.scatter(train_data["GrLivArea"],train_data["SalePrice"])
plt.xlabel("GrLivArea")
plt.ylabel("Price")
#Somewhat strong, positive and linear correlation

#SalePrice vs GarageCars
plt.figure()
plt.scatter(train_data["GarageCars"],train_data["SalePrice"])
plt.xlabel("GarageCars")
plt.ylabel("Price")
#Moderate, positive correlation

'''
Bivariate Observations
'''
#Variables ExterQual, KitchenQual and BsmtQual appear to have very similar associations with 
#SalePrice. All of them have very similar correlation values with SalePrice. Further bivariate 
#analysis is conducted between these 3 variables in the next section

'''
Checking for more collinearity between variables ExterQual, BsmtQual and KitchenQual
'''

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
ext_bsmt_qual = train_data.loc[:,["ExterQual","BsmtQual","KitchenQual"]]
val1 = ext_bsmt_qual[ext_bsmt_qual["BsmtQual"]==0]
val2 = ext_bsmt_qual[ext_bsmt_qual["BsmtQual"]==1]
val3 = ext_bsmt_qual[ext_bsmt_qual["BsmtQual"]==2]
val4 = ext_bsmt_qual[ext_bsmt_qual["BsmtQual"]==3]
plt.hist([val1["ExterQual"],val2["ExterQual"],val3["ExterQual"],val4["ExterQual"]],range=(0,5),bins=10,alpha=0.8,histtype="barstacked",label=["Excellent","Good","Typical","Fair"])
plt.xlabel("ExterQual")
plt.ylabel("BsmtQual")
plt.legend()

plt.subplot(1,2,2)
ext_bsmt_qual = train_data.loc[:,["ExterQual","BsmtQual","KitchenQual"]]
val1 = ext_bsmt_qual[ext_bsmt_qual["BsmtQual"]==0]
val2 = ext_bsmt_qual[ext_bsmt_qual["BsmtQual"]==1]
val3 = ext_bsmt_qual[ext_bsmt_qual["BsmtQual"]==2]
val4 = ext_bsmt_qual[ext_bsmt_qual["BsmtQual"]==3]
val5 = ext_bsmt_qual[ext_bsmt_qual["BsmtQual"]==4]
val6 = ext_bsmt_qual[ext_bsmt_qual["BsmtQual"]==5]
plt.hist([val1["KitchenQual"],val2["KitchenQual"],val3["KitchenQual"],val4["KitchenQual"],val5["KitchenQual"],val6["KitchenQual"]],range=(0,5),bins=10,alpha=0.8,histtype="barstacked",label=["Excellent","Good","Typical","Fair","Poor","NA"])
plt.xlabel("Kitchen Qual")
plt.ylabel("BsmtQual")
plt.legend()

#Note that stacked histograms/bar plots had to be created for the above 3 variables since
#they give us a better visualization of the distribution of each variable across the other
#Scatter plots are not too helpful in this case

#From the plots below, it can be seen that the distribution of both ExterQual and KitchenQual 
#across BsmtQual are very similar, which is expected

########### Correlations between the three variables ############
ext_bsmt_qual.corr().style.background_gradient()
#################################################################

#Since the variables provide very similar information towards predicting SalePrice,
#we can include just one of them to prevent collinearity. Hence we select ExterQual as it
#has the highest absolute correlation with SalePrice

train_data = train_data.drop(["KitchenQual","BsmtQual"],axis=1)
test_data = test_data.drop(["KitchenQual","BsmtQual"],axis=1)

#Removing ExterQual and GarageFinish due to relatively low correlation with SalePrice
#train_data = train_data.drop(["ExterQual","GarageFinish"],axis=1)
#test_data = test_data.drop(["ExterQual","GarageFinish"],axis=1)
'''
Handling missing values - BsmtQual and GarageFinish
'''
#Doing some pre-processing on train_data_copy
train_data_copy = train_data_copy.loc[:,train_data.columns]

#Verifying the variables with the null values - GarageFinish
train_data_copy.isnull().apply(sum)

#Converting NA values into string
train_data_copy["GarageFinish"] = train_data_copy["GarageFinish"].map(str)
#train_data_copy["KitchenQual"] = train_data_copy["KitchenQual"].map(str)
#train_data_copy["BsmtQual"] = train_data_copy["BsmtQual"].map(str)

#Label Encoding
le = LabelEncoder().fit(train_data["GarageFinish"].map(str).append(test_data["GarageFinish"].map(str)).append(train_data_copy["GarageFinish"]))
train_data_copy["GarageFinish"] = le.transform(train_data_copy["GarageFinish"])
le = LabelEncoder().fit(train_data["ExterQual"].map(str).append(test_data["ExterQual"].map(str)).append(train_data_copy["ExterQual"]))
train_data_copy["ExterQual"] = le.transform(train_data_copy["ExterQual"])
#le = LabelEncoder().fit(train_data["KitchenQual"].map(str).append(test_data["KitchenQual"].map(str)).append(train_data_copy["KitchenQual"]))
#train_data_copy["KitchenQual"] = le.transform(train_data_copy["KitchenQual"])
#le = LabelEncoder().fit(train_data["BsmtQual"].map(str).append(test_data["BsmtQual"].map(str)).append(train_data_copy["BsmtQual"]))
#train_data_copy["BsmtQual"] = le.transform(train_data_copy["BsmtQual"])

#Filling missing values with mean
train_data_copy.shape
'''
More pre-processing
'''
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

#Separatin the target variable and the input features
x = train_data.drop("SalePrice",axis=1)
test_x = test_data.drop("SalePrice",axis=1)
y = train_data["SalePrice"]
test_y = test_data["SalePrice"]
x_copy = train_data_copy.drop("SalePrice",axis=1)
y_copy = train_data_copy["SalePrice"]
'''
Modeling and Scaling
'''

from sklearn.metrics.regression import r2_score, explained_variance_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

#Train/Test split
X_train,X_test,y_train,y_test = train_test_split(x_copy,y_copy,random_state=0,train_size=0.9)

#Normalization using MinMax scaler
#scaler = MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

'''
Gradient Boosting Regressor
'''
'''
#Creating and fitting the model
#gbc = GradientBoostingRegressor(random_state=0,learning_rate=0.1,n_estimators=25,max_depth=6,subsample=0.8,min_samples_split=10)
#gbc = GradientBoostingRegressor(random_state=0,learning_rate=0.1,n_estimators=100,max_depth=3,subsample=1,min_samples_split=2)
gbc = GradientBoostingRegressor(random_state=0,learning_rate=0.1,n_estimators=100,max_depth=3,subsample=1,min_samples_split=2)

model = gbc.fit(X_train,y_train)
#linreg = LinearRegression()
#model = linreg.fit(X_train,y_train)

#Prediction
y_predict = model.predict(X_test)
r2 = r2_score(y_test,list(map(int,y_predict)))

print (r2)
print (np.corrcoef(y_test,y_predict))
plt.scatter(range(len(y_test)),y_test)
plt.scatter(range(len(list(map(int,y_predict)))),list(map(int,y_predict)))
#plt.scatter(X_test,y_predict)
plt.show()
'''
'''
More Feature Selection Following GBC
'''
corr_matrix = train_data.corr(method="pearson")
corr_matrix.style.background_gradient()
'''
Re-running GBC
'''
import math
from sklearn.metrics import mean_squared_log_error

X_train,X_test,y_train,y_test = train_test_split(x_copy,y_copy,random_state=0,train_size=0.9)

#Fitting
#gbc = GradientBoostingRegressor(random_state=0,learning_rate=0.1,n_estimators=100,max_depth=2,subsample=0.5,min_samples_split=2)
#gbc = GradientBoostingRegressor(random_state=0)
#With KitchenQual, BsmtQual, ExterQual and GarageFinish
#gbc = GradientBoostingRegressor(random_state=0,learning_rate=0.1,n_estimators=65,max_depth=3,subsample=1,min_samples_split=2)
#With ExterQual and GarageFinish
gbc = GradientBoostingRegressor(random_state=0,learning_rate=0.1,n_estimators=50,max_depth=3,subsample=0.5,min_samples_split=2)
#Without ExterQual and GarageFinish
#gbc = GradientBoostingRegressor(random_state=0,learning_rate=0.1,n_estimators=50,max_depth=3,subsample=0.9,min_samples_split=2)

model = gbc.fit(X_train,y_train)

#Prediction
y_predict = model.predict(X_test)
r2 = r2_score(y_test,list(map(int,y_predict)))

print (x_copy.columns)
print (test_x.columns)
print (X_test.columns)
print ("R-squared: ",r2)
print ("RMSLE ",math.sqrt(mean_squared_log_error(y_test,list(map(int,y_predict)))))
print (np.corrcoef(y_test,y_predict))

'''
R-Squared - estimators: 100, subsample: 1
'''
#GBC with no parameters
#With all variables
#0.8756

#Dropped ExterQual
#0.8759

#Dropped ExterQual and GarageFinish
#0.8772

#Dropped ExterQual, GarageFinish and YearBuilt
#0.8666

#Dropped ExterQual, GarageFinish, YearBuilt and FullBath
#0.8543

'''
Root Mean Squared Log Error
'''

#Estimators: 200, subsample: 0.8
#With all variables
#0.1486

#Dropped ExterQual
#0.1483

#Dropped ExterQual and GarageFinish
#0.1446

print (x_copy.columns)
'''
Manual Grid Search - Root Mean Squared Log Error
'''
'''
for i in [0.01,0.1]:
    for j in [25,50,60,75,85,100,125,150,175,200]:
        for k in [2,3]:
            for l in [0.5,0.7,0.8,0.9,1]:
                gbc = GradientBoostingRegressor(random_state=0,learning_rate=i,n_estimators=j,max_depth=k,subsample=l,min_samples_split=2)
                model = gbc.fit(X_train,y_train)
                y_predict = model.predict(X_test)
                r2 = r2_score(y_test,list(map(int,y_predict)))
                msle = math.sqrt(mean_squared_log_error(y_test,list(map(int,y_predict))))
                if msle < 0.13:
                    print ("R-squared: ",r2)
                    print ("RMSLE ",msle)
                    print ("learning rate: ",i)
                    print ("estimators: ",j)
                    print ("max depth: ",k)
                    print ("subsample: ",l)
                    print ("")
'''
'''
Prediction using test data
'''
#Pre-processing the test dataset to include the same columns as that in train data
test_data_copy = test_data_copy.loc[:,x_copy.columns]
#test_data_copy = test_data_copy.drop("SalePrice",axis=1)

#Label Encoding the test data
#Selecting columns that contain only string values in the test_data_copy dataframe
categorical_test_data_copy = test_data_copy.select_dtypes(include="object").columns

for i in categorical_test_data_copy:
    test_data_copy[i] = test_data_copy[i].map(str)
    train_data_copy[i] = train_data_copy[i].map(str)
    le = LabelEncoder().fit(train_data_copy[i].append(test_data_copy[i]))
    test_data_copy[i] = le.transform(test_data_copy[i])

#More pre-processing (Only one NA value so filling it with the mean)
test_data_copy["GarageCars"] = test_data_copy["GarageCars"].fillna(np.mean(test_data_copy["GarageCars"]))
#test_data_copy["BsmtQual"] = test_data_copy["BsmtQual"].fillna(np.mean(test_data_copy["BsmtQual"]))
#test_data_copy["GarageFinish"] = test_data_copy["GarageFinish"].fillna(np.mean(test_data_copy["GarageFinish"]))
#test_data_copy["KitchenQual"] = test_data_copy["KitchenQual"].fillna(np.mean(test_data_copy["KitchenQual"]))

predict_test_df = model.predict(test_data_copy)
predict_test_df.shape
'''
Presenting final results
'''
results_df = pd.DataFrame()
results_df["Id"] = test_id
results_df["SalePrice"] = predict_test_df
#results_df = results_df.loc[:,["Id","SalePrice"]]
#results_df = results_df.set_index("Id")

#print (results_df)
#results_df.to_csv("Final_Predictions.csv")
results_df
results_df.to_csv("test.csv",index=False)
'''
GBC Grid Search
'''
'''
#grid_values = {'learning_rate': [0.01, 0.05, 0.1]}
grid_values = {'learning_rate':[0.01,0.1],'max_depth':[2,3,4],'n_estimators':[50,65,100,150,175,200],'subsample':[0.5,0.6,0.7,0.8,0.9,1],'min_samples_split':[2]}
#grid_values = {'learning_rate':[0.001,0.01,0.1,1],'max_depth':[2,3],'n_estimators':[50,100,150],'subsample':[0.8,0.9,1],'min_samples_split':[2]}
    
#grid_clf = GridSearchCV(gbc,param_grid = grid_values, scoring='r2',cv=20)
grid_clf = GridSearchCV(gbc,param_grid = grid_values, scoring='neg_mean_squared_log_error',cv=10)
grid_clf.fit(X_train, y_train)

print (grid_clf.cv_results_)
print ("")
print ("Max accuracy parameters ",grid_clf.best_params_)
print ("Best accuracy score ",np.sqrt(abs(grid_clf.best_score_)))
    
#y_decision_fn_scores = grid_clf.decision_function(X_test) 
'''