#I am using this dataset as a playgroud for my learning of ML-algorithms in the Udemy course
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Let's start by exploring the data
test=pd.read_csv("../input/test.csv")
test2=test.copy()
test.shape
test.info()
test.columns
test.describe()
#test does not have a value column
train=pd.read_csv("../input/train.csv")
train2=train.copy()
train.shape
train.columns
#Here we go: SalePrice is the y to predict
train.head()
# Let's plot some of the data
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#There are NaN values in the data (therefore pairplot didn't work)
train.info()
# Now lets start with linear model
from sklearn.linear_model import LinearRegression
model1=LinearRegression()
# For now, I use only already data in proper number format and interpretation
train_quant=train[["LotFrontage", "LotArea", "OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF", "TotalBsmtSF","1stFlrSF","2ndFlrSF"     ,"LowQualFinSF" ,"GrLivArea"    ,"BsmtFullBath" , "BsmtHalfBath" , "FullBath","HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars","GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch","PoolArea", "MiscVal","MoSold", "YrSold"]]
# For now, easily fill NaN's with average
train_quant.fillna(train_quant.mean(),inplace=True)
model1.fit(train_quant,train["SalePrice"])
model1.score(train_quant,train["SalePrice"]) #=R^2
parameters=pd.DataFrame(model1.coef_,train_quant.columns,["Parameters"])
parameters
#Now predict
test.fillna(test.mean(),inplace=True)
predictions=model1.predict(test[["LotFrontage", "LotArea", "OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF", "TotalBsmtSF","1stFlrSF","2ndFlrSF"     ,"LowQualFinSF" ,"GrLivArea"    ,"BsmtFullBath" , "BsmtHalfBath" , "FullBath","HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars","GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch","PoolArea", "MiscVal","MoSold", "YrSold"]])
solution=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
# Two predictions were below 0. For now correct them to 0 to have a first valid result for scoring
# Quite brutal. As two solutions were <0 and the Kaggle solution didn't accept negative house values set them manually to 0 for now
solution.loc[solution["SalePrice"]<0, "SalePrice"] = 0
solution.to_csv("solution.csv", index=False)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(train_quant.corr(),cmap="coolwarm")
from sklearn.feature_selection import f_regression
standalone_p=f_regression(train_quant,train["SalePrice"], center=True)[1]
# Returns two lists: F-score and p-value
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html
# Linear model for testing the individual effect of each of many regressors. This is a scoring function to be used in a feature seletion procedure, not a free standing feature selection procedure.
parameters["p_values"]=standalone_p
parameters
#kick-out from model BsmtFinSF2,LowQualFinSF,BsmtHalfBath,MiscVal,YrSold,MoSold,3SsnPorch
# Now predict again
model2=LinearRegression()
train_quant=train[["LotFrontage", "LotArea", "OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtUnfSF", "TotalBsmtSF","1stFlrSF","2ndFlrSF","GrLivArea"    ,"BsmtFullBath" , "FullBath","HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars","GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch","PoolArea"]]
train_quant.fillna(train_quant.mean(),inplace=True)
model2.fit(train_quant,train["SalePrice"])
parameters=pd.DataFrame(model2.coef_,train_quant.columns,["Parameters"])
parameters

test.fillna(test.mean(),inplace=True)
predictions=model2.predict(test[["LotFrontage", "LotArea", "OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtUnfSF", "TotalBsmtSF","1stFlrSF","2ndFlrSF","GrLivArea"    ,"BsmtFullBath" , "FullBath","HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars","GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch","PoolArea"]])
solution2=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution2.loc[solution2["SalePrice"]<0, "SalePrice"] = 0
solution2.to_csv("solution.csv", index=False)
# Ok it got 0.0003 worse
# So what actually is the disadvantage of having lots of lots of features?
model2.score(train_quant,train["SalePrice"])
# Lets write a function for parametrization (What is the correct word?) of qualitative features
# Function could be rewritten to include list of columns to return column-names w/o doubling
def parametrize(data):
    #Setup the dataframe
    #unique values within the series are column-labels
    features=data.unique()
    df = pd.DataFrame(np.zeros(shape=(len(data),len(features))), columns=features)
    for element in features:
        df[element].loc[data==element] = 1
    #Rename column names to avoid numerous same columns in result
    new_columns=[]
    for element in df.columns:
        new_columns.append(str(data.name)+" - "+str(element))
    df.columns=new_columns
    return df

#Write parametrized qualitative features to training data 
qualitative_features=["MSSubClass","MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition"]
for element in qualitative_features:
    new=parametrize(train[element])
    train=pd.concat([train,new],axis=1)

#Delete qualitative columns
train.drop(qualitative_features,axis=1,inplace=True)
# Also the test data needs parametrization
for element in qualitative_features:
    new=parametrize(test[element])
    test=pd.concat([test,new],axis=1)

#Delete qualitative columns
test.drop(qualitative_features,axis=1,inplace=True)
# Cleaning the full datasets
train.fillna(train.mean(),inplace=True)
test.fillna(train.mean(),inplace=True)
#test had 12 dimensions less than train (+8 that were not in test)
test.drop(["MSSubClass - 150","MSZoning - nan","Utilities - nan","Exterior1st - nan","Exterior2nd - nan","KitchenQual - nan","Functional - nan","SaleType - nan"],axis=1,inplace=True)
train.drop(['Utilities - NoSeWa', 'Condition2 - RRNn', 'Condition2 - RRAn', 'Condition2 - RRAe', 'HouseStyle - 2.5Fin', 'RoofMatl - Metal', 'RoofMatl - Membran', 'RoofMatl - Roll', 'RoofMatl - ClyTile', 'Exterior1st - Stone', 'Exterior1st - ImStucc', 'Exterior2nd - Other', 'Heating - OthW', 'Heating - Floor', 'Electrical - Mix', 'Electrical - nan', 'GarageQual - Ex', 'PoolQC - Fa', 'MiscFeature - TenC'],axis=1,inplace=True)
# Now predict again with the parametrized qualitative features included
# Fit the model
model3=LinearRegression()
X_train=train.drop("SalePrice",axis=1)
model3.fit(X_train,train["SalePrice"])
parameters=pd.DataFrame(model3.coef_,X_train.columns,["Parameters"])
parameters

# Predict using the test-data
predictions=model3.predict(test)
solution3=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution3.loc[solution3["SalePrice"]<0, "SalePrice"] = 0
solution3.to_csv("solution.csv", index=False)
# Result: 
# Really, really worse than the simpler model before
# Score: 1.14436
# Why?
# ~300 Parameters instead of ~70?
# R^2 of model is much higher for train
# Overfitting. R^2 gets strictly better if adding features
model3.score(X_train,train["SalePrice"])
#Check the parameters
standalone_p=f_regression(X_train,train["SalePrice"], center=True)[1]
# Returns two lists: F-score and p-value
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html
# Linear model for testing the individual effect of each of many regressors. This is a scoring function to be used in a feature seletion procedure, not a free standing feature selection procedure.
parameters=pd.DataFrame(model3.coef_,X_train.columns,["Parameters"])
parameters["p_values"]=standalone_p
parameters

# https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

# Normalize features (Why)
# Throw out features with small p-Value
# Test for each feature what % of deviation it explains
    # Check highest correlation with SalePrice!!! Very easy
# Check for outliers in most important data
# Taking logs of SalePrice means that errors in predicting expensive houses and cheap houses will affect the result equally.
# Properly handle NaNs: # Handle missing values for features where median/mean or most common value doesn't make sense
# Some numerical features are actually really categories
# Encode some categorical features as ordered numbers when there is information in the order
# Simplifications of existing features
# Combinations of existing features
# Polynomials on the top 10 existing features
# Log transform of the skewed numerical features to lessen impact of outliers
    # Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
    # As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
        # skewness = train_num.apply(lambda x: skew(x))
        # train_num[skewed_features] = np.log1p(train_num[skewed_features])
# Standardization? StandardScaler() ???
    # This is useful when you want to compare data that correspond to different units. In that case, you want to remove the units. To do that in a consistent way of all the data, you transform the data in a way that the variance is unitary and that the mean of the series is 0.
# Split into training and validation data

# Use Regularization models like ridge or lasso or ElasticNet
# What's regularization again?
# Just for fun: Check how what portion of Error each single feature explains if used alone for prediction
model4=LinearRegression()
results=[]
for element in X_train.columns:
    X=X_train[element].reshape(-1,1)
    model4.fit(X,train["SalePrice"])
    results.append([element,model4.score(X,train["SalePrice"])])
result=pd.DataFrame(results,columns=["Variable","R^2"])
result.sort_values("R^2",ascending=False).head(10)
# Ok, lets use only top-predictors
# Only GarageCars, not GarageArea additionally
# ExterQual into categorical column 1-5
# BsmtQual into categorial column 1-6

# New function for categorial column
# To encode the categorical features into 0 or 1 was probably not too smart. Jumps extremely on these
# Probably encoding them 1-5 or similar is better
def categorize(data, categories):
    #Setup the pd.Series
    result = pd.Series(np.zeros(shape=len(data)))
    i=0
    for element in categories:
        i+=1
        result.loc[data==element] = i
    return result

X_train2=train2[["OverallQual","GrLivArea","GarageArea","TotalBsmtSF","1stFlrSF"]]
X_train2["ExterQual"]=categorize(train2.ExterQual,["Ex","Gd","TA","Fa","Po"])
X_train2["BsmtQual"]=categorize(train2.BsmtQual,["Ex","Gd","TA","Fa","Po","NA"])

X_test2=test2[["OverallQual","GrLivArea","GarageArea","TotalBsmtSF","1stFlrSF"]]
X_test2["ExterQual"]=categorize(test2.ExterQual,["Ex","Gd","TA","Fa","Po"])
X_test2["BsmtQual"]=categorize(test2.BsmtQual,["Ex","Gd","TA","Fa","Po","NA"])
X_test2.fillna(X_test2.mean(),inplace=True)
# Now predict again
# Fit the model
model4=LinearRegression()
model4.fit(X_train2,train["SalePrice"])
parameters=pd.DataFrame(model4.coef_,X_train2.columns,["Parameters"])
parameters

# Predict using the test-data
predictions=model4.predict(X_test2)
solution4=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution4.loc[solution4["SalePrice"]<0, "SalePrice"] = 0
solution4.to_csv("solution.csv", index=False)
# Result: 
# This improved my result.
# 0.20571
# Place 3033
# Let's figure out what Normalization of data does.
# At first only Salesprice
train["SalePrice"].plot.hist(bins=50)
# On to do list: Read Scikit.learn documentation, e.g. on Pre-Processing:
# http://scikit-learn.org/stable/modules/preprocessing.html
train.SalePrice[0:5]
mean=train.SalePrice.mean()
train.SalePrice.mean()
Standard_error=train.SalePrice.std()
train.SalePrice.std()
from sklearn import preprocessing
y_scaled = preprocessing.scale(train.SalePrice)
sns.distplot(y_scaled)
y_scaled[0:5]
y_original=y_scaled*Standard_error+mean
y_original[0:5]
# Does this change prediction?
model4.fit(X_train2,y_scaled)
predictions=model4.predict(X_test2)
#Scaling y back to original
predictions=predictions*Standard_error+mean
predictions[predictions<0]=0
solution4=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution4.to_csv("solution.csv", index=False)
# Result: 
# Not at all
# Lets try with the other features
for item in ["OverallQual","GrLivArea","GarageArea","TotalBsmtSF","1stFlrSF", "ExterQual","BsmtQual"]:
    train_scale=preprocessing.scale(X_train2[item])
    test_scale=preprocessing.scale(X_test2[item])
    X_train2[item]=train_scale
    X_test2[item]=test_scale
# Does this change prediction?
model4.fit(X_train2,y_scaled)
predictions=model4.predict(X_test2)
#Scaling y back to original
predictions=predictions*Standard_error+mean
predictions[predictions<0]=0
solution4=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution4.to_csv("solution.csv", index=False)
# Result: 
# Indeed: 0.19900 (from 0.20571)
sns.pairplot(X_train2)
# Just plotted for fun. Doesnt make much sense to plot after normaliza
# Just learned that DecisionTree / RandomForest can also work out regressions and want to give it a try
from sklearn.tree import DecisionTreeRegressor
dtree=DecisionTreeRegressor()
dtree.fit(X_train2,y_scaled)
predictions=dtree.predict(X_test2)
predictions=predictions*Standard_error+mean
predictions[predictions<0]=0
solution5=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution5.to_csv("solution.csv", index=False)
# Result: 
# Worse: 0.22194
# So lets try a Random Forest
from sklearn.ensemble import RandomForestRegressor
RFR=RandomForestRegressor()
RFR.fit(X_train2,y_scaled)
predictions=RFR.predict(X_test2)
predictions=predictions*Standard_error+mean
predictions[predictions<0]=0
solution5=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution5.to_csv("solution.csv", index=False)
# Result: 
# Great: 0.17746 - 500 places up
# And not yet in any way optimized
# And because its working so fine - also KNN can be a regressor:
from sklearn.neighbors import KNeighborsRegressor
KNR=KNeighborsRegressor(n_neighbors=7)
KNR.fit(X_train2,y_scaled)
predictions=KNR.predict(X_test2)
predictions=predictions*Standard_error+mean
predictions[predictions<0]=0
solution5=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution5.to_csv("solution.csv", index=False)
# Result: 
# For Standard n=5:
# Better than Random Forest: 0.17339 - 40 more places
# n=10 worse: 0.17502
# n=2 worse: 0.18261
# n=7 
# Having read the sklearn documentation http://scikit-learn.org/stable/modules/neighbors.html#regression
# lets try some model-options:
# Under some circumstances, it can be advantageous to weight points such that nearby points contribute more to the regression than faraway points. This can be accomplished through the weights keyword. The default value, weights = 'uniform', assigns equal weights to all points. weights = 'distance'
KNR=KNeighborsRegressor(n_neighbors=7, weights = 'distance')
KNR.fit(X_train2,y_scaled)
predictions=KNR.predict(X_test2)
predictions=predictions*Standard_error+mean
predictions[predictions<0]=0
solution5=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution5.to_csv("solution.csv", index=False)
# Result: 
# Better: 0.16996 - 40 places up
# Lets see how a Support Vector Machine works on regression
from sklearn.svm import SVR
model=SVR()
model.fit(X_train2,y_scaled)
predictions=model.predict(X_test2)
predictions=predictions*Standard_error+mean
predictions[predictions<0]=0
solution5=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution5.to_csv("solution.csv", index=False)
# Result: 
# 1 place up: 0.16989
"""# Just learned about GridSearchCV, lets try
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf','sigmoid']} 
# from documentation VR:
# Penalty parameter C of the error term
# gamma = Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
# kernel = Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.
grid=GridSearchCV(SVR,param_grid,refit=True, verbose=2)
grid.fit(X_train2,y_scaled)
predictions = grid.predict(X_test2)
predictions=predictions*Standard_error+mean
predictions[predictions<0]=0
solution5=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution5.to_csv("solution.csv", index=False)
# Result:
# Quite a boost: 0.16033, 120 places up"""
"""# Lets see what boosting the knn regression model results in
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1,200,20), 'weights': ['uniform','distance'],'algorithm':['auto','ball_tree', 'kd_tree', 'brute'], 'p': np.arange(1,10,2), 'leaf_size':np.arange(10,50,10)} 
# from documentation knr:
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
grid=GridSearchCV(KNR,param_grid,refit=True, verbose=2)
grid.fit(X_train2,y_scaled)
predictions = grid.predict(X_test2)
predictions=predictions*Standard_error+mean
predictions[predictions<0]=0
solution5=pd.DataFrame({"id":test.Id, "SalePrice":predictions})
solution5.to_csv("solution.csv", index=False)
# Result:
# Not an improvement: 0.17594"""
# Learned about Principal Component Analysis
# The method to find a "Principal component" (a vector) in the data that explains the biggest part of the variance in the data
# This way it is possible to explain the data with very few dimensions
# However, the interpretation of these dimensions / vectors / principal components is not straight forward as it is a mix of the features
# Let's try it out on the first versions of the data were I had lots of features:
X_train.head()
test.head()
X_train.info()
test.info()
#PCA requires scaling of data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_train)
#scaler.fit(test)
#test_scaled = scaler.transform(test)
from sklearn.decomposition import PCA
# Define the first two principal components
pca = PCA(n_components=2)
pca.fit(X_test_scaled)
x_pca = pca.transform(X_test_scaled)
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
# Ok what does this mean?
pca.components_
# Results in two times 300 weights, i.e. how do the 300 features influence the principal component 1 and 2
df_comp = pd.DataFrame(pca.components_,columns=X_train.columns)
plt.figure(figsize=(20,6))
sns.heatmap(df_comp,cmap='plasma',)
df_comp.head()
df_diff=df_comp.loc[0]-df_comp.loc[1]
df_diff.sort_values().head(10)
df_diff.sort_values(ascending=False).head(15)