import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.linear_model import LinearRegression



from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.model_selection import cross_val_score, train_test_split



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/train.csv")
#Combining the test & train dataframes and removing Id and SalePrice(Target Variable)

total_data = pd.concat((train.loc[:,"MSSubClass":"SaleCondition"],test.loc[:,"MSSubClass":"SaleCondition"]))
total_data.head(10)
plt.figure()

sns.distplot(train["SalePrice"],fit = norm)



plt.figure()

stats.probplot(train['SalePrice'], plot=plt)



plt.show()


y = np.log(train["SalePrice"])

# We are creating a new variable sale_log for further use.While making a model we will use sales_Log.

# For now I will be using SalePrice 

plt.figure()

sns.distplot(y,fit = norm)



plt.figure()

stats.probplot(y, plot=plt)



plt.show()
num_cols =  train.select_dtypes(exclude=["object"]).columns
corrmat = train[num_cols].corr()

mask = np.zeros_like(corrmat,dtype=bool)

mask[np.triu_indices_from(mask)] = True

f,ax = plt.subplots(figsize = (11,9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corrmat, mask=mask, cmap=cmap,center=0 )

plt.show()
k=10

cols =  corrmat.nlargest(k,"SalePrice").index

cm  = np.corrcoef(train[cols].values.T)

sns.set(font_scale  =1.25)

hm = sns.heatmap(cm,cbar=True,annot = True,square =True,fmt=" .2f",annot_kws={"size":10},

                 yticklabels = cols.values,xticklabels = cols.values)
# Highest correlated variables 

# TotalBsmtSF

plt.figure()

plt.subplot()

sns.scatterplot(train["TotalBsmtSF"],train["SalePrice"]).axis(ymin=0, ymax=800000)





#GarageArea: Size of garage in square feet

# Well the  data is scattered every where it has some outliers and dif. has some unsual values like area = 0 

plt.figure()

plt.subplot()

sns.scatterplot(train["GarageArea"],train["SalePrice"]).axis(ymin=0, ymax=800000)





# GrLivArea: Above grade (ground) living area square feet

# there some few outliers 

plt.figure()

plt.subplot()

sns.scatterplot(train["GrLivArea"],train["SalePrice"]).axis(ymin=0, ymax=800000)

plt.show()



plt.figure()

plt.subplot()

sns.scatterplot(train["1stFlrSF"],train["SalePrice"]).axis(ymin=0, ymax=800000)



plt.figure()

plt.subplot()

sns.scatterplot(train["YearBuilt"],train["SalePrice"]).axis(ymin=0, ymax=800000)



plt.show()



plt.figure(figsize=(5,5,))

sns.barplot(train["OverallQual"], train["SalePrice"]).axis(ymin=0, ymax=800000)



plt.figure(figsize=(5,5,))

sns.barplot(train["TotRmsAbvGrd"], train["SalePrice"]).axis(ymin=0, ymax=800000)



plt.figure(figsize=(5,5,))

sns.barplot(train["FullBath"], train["SalePrice"]).axis(ymin=0, ymax=800000)





plt.show()
# Missing data

missing_data= total_data.isnull().sum().sort_values(ascending = False)

missing_data[missing_data >0]
# Removing cols with more than 90% data missing

total_data=total_data.drop("PoolQC",axis=1)

total_data=total_data.drop("MiscFeature",axis=1)

total_data=total_data.drop("Alley",axis=1)

total_data=total_data.drop("Fence",axis=1)
# Cleaning the data

total_data.FireplaceQu[total_data["FireplaceQu"].isnull()]="No"

#LotFrontage: Linear feet of street connected to property . 

total_data.LotFrontage[total_data["LotFrontage"].isnull()]= 0



total_data.GarageType[total_data["GarageType"].isnull()] = "No"

total_data.GarageCond[total_data["GarageCond"].isnull()] = "No"

total_data.GarageFinish[total_data["GarageFinish"].isnull()] = "No"

total_data.GarageQual[total_data["GarageQual"].isnull()] = "No"

total_data.GarageYrBlt[total_data["GarageYrBlt"].isnull()] = 0





total_data.BsmtFinType2[total_data["BsmtFinType2"].isnull()] = "No"

total_data.BsmtExposure[total_data["BsmtExposure"].isnull()] = "No"

total_data.BsmtQual[total_data["BsmtQual"].isnull()] ="No"

total_data.BsmtCond[total_data["BsmtCond"].isnull()] = "No"

total_data.BsmtFinType1[total_data["BsmtFinType1"].isnull()] = "No"



                   

total_data.MasVnrType[total_data["MasVnrType"].isnull()] = "No"

total_data.MasVnrArea[total_data["MasVnrArea"].isnull()] = 0





total_data.Electrical[total_data["Electrical"].isnull()] = "SBrkr"



total_data.MSZoning[total_data["MSZoning"].isnull()] = "RL"

total_data.Functional[total_data["Functional"].isnull()] = "Typ"

total_data.Utilities[total_data["Utilities"].isnull()] = "AllPub"

total_data.BsmtFullBath[total_data["BsmtFullBath"].isnull()] = 0

total_data.BsmtHalfBath[total_data["BsmtHalfBath"].isnull()] = 0

total_data.GarageCars[total_data["GarageCars"].isnull()] = 0

total_data.BsmtFinSF2[total_data["BsmtFinSF2"].isnull()] = 0

total_data.BsmtUnfSF[total_data["BsmtUnfSF"].isnull()] = 0

total_data.TotalBsmtSF[total_data["TotalBsmtSF"].isnull()] = 0



total_data.SaleType[total_data["SaleType"].isnull()] = "WD"

total_data.Exterior2nd[total_data['Exterior2nd'].isnull()] = "VinylSd"



total_data.Exterior1st[total_data['Exterior1st'].isnull()] = "VinylSd"



total_data.KitchenQual[total_data['KitchenQual'].isnull()] = "TA"



total_data.BsmtFinSF1[total_data["BsmtFinSF1"].isnull()] = 0

total_data.GarageArea[total_data["GarageArea"].isnull()] = 0



# Check that all there is no NA value

total_data.isnull().sum().max()
# I would like to encode categorical ordinal data 

total_data= total_data.replace({'Street':{'Pave':2,'Grvl':1},

                       'BldgType': {'1Fam':5 ,'2fmCon':3, 'Duplex':3, 'TwnhsE':4 ,'Twnhs':3},

                       

                       'HouseStyle': {'2Story':7 ,'1Story':6 ,'1.5Fin':4 ,'1.5Unf':1, 'SFoyer':2, 'SLvl':5 ,'2.5Unf':3,'2.5Fin':8},

                       

                       'ExterQual': {'Gd':4, 'TA':3, 'Ex':5, 'Fa':2},

                       

                       'ExterCond': {'TA':3, 'Gd':4, 'Fa':2 ,'Po':1 ,'Ex':5},

                       

                       "BsmtQual": {'TA':3 ,'Gd':4 ,'Fa':2, 'Po':1 ,'Ex':5,"No":0},

                       

                       "BsmtCond":{'TA':3 ,'Gd':4 ,'Fa':2, 'Po':1 ,'Ex':5,"No":0},

                        

                        "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                       

                    



                       

                       "HeatingQC":{'TA':3 ,'Gd':4 ,'Fa':2, 'Po':1 ,'Ex':5,'No':0},

                       

                       "FireplaceQu":{'TA':3 ,'Gd':4 ,'Fa':2, 'Po':1 ,'Ex':5,'No':0},

                       

                       "GarageCond":{'TA':3 ,'Gd':4 ,'Fa':2, 'Po':1 ,'Ex':5,'No':0},

                       

                       "GarageQual":{'TA':3 ,'Gd':4 ,'Fa':2, 'Po':1 ,'Ex':5,'No':0},

                       

                       "KitchenQual":{'TA':3 ,'Gd':4 ,'Fa':2, 'Po':1 ,'Ex':5,'No':0},

                       

                        "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       

                        "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

                        "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

                                         

                       

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       

                       "GarageFinish": {'RFn':2 ,'Unf':1 ,'Fin':3 ,"No":0},

                       

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       

                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}})

                              

                                

                      

                                       

                       

# Feauter Engineering 



total_data["Garage_Score"] = total_data["GarageCond"] + total_data["GarageQual"] + total_data["GarageFinish"]



                          



total_data["TotalBsmtFinSF"] = total_data["BsmtFinSF1"] + total_data["BsmtFinSF2"]

    

total_data["Basment_Score"] = total_data["BsmtQual"] + total_data["BsmtCond"] + total_data["BsmtFinType2"] + total_data["BsmtFinType1"] + total_data["BsmtExposure"]





total_data["total_flr_area"] = total_data["1stFlrSF"] + total_data["2ndFlrSF"]





total_data["Total_gr_area"] = total_data["GrLivArea"] + total_data["TotalBsmtSF"]



total_data["Exter_Score"] = total_data["ExterCond"] + total_data["ExterQual"]





total_data["Overall_Score"] = (total_data["OverallCond"] + total_data["OverallQual"])/2





# House Age 

total_data["House_Age"] = total_data["YearRemodAdd"] - total_data["YearBuilt"]

total_data["Total_Porch"] = total_data["OpenPorchSF"] + total_data["EnclosedPorch"] + total_data["3SsnPorch"] +  total_data["ScreenPorch"] + total_data["WoodDeckSF"]

                               



total_data["Total_baths"] = total_data["BsmtFullBath"] + 0.5*total_data["BsmtHalfBath"] + total_data["FullBath"] + 0.5*total_data["HalfBath"]







total_data.columns
numeric_cols = total_data.dtypes[total_data.dtypes != "object"].index
numeric_cols = total_data[numeric_cols].apply(lambda x:stats.skew(x))
skew_cols = numeric_cols[abs(numeric_cols) > 0.75 ].index

total_data[skew_cols] = np.log1p(total_data[skew_cols])

total_data = pd.get_dummies(total_data)
from sklearn.linear_model import Ridge, LassoCV

from sklearn.model_selection import cross_val_score
X_train = total_data[:train.shape[0]]

X_test = total_data[train.shape[0]:]

Y = y
# Defining our valuation function

def rmse_cv(model):

    rmse = np.sqrt(-cross_val_score(model,X_train,y,scoring = "neg_mean_squared_error",cv = 5))

    return rmse
model = Ridge()
# Hyper parameter tuning .

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation ")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
model_lasso = LassoCV(alphas = [1,0.1,0.0001,0.0005]).fit(X_train,y)
rmse_cv(model_lasso)
rmse_cv(model_lasso).min()
X_test = np.nan_to_num(X_test)
Y_pred = np.expm1(model_lasso.predict(X_test))
Final = pd.DataFrame()
Final["Id"] =test.Id

Final["SalePrice"] = Y_pred
Final.to_csv("Housing.csv",index=False )