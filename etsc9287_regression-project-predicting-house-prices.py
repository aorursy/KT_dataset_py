import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.model_selection import learning_curve

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import RANSACRegressor



import statsmodels.api as sm

import statsmodels.formula.api as smf



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train.info()
def missing_props(df):

    missing_values = []

    for i in df.columns:

        missing_values.append(round(df[i].isnull().sum() / len(df), 3))

    missing_props = pd.DataFrame(list(zip(df.columns, missing_values)), columns = ["Var", "Prop_Missing"]).sort_values(by = "Prop_Missing", ascending = False)

    

    return missing_props[missing_props["Prop_Missing"] != 0]
table1 = missing_props(train)
table1
train = train.drop(columns = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"])
#Lot Frontage is the amount of street connected to the property (ft).  NA values represent 

#houses with no connected street, probably in rural areas.



train["LotFrontage"] = train["LotFrontage"].fillna(0)
#All garage variables have the same amount of missing values, illustrating that these values 

#are within the same rows.  These houses have no garage.



train["GarageYrBlt"] = train["GarageYrBlt"].fillna("No Garage")

train["GarageCond"] = train["GarageCond"].fillna("No Garage")

train["GarageType"] = train["GarageType"].fillna("No Garage")

train["GarageFinish"] = train["GarageFinish"].fillna("No Garage")

train["GarageQual"] = train["GarageQual"].fillna("No Garage")
#All basement variables have the same amount of missing values, illustrating that these values 

#are within the same rows.  These houses have no basement.



train["BsmtFinType1"] = train["BsmtFinType1"].fillna("No Basement")

train["BsmtFinType2"] = train["BsmtFinType2"].fillna("No Basement")

train["BsmtExposure"] = train["BsmtExposure"].fillna("No Basement")

train["BsmtQual"] = train["BsmtQual"].fillna("No Basement")

train["BsmtCond"] = train["BsmtCond"].fillna("No Basement")
#These missing values represent houses with no masonry vaneers.



train["MasVnrArea"] = train["MasVnrArea"].fillna(0)

train["MasVnrType"] = train["MasVnrType"].fillna("None")
#Lastly, these missing values have no known electrical system.



train["Electrical"] = train["Electrical"].fillna("None")
train.head()
train["TotalInsideArea"] = train["TotalBsmtSF"] + train["GrLivArea"] + train["GarageArea"]
train["TotalOutsideArea"] = train["WoodDeckSF"] + train["OpenPorchSF"] + train["EnclosedPorch"] + train["3SsnPorch"] + train["ScreenPorch"] + train["PoolArea"]
train["Pool"] = train["PoolArea"].apply(lambda x: "Yes" if x > 0 else "No")
train = train.drop(columns = ["TotalBsmtSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF",

                              "EnclosedPorch", "PoolArea", "3SsnPorch", "ScreenPorch", "1stFlrSF", "2ndFlrSF",

                             "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "MasVnrArea"])
import statsmodels.api as sm
def model_summary(df):

    

    x = df.drop(columns = ["SalePrice"]).select_dtypes(exclude=['object'])

    y = df["SalePrice"].values

    

    model_original = sm.OLS(y, x).fit()

    return model_original.summary()
model_summary(train)
corr_matrix = train.corr()

pd.options.display.float_format = "{:,.2f}".format

plt.figure(figsize = (12,10))

sns.heatmap(corr_matrix)

plt.title("Correlation Matrix of All Continuous Variables (Target = SalePrice)")

plt.show()
var_corrs = pd.DataFrame((corr_matrix["SalePrice"] ** 2).sort_values(ascending = False))

var_corrs
def corr_matrix_filter(corr_matrix, r2): # <- We can experiment with different r^2 thresholds when we tune our model

    

    corr_matrix2 = corr_matrix[corr_matrix["SalePrice"] ** 2 >= r2]

    

    low_corrs = list(corr_matrix[corr_matrix["SalePrice"] ** 2 < r2].index)

    

    if r2 != 0:

        corr_matrix2 = corr_matrix2.drop(columns = low_corrs)



    return corr_matrix2
corr_matrix2 = corr_matrix_filter(corr_matrix, 0.1) #only includes variables with r^2 >= 0.1

pd.options.display.float_format = "{:,.2f}".format

plt.figure(figsize = (12,10))

sns.heatmap(corr_matrix2, annot = True)

plt.title("Filtered Correlation Matrix of Continuous Variables (SalePrice R^2 > 0.1)")

plt.savefig("corrmap1.png")

plt.show()
pearson_selection_cont = list(corr_matrix2.index)

pearson_selection_cont
train_cont = train[pearson_selection_cont]

model_summary(train_cont)
cat_vars = train.select_dtypes(include = "object")
cat_vars_encode = pd.get_dummies(cat_vars)
cat_vars_encode["SalePrice"] = train["SalePrice"]
cat_vars_encode.head()
corr_matrix_cat = cat_vars_encode.corr()
var_corrs_cat = pd.DataFrame((corr_matrix_cat["SalePrice"] ** 2).sort_values(ascending = False))

var_corrs_cat.head(22)
corr_matrix_cat2 = corr_matrix_filter(corr_matrix_cat, 0.1)
plt.figure(figsize = (12,10))

sns.heatmap(corr_matrix_cat2)

plt.title("Filtered Correlation Matrix of Categorical Variables (SalePrice R^2 > 0.1)")

plt.savefig("corrmap2.png")

plt.show()
relevant_cat_vars = list(corr_matrix_cat2.index)
remove_underscores = []

for i in relevant_cat_vars:

    remove_underscores.append(i.split("_", 1)[0]) #Splits the variables from their distinct values

pearson_selection_cat = list(set(remove_underscores)) #Removes repeats to get list of relevant variables.
pearson_selection_cat
pearson_selection_cat.remove('SalePrice') #The target variable, sale price, is already included in continous list
train_cat = train[pearson_selection_cat]
train2 = pd.concat([train_cont, train_cat], axis = 1)
train2.head()
list(train2.columns)
def graph_cont(df, var):

    plt.hist(df[var])

    plt.xlabel(f"{var}")

    plt.ylabel("Count")

    plt.title(f"Distribution of {var}")

    plt.show()
[graph_cont(train_cont, i) for i in train_cont.columns]
def graph_disc(df, var):

    sns.countplot(df[var])

    plt.xlabel(f"{var}")

    plt.ylabel("Count")

    plt.title(f"Distribution of {var}")

    plt.xticks(rotation = 90)

    plt.show()
[graph_disc(train_cat, i) for i in train_cat.columns]
def scatter_plot(df, var):

    plt.scatter(df[var], df["SalePrice"])

    plt.xlabel(f"{var}")

    plt.ylabel("SalePrice")

    plt.title(f"{var} vs. SalePrice")

    plt.xticks(rotation = 90)

    plt.show()
[scatter_plot(train_cont, i) for i in train_cont.columns]
def box_plot(train2, df, var):

    df["SalePrice"] = train2["SalePrice"]

    sns.boxplot(df[var], df["SalePrice"])

    plt.xlabel(f"{var}")

    plt.ylabel("SalePrice")

    plt.title(f"{var} vs. SalePrice")

    plt.xticks(rotation = 90)

    plt.show()
[box_plot(train2, train_cat, i) for i in train_cat.columns if i != "SalePrice"]
for i in list(train2.select_dtypes(exclude = "object").columns):

    sm.qqplot(train2[i])

    plt.title(f"QQ Plot for {i}")

    plt.show()
train2["SalePrice"] = np.log1p(train2["SalePrice"])
graph_cont(train2, "SalePrice")
train2["TotalInsideArea"] = train_cont["TotalInsideArea"]

train2["TotalInsideArea"] = np.log1p(train2["TotalInsideArea"])
graph_cont(train2, "TotalInsideArea")
scatter_plot(train2, "TotalInsideArea")
train2 = train2[(train2["SalePrice"] > 11) & (train2["TotalInsideArea"] < 9)]
scatter_plot(train2, "TotalInsideArea")
train2["TotalOutsideArea"] = np.log1p(train2["TotalOutsideArea"])
#train2 = train2[train2["TotalOutsideArea"] != 0]
len(list(train[train["TotalOutsideArea"] == 0]))
graph_cont(train2, "TotalOutsideArea")
scatter_plot(train2, "TotalOutsideArea")
train_set = pd.get_dummies(train2)
x_vals = train_set.drop(columns = ["SalePrice"])

y_val = train_set["SalePrice"].values.reshape(-1,1)
import math



def RMSLE(predict, target):

    

    total = 0 

    

    for k in range(len(predict)):

        

        LPred= np.log1p(predict[k]+1)

        LTarg = np.log1p(target[k] + 1)

        

        if not (math.isnan(LPred)) and  not (math.isnan(LTarg)): 

            

            total = total + ((LPred-LTarg) **2)

        

    total = total / len(predict)  

    

    return np.sqrt(total)
def create_model(x_vals, y_val, model_type, t):

    

    x_train, x_test, y_train, y_test = train_test_split(x_vals, y_val, test_size = 0.2) #splitting into train and test

    

    model = model_type

    model.fit(x_train, y_train) #fitting the model

    

    y_train_pred = np.expm1(model.predict(x_train)) #predicting and converting back from log(SalePrice)

    y_test_pred = np.expm1(model.predict(x_test))

    y_train = np.expm1(y_train)

    y_test = np.expm1(y_test)

    

    if t == "test":

        return RMSLE(y_test_pred, y_test) #evaluating

    elif t == "train":

        return RMSLE(y_train_pred, y_train)
print("Average Train Accuracy:", round(np.mean([create_model(x_vals, y_val, LinearRegression(), "train") for i in range(100)]), 4))

print("Average Test Accuracy:", round(np.mean([create_model(x_vals, y_val, LinearRegression(), "test") for i in range(100)]), 4))
print("Average Train Accuracy:", round(np.mean([create_model(x_vals, y_val, Ridge(alpha = 0.01), "train") for i in range(100)]), 4))

print("Average Test Accuracy:", round(np.mean([create_model(x_vals, y_val, Ridge(alpha = 0.01), "test") for i in range(100)]), 4))
print("Average Train Accuracy:", round(np.mean([create_model(x_vals, y_val, Lasso(alpha = 0.01), "train") for i in range(100)]), 4))

print("Average Test Accuracy:", round(np.mean([create_model(x_vals, y_val, Lasso(alpha = 0.01), "test") for i in range(100)]), 4))
print("Average Train Accuracy:", round(np.mean([create_model(x_vals, y_val, ElasticNet(alpha = 0.01), "train") for i in range(100)]), 4))

print("Average Test Accuracy:", round(np.mean([create_model(x_vals, y_val, ElasticNet(alpha = 0.01), "test") for i in range(100)]), 4))
print("Average Train Accuracy:", round(np.mean([create_model(x_vals, y_val, RANSACRegressor(), "train") for i in range(100)]), 4))

print("Average Test Accuracy:", round(np.mean([create_model(x_vals, y_val, RANSACRegressor(), "test") for i in range(100)]), 4))
train_sizes, train_scores, validation_scores = learning_curve(

estimator = Ridge(),

X = x_vals,

y = y_val, train_sizes = [1,10,50,100,300,600,900], cv = 5,

scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)

validation_scores_mean = -validation_scores.mean(axis = 1)
plt.plot(train_sizes, train_scores_mean, label = 'Training error')

plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('MSE')

plt.xlabel('Training set size')

plt.title('Learning curves for Linear regression model')

plt.legend()

plt.show()
#cleaning and feature engineering



test["TotalBsmtSF"] = test["TotalBsmtSF"].fillna(0)

test["GrLivArea"] = test["GrLivArea"].fillna(0)

test["GarageArea"] = test["GarageArea"].fillna(0)

test["GarageCars"] = test["GarageCars"].fillna(0)

test["GarageYrBlt"] = test["GarageYrBlt"].fillna("No Garage")

test["GarageCond"] = test["GarageCond"].fillna("No Garage")

test["GarageType"] = test["GarageType"].fillna("No Garage")

test["GarageFinish"] = test["GarageFinish"].fillna("No Garage")

test["GarageQual"] = test["GarageQual"].fillna("No Garage")

test["BsmtFinType1"] = test["BsmtFinType1"].fillna("No Basement")

test["BsmtFinType2"] = test["BsmtFinType2"].fillna("No Basement")

test["BsmtExposure"] = train["BsmtExposure"].fillna("No Basement")

test["BsmtQual"] = test["BsmtQual"].fillna("No Basement")

test["BsmtCond"] = test["BsmtCond"].fillna("No Basement")

test["MasVnrArea"] = test["MasVnrArea"].fillna(0)

test["MasVnrType"] = test["MasVnrType"].fillna("None")

test["TotalInsideArea"] = test["TotalBsmtSF"] + test["GrLivArea"] + test["GarageArea"]

test["TotalOutsideArea"] = test["WoodDeckSF"] + test["OpenPorchSF"] + test["EnclosedPorch"] + test["3SsnPorch"] + test["ScreenPorch"] + test["PoolArea"]

train2_drop = train2.drop(columns = ["SalePrice"])
test.info()
#creating test and train



x_train = train_set.drop(columns = ["SalePrice"])

y_train = train_set["SalePrice"].values

x_test = test[train2_drop.columns] #accessing only the features I selected earlier in the notebook

x_test = pd.get_dummies(x_test)



#preprocessing



x_test["TotalInsideArea"] = np.log1p(x_test["TotalInsideArea"])

x_train = x_train[x_train["TotalInsideArea"] < 9]

x_test["TotalOutsideArea"] = np.log1p(x_test["TotalOutsideArea"])

x_train = x_train[x_train["TotalOutsideArea"] != 0]



#fitting the Ridge model (most accurate from earlier)

    

model = Ridge()

model.fit(x_train, y_train) #fitting the model

    

y_train_pred = np.expm1(model.predict(x_train)) #predicting and converting back from log(SalePrice)

y_test_pred = np.expm1(model.predict(x_test))
x_train.info()
y_train_pred
y_test_pred
preds = y_test_pred
ids = np.array(test["Id"])
submissions = pd.DataFrame({"Id":ids, "SalePrice":preds})
submissions.head()
submissions.to_csv("submission2.csv", index = False)