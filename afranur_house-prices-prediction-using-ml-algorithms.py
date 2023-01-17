import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df.head(3)
df.shape
# Type of datas
df.info()
# To see all rows;
pd.options.display.max_rows
pd.set_option("display.max_rows",None)


# Any missing values in dataset?
df.isnull().sum().sort_values(ascending=False)
# Now, let's look at the percentage of null values to decide dropping them

print("Percentage of PoolQC: % {:.4f}".format((df["PoolQC"].isnull().sum())/len(df)*100))
print("Percentage of MiscFeature: % {:.4f}".format((df["MiscFeature"].isnull().sum())/len(df)*100))
print("Percentage of Alley: % {:.4f}".format((df["Alley"].isnull().sum())/len(df)*100))
print("Percentage of Fence: % {:.4f}".format((df["Fence"].isnull().sum())/len(df)*100))
print("Percentage of FireplaceQu: % {:.4f}".format((df["FireplaceQu"].isnull().sum())/len(df)*100))
print("Percentage of LotFrontage: % {:.4f}".format((df["LotFrontage"].isnull().sum())/len(df)*100))
# Let'S visualize, how much are there missing values in this dataset?

miss=df.isnull().sum().sort_values(ascending=False).head(20)
plt.figure(figsize=(8,5),dpi=100)
miss_val=pd.Series(miss, miss.index)
miss_val.plot(kind="barh");
# We can drop first 5 columns and "id" column

df.drop(["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","Id"], axis=1, inplace=True)

df.head(2)
# Fill with mean() function the numerical variables and fill with mode() function the categorical variables

df["LotFrontage"].fillna(df["LotFrontage"].mean(), inplace=True)
df["GarageType"].fillna(df["GarageType"].mode()[0], inplace=True)
df["GarageYrBlt"].fillna(df["GarageYrBlt"].mode()[0], inplace=True)
df["GarageFinish"].fillna(df["GarageFinish"].mode()[0], inplace=True)
df["GarageCond"].fillna(df["GarageCond"].mode()[0], inplace=True)
df["GarageQual"].fillna(df["GarageQual"].mode()[0], inplace=True)
df["BsmtExposure"].fillna(df["BsmtExposure"].mode()[0], inplace=True)
df["BsmtFinType2"].fillna(df["BsmtFinType2"].mode()[0], inplace=True)
df["BsmtFinType1"].fillna(df["BsmtFinType1"].mode()[0], inplace=True)
df["BsmtCond"].fillna(df["BsmtCond"].mode()[0], inplace=True)
df["BsmtQual"].fillna(df["BsmtQual"].mode()[0], inplace=True)
df["MasVnrType"].fillna(df["MasVnrType"].mode()[0], inplace=True)
df["MasVnrArea"].fillna(df["MasVnrArea"].mode()[0], inplace=True)
df["Electrical"].fillna(df["Electrical"].mode()[0], inplace=True)

print("Are there any missing values in all dataset: ", (df.isnull().sum()>0).any())
# We should seperate categorical and continuous variable:

df_cat=df.select_dtypes(include="O")
df_num=df.select_dtypes(exclude="O")

display(df_cat.head(2))
display(df_num.head(2))
df_cat_col=df.select_dtypes(exclude="O").columns
df_year=pd.DataFrame()

# "YrSold" column must be higher than other columns in df_year dataframe, so we should check it:

for k in range(len(df_cat_col)):
    if ("Year" in df_cat_col[k]) or ("Yr" in df_cat_col[k]):
        df_year=df_year.append(df[df_cat_col[k]]) 
        
df_year=df_year.T        
display(df_year.head()) 

print("YearBuilt column is higher than YrSold column", np.where(df_year["YearBuilt"]>df_year["YrSold"]),"\n")
print("YearRemodAdd column is higher than YrSold column", np.where(df_year["YearRemodAdd"]>df_year["YrSold"]),"\n")
print("GarageYrBlt column is higher than YrSold column", np.where(df_year["GarageYrBlt"]>df_year["YrSold"]))

# YearRemodAdd column has a wrong value, so we must change it
for index, row in df_year.iterrows():
    if row["YearRemodAdd"]>row["YrSold"]:
        print(index, row["YearRemodAdd"], row["YrSold"])

# At 523. row, YearRemodAdd is higher than YrSold, now i will replace "2008.0 "
df_num["YearRemodAdd"][523]=2007
# We changed the specific row with 2007
df_num.hist(layout=(10,8), figsize=(28,25))

plt.show()

# Like we see in the graphs, most of numerical variables is not the normal distributions, to predict target value, we should normalize them
# OUTLIERS:

df_num_col = list(df_num.iloc[:,:36].columns)

plt.figure(figsize=(28,25),dpi=100)
for i in range(len(df_num_col)):
    plt.subplot(6,6,i+1)
    plt.title(df_num_col[i])
    plt.boxplot(df_num[df_num_col[i]]);
from scipy import stats

# First, look at "LotFrontage" column to understand zscore

z_lot=stats.zscore(df_num.LotFrontage)

for i in range(1,7):
    print("Threshold value: ", i)
    print("Number of outliers:", len(np.where(z_lot>i)[0]),"\n")

print(len(np.where(z_lot>3)[0]))
# If it is applied log transformation to "LotFrontage" column, number of outliers:

df_num["LotFrontage_log"] = np.log(df_num["LotFrontage"])
zscore_log=stats.zscore(df_num["LotFrontage_log"])
print(len(np.where(zscore_log>2.5)[0]))
plt.boxplot(zscore_log, whis=2.5);

# Then drop it
df_num.drop(["LotFrontage_log"], axis=1, inplace=True)
# I have chosen in below "outlier" list because, they are continuous variables

outlier=["LotFrontage","LotArea","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","LowQualFinSF",
         "GrLivArea","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal"]


for i in range(len(outlier)):
    print("Variable: ", outlier[i])
    zscore_outlier=stats.zscore(df_num[outlier[i]])
    print("Number of outliers higher than threshold 2.5 : ", len(np.where(zscore_outlier>2.5)[0]))
    print("*"*40)

# Let's analyze using percentile method for "LotFrontage" column as an example

Q1 = np.percentile(df_num["LotFrontage"],25)
Q3 = np.percentile(df_num["LotFrontage"],75)

IQR= Q3-Q1
Lower=Q1 - (1.5*IQR)
Upper=Q3 + (1.5*IQR)

print("Number of outliers lower than Lower",len(df_num[df_num["LotFrontage"]<Lower]))
print("Number of outliers higher than Higher", len(df_num[df_num["LotFrontage"]>Upper]))
from scipy.stats.mstats import winsorize

# Now, firstly i will get log transformation continuous variable in "outlier" list and i'll winsorize "outliers_wins" list; then i will cut outliers if threshold is higher than 2.5:
from scipy import stats

outlier=["LotFrontage","LotArea","BsmtUnfSF","TotalBsmtSF","1stFlrSF","GrLivArea","GarageArea"]

for i in range(len(outlier)):
    df_num[outlier[i]]=np.log(df_num[outlier[i]]+1)
    
print("*"*40,"Continuous Variables","*"*40)    
for j in range(len(outlier)):
    print("Variable: ", outlier[j])
    zscore_outlier=stats.zscore(df_num[outlier[j]])
    print("Number of outliers higher than threshold 2 : ", len(np.where(zscore_outlier>2)[0]))
    print("*"*40,"\n")    

 
    
    
outlier_wins=["MasVnrArea","BsmtFinSF1","BsmtFinSF2","LowQualFinSF","WoodDeckSF","OpenPorchSF","EnclosedPorch",
              "3SsnPorch","ScreenPorch","PoolArea","MiscVal"]

for k in range(len(outlier_wins)):
    df_num[outlier_wins[k]]=stats.mstats.winsorize(df_num[outlier_wins[k]], limits=0.05)    

    
print("*"*40,"Categorical Numeric Variables","*"*40)
for j in range(len(outlier_wins)):
    
    print("Variable: ", outlier_wins[j])
    zscore_outlier_wins=stats.zscore(df_num[outlier_wins[j]])
    print("Number of outliers higher than threshold 2 : ", len(np.where(zscore_outlier_wins>2)[0]))
    print("*"*40)  
    
    
# After the log transformation, we get rid of most of outliers.
# I have seperated categorical variable and numerical variable so i will firstly analyze relationship between numerical variables
focus_col=["SalePrice"]

df_num.corr().filter(focus_col).drop(focus_col).abs().sort_values(by="SalePrice", ascending=False).plot(kind="barh", figsize=(15,10));

# According to graph, the most correlated value is "OverallQual".
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X=df_num.drop(["SalePrice"], axis=1)
Y=df_num["SalePrice"]
X_scaled=StandardScaler().fit_transform(X)


pca=PCA(n_components=35)

X_scaled=pca.fit_transform(X_scaled)

exp_var=pca.explained_variance_ratio_
cumsum_exp=np.cumsum(exp_var)

plt.plot(cumsum_exp)
plt.grid();

# According to graph, first 15 columns are important to predict 
X_scaled=pca.fit_transform(X_scaled)

pca_new=PCA(n_components=16)

X_new=pca_new.fit_transform(X_scaled)

exp_var_new=pca_new.explained_variance_ratio_
cumsum_exp_new=np.cumsum(exp_var_new)

plt.plot(cumsum_exp_new)
plt.grid();

X_new_df=pd.DataFrame(X_new)
X_new_df.head(3)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split
dummy=X_new_df.copy()
dummy["target"]=Y


X1=dummy.iloc[:,:16]
Y1=dummy["target"]
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.20, random_state=42)

Y1_train = Y1_train.values.reshape(-1,1)
Y1_test=Y1_test.values.reshape(-1,1)

X1_train=StandardScaler().fit_transform(X1_train)
X1_test=StandardScaler().fit_transform(X1_test)
Y1_train=StandardScaler().fit_transform(Y1_train)
Y1_test=StandardScaler().fit_transform(Y1_test)


forest=RandomForestRegressor(n_estimators=25,
                            random_state=42)

forest.fit(X1_train, Y1_train)

y_test_pred=forest.predict(X1_test)

print("MSE:", mean_squared_error(Y1_test,y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y1_test,y_test_pred)))
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
X5=dummy.iloc[:,:16]
Y5=dummy["target"]
X5_train, X5_test, Y5_train, Y5_test = train_test_split(X5,Y5, test_size=0.20, random_state=42)

Y5_train = Y5_train.values.reshape(-1,1)
Y5_test=Y5_test.values.reshape(-1,1)


X5_train=StandardScaler().fit_transform(X5_train)
X5_test=StandardScaler().fit_transform(X5_test)
Y5_train=StandardScaler().fit_transform(Y5_train)
Y5_test=StandardScaler().fit_transform(Y5_test)

lrm=LinearRegression()

lrm.fit(X5_train,Y5_train)

lrm_y_test_pred=lrm.predict(X5_test)

plt.figure(figsize=(8,3),dpi=100)
plt.subplot(1,2,1)
plt.scatter(Y5_test,lrm_y_test_pred)
plt.plot(Y5_test,Y5_test,c="r");

# Feature importance with linear regression:
plt.subplot(1,2,2)
plt.bar([x for x in range(len(lrm.coef_[0]))], lrm.coef_[0]);
# According to the Gauss Markov's Assumption, errors shouldn't be correlation each other.

error=Y5_test-lrm_y_test_pred
plt.plot(error);
X5_train=sm.add_constant(X5_train)
X5_test=sm.add_constant(X5_test)

predict=sm.OLS(Y5_train,X5_train).fit()
predict.summary()
X2=df_num.drop(["SalePrice"], axis=1)
Y2=df_num["SalePrice"]

# All continuous variable is defined by X2 variable.
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2,test_size=0.20, random_state=42)

Y2_train=Y2_train.values.reshape(-1,1)
Y2_test=Y2_test.values.reshape(-1,1)

X2_train=StandardScaler().fit_transform(X2_train)
X2_test=StandardScaler().fit_transform(X2_test)
Y2_train=StandardScaler().fit_transform(Y2_train)
Y2_test=StandardScaler().fit_transform(Y2_test)
X2_train=sm.add_constant(X2_train)
X2_test=sm.add_constant(X2_test)

pred2=sm.OLS(Y2_train,X2_train).fit()
pred2.summary()
X3=df_num[["LotArea","OverallQual","FullBath","OverallCond", "MasVnrArea", "BsmtFinSF1","TotalBsmtSF","1stFlrSF","2ndFlrSF",
           "LowQualFinSF","GrLivArea","LotFrontage","GarageCars","GarageArea","WoodDeckSF","YearBuilt","Fireplaces","OpenPorchSF"]]
Y3=df_num["SalePrice"]

# These features are meaningful because of p_value<0.05
X3_train,X3_test, Y3_train,Y3_test = train_test_split(X3,Y3, test_size=0.20, random_state=42)

X3_train=sm.add_constant(X3_train)
X3_test=sm.add_constant(X3_test)

predict3=sm.OLS(Y3_train,X3_train).fit()
predict3.summary()
# I will choose below features for continuous variables so we shall keep them.

X8=df_num[["LotArea","OverallQual", "OverallCond","BsmtFinSF1","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea",
          "GarageCars","GarageArea","WoodDeckSF","YearBuilt","Fireplaces","OpenPorchSF"]]
Y8=df_num["SalePrice"]
# Now, I will analyze the categorical variables
df_cat.head(2)
# First, visualize them:

df_cat_col=list(df_cat.columns)
plt.figure(figsize=(28,25),dpi=100)
for i in range(len(df_cat_col)):
    plt.subplot(8,5,i+1)
    sns.countplot(df_cat[df_cat_col[i]]);
# Machine learning algorithm uses only numerical variable, so we should transform them using variety methods; one of them label encoder:

from sklearn.preprocessing import LabelEncoder

df_cat_col=list(df_cat.columns)
for i in range(len(df_cat_col)):
    df_cat[df_cat_col[i]] = LabelEncoder().fit_transform(df_cat[df_cat_col[i]])
    
# For prediction, we should add target variable in this dataset:
df_cat["SalePrice"]=df_num["SalePrice"]
df_cat.head(3)
# I have seperated categorical variable and numerical variable so i will firstly analyze relationship between numerical variables
focus_cols=["SalePrice"]

df_cat.corr().filter(focus_cols).drop(focus_cols).abs().sort_values(by="SalePrice", ascending=False).plot(kind="barh", figsize=(15,10));
X4=df_cat.drop(["SalePrice"], axis=1)
Y4=df_cat["SalePrice"]

X4_train,X4_test,Y4_train,Y4_test= train_test_split(X4,Y4,test_size=0.20, random_state=42)
Y4_train=Y4_train.values.reshape(-1,1)
Y4_test=Y4_test.values.reshape(-1,1)

X4_train=StandardScaler().fit_transform(X4_train)
X4_test=StandardScaler().fit_transform(X4_test)
Y4_train=StandardScaler().fit_transform(Y4_train)
Y4_test=StandardScaler().fit_transform(Y4_test)

X4_train=sm.add_constant(X4_train)
X4_test=sm.add_constant(X4_test)

predict4=sm.OLS(Y4_train,X4_train).fit()
predict4.summary()
from sklearn.ensemble import RandomForestRegressor

X_cat=df_cat.drop(["SalePrice"], axis=1)
Y_cat=df_cat["SalePrice"]

X_cat_train, X_cat_test, Y_cat_train, Y_cat_test = train_test_split(X_cat, Y_cat, test_size=0.20, random_state=42)

forest_cat=RandomForestRegressor()
forest_cat.fit(X_cat_train,Y_cat_train)

feature_import=pd.Series(data=forest_cat.feature_importances_, index=X_cat_train.columns)
feature_import=feature_import.sort_values(ascending=False)
plt.figure(figsize=(10,8),dpi=100)
feature_import.plot(kind="barh");

# It looks like correlation matrix
feature_import.nlargest(15)

# I want to choose the first 15 features.
# Let's concat both categorical variable columns and numerical variable columns

df_cat=df_cat[['ExterQual', 'BsmtQual', 'Neighborhood', 'KitchenQual', 'GarageFinish', 'Exterior2nd', 'HouseStyle', 'Exterior1st',
       "BldgType","SaleCondition","GarageType",'BsmtExposure',"SalePrice","LotShape","CentralAir"]]

df_num=df_num[["LotArea","OverallQual", "OverallCond","BsmtFinSF1","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea",
          "GarageCars","GarageArea","WoodDeckSF","YearBuilt","Fireplaces","OpenPorchSF"]]

df_end=pd.concat([df_cat,df_num], axis=1)
df_end.head(3)
# Target variable should be at the end

cols=[col for col in df_end if col!="SalePrice" ] + ["SalePrice"]
df_end = df_end[cols]

df_end.head(3)
X_end=df_end.drop(["SalePrice"], axis=1)
Y_end=df_end["SalePrice"]
from sklearn.preprocessing import StandardScaler

X_end_train, X_end_test, Y_end_train, Y_end_test = train_test_split(X_end,Y_end, test_size=0.30, random_state=42)

print(len(X_end_train))
print(len(X_end_test))
X_end_train=StandardScaler().fit_transform(X_end_train)
X_end_test=StandardScaler().fit_transform(X_end_test)

Y_end_train=Y_end_train.values.reshape(-1,1)
Y_end_test=Y_end_test.values.reshape(-1,1)

Y_end_train=StandardScaler().fit_transform(Y_end_train)
Y_end_test=StandardScaler().fit_transform(Y_end_test)
lrm_end=LinearRegression(normalize=True)
lrm_end.fit(X_end_train,Y_end_train)

predict_end_test=lrm_end.predict(X_end_test)
predict_end_train=lrm_end.predict(X_end_train)

plt.figure(figsize=(10,4),dpi=100)
plt.subplot(1,2,1)
plt.title("Test set prediction")
plt.scatter(Y_end_test,predict_end_test)
plt.plot(Y_end_test,Y_end_test, c="r")

plt.subplot(1,2,2)
plt.title("Train set prediction")
plt.scatter(Y_end_train,predict_end_train)
plt.plot(Y_end_train,Y_end_train, c="r");
import statsmodels.api as sm

result_end=sm.OLS(Y_end_train,X_end_train).fit()
result_end.summary()
forest_end=RandomForestRegressor(n_estimators=20,
                                max_depth=7,
                                criterion="mae")

forest_end.fit(X_end_train, Y_end_train)

forest_test_pred=forest_end.predict(X_end_test)
forest_train_pred=forest_end.predict(X_end_train)

print("MSE:", mean_squared_error(Y_end_test,forest_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_end_test,forest_test_pred)))

plt.figure(figsize=(7,5),dpi=100)
imp_for=pd.Series(data=forest_end.feature_importances_, index=X_end.columns).sort_values(ascending=False)

imp_for.plot(kind="barh");
from sklearn.model_selection import GridSearchCV
param_forest={"n_estimators":np.arange(10,35,5),
             "criterion":["mse","mae"],
             "max_depth":np.arange(3,10,1)
             }

grid_forest=GridSearchCV(estimator=forest_end,
                        param_grid=param_forest,
                        cv=10
                        )

grid_forest.fit(X_end_train,Y_end_train)

print("Best params:", grid_forest.best_params_)
print("Best score:", grid_forest.best_score_)
plt.figure(figsize=(5,4), dpi=100)
plt.title("Random Forest Prediction")
plt.scatter(forest_test_pred,Y_end_test)
plt.plot(Y_end_test,Y_end_test, c="r");
grid_forest=pd.DataFrame(grid_forest.cv_results_)

grid_forest[["param_criterion","param_max_depth","param_n_estimators","mean_test_score"]].sort_values(by=["mean_test_score"], ascending=False).head(6)

# According to grid search, param estimator is 20, max_depth is 7, criterion is mae. That's why, we can run again with this parameters
from xgboost import XGBRegressor


xgboost=XGBRegressor(objective='reg:linear',
                    max_depth =7,
                    n_estimators =15,
                    learning_rate=0.07,
                    seed=42,
                    )

xgboost.fit(X_end_train, Y_end_train)

xgboost_test_pred=forest_end.predict(X_end_test)
xgboost_train_pred=forest_end.predict(X_end_train)

print("MSE:", mean_squared_error(Y_end_test,xgboost_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_end_test,xgboost_test_pred)))
plt.figure(figsize=(5,4),dpi=100)
plt.title("XGBOOST Prediction")
plt.scatter(xgboost_test_pred,Y_end_test)
plt.plot(Y_end_test,Y_end_test, c="r");