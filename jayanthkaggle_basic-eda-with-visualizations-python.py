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
#First things first

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#Creating dataframes of both train and test data from csv files

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



#Checking for the full capture of data V1.0

train_df.head()
#Checking for the full capture of data V1.1

train_df.tail()



#As seen below, there are missing values (NaN) which are to be treated to use the data for analysis
#To view potential predictor variables as a whole

train_df.columns



#Check for the dimension of the data sets

train_df.shape, test_df.shape
#In the given data, 'Id' is the primary key respective to which the house prices are predicted

ID_dup = train_df["Id"]

train_df[ID_dup.isin(ID_dup[ID_dup.duplicated()])]



#As seen below, the resulting dataframe has zero records, which means that the train data has no

#duplicate Ids
#Data exploration from Data Visualization

#Check for the basic summary statistics of the variables

train_df.describe()
#Since 'SalePrice' is of the highlight here, we specifically see the summary of it

print("Basic summary statistics of 'SalePrice'")

train_df['SalePrice'].describe()
#Check for the diversity (distribution) of 'SalePrice'

import seaborn as sns



sns.distplot(train_df['SalePrice'], rug=True, hist=True, kde=True, color='g')
#Drawing a best-fit line in linear probability for our probable scale

from scipy import stats

import matplotlib.pyplot as plt



stats.probplot(train_df['SalePrice'], plot=plt, fit=True, rvalue=False)

plt.show()
#Checking for the correlated variables is equally important to see how strongly/weakly the predicor

#variables are related to 'SalePrice'

#Correlation test is done for both numeric and categorical variables

#In this case, lets consider 'LotArea', 'BsmtFinSF1', 'MasVnrArea', 'GrLivArea', 'GarageArea', 'WoodDeckSF'



plt.figure(1)

fig, arr = plt.subplots(3, 2, figsize=(10, 10))

SP = train_df.SalePrice.values

arr[0, 0].scatter(train_df.LotArea.values, SP, color='green')

arr[0, 0].set_title('LotArea')

arr[0, 1].scatter(train_df.BsmtFinSF1.values, SP, color='green')

arr[0, 1].set_title('BsmtFinSF1')

arr[1, 0].scatter(train_df.MasVnrArea.values, SP, color='green')

arr[1, 0].set_title('MasVnrArea')

arr[1, 1].scatter(train_df.GrLivArea.values, SP, color='green')

arr[1, 1].set_title('GrLivArea')

arr[2, 0].scatter(train_df.GarageArea.values, SP, color='green')

arr[2, 0].set_title('GarageArea')

arr[2, 1].scatter(train_df.WoodDeckSF.values, SP, color='green')

arr[2, 1].set_title('WoodDeckSF')



fig.text(-0.02, 0.5, 'Sale Price', va='center', rotation='vertical', fontsize = 10)

plt.tight_layout()

plt.show()
#Now lets consider correlation of 'SalePrice' with categorical variables

#Consider 'YearBuilt', 'LandContour', 'BldgType', 'RoofStyle', 'Foundation', 'Heating'



fig, axis = plt.subplots(6, 1, figsize = (20, 50))

sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = train_df, ax = axis[0])

sns.boxplot(x = 'LandContour', y = 'SalePrice', data = train_df, ax = axis[1])

sns.boxplot(x = 'BldgType', y = 'SalePrice', data = train_df, ax = axis[2])

sns.boxplot(x = 'RoofStyle', y = 'SalePrice', data = train_df, ax = axis[3])

sns.boxplot(x = 'Foundation', y = 'SalePrice', data = train_df, ax = axis[4])

sns.boxplot(x = 'Heating', y = 'SalePrice', data = train_df, ax = axis[5])

plt.tight_layout()
#Now lets create a correlation matrix using heatmap



correlation_mat = train_df.corr()

fig, ax = plt.subplots(figsize=(16,12))

sns.heatmap(correlation_mat, vmin=0, vmax=1, square=True, center=0, cmap="YlGnBu")
#The next vital part of exploratory data analysis is checking for missing values

#Missing data should be handled before the data sets are used for further processing because of

#the following reasons:

#1. Missing data implies the reduction of the potential sample data size.

#2. With the missing data, comes the question of credibility for the inference drawn from the sample



#Therefore, both the numerical and categorical data should be treated accordingly

#Numeric data treatment: If there is a significant correlation between a numerical predictor variable

#and 'SalePrice' then the missing values in that variable could be replaced by median or mean, depending

#on the distribution of data

#Categorical data treatment: If there is a significant correlation between a categorical predictor

#variable and 'SalePrice' the nthe missing values could be replaced by the mode of the categorical values



#Check for missing values



train_df.apply(lambda x: sum(x.isnull().values), axis=1)

train_df.apply(lambda x: sum(x.isnull().values), axis=0)
#The below show the examples of treatment of data

#Treatment of numeric data'MasVnrArea':

MasVnrAreaMean = train_df.describe().MasVnrArea['mean']

train_df[(train_df['MasVnrArea'] == None)] = MasVnrAreaMean



#Confirm the missing data treatment is reflected

train_df['MasVnrArea'].isnull()
#Project2



#Features Engineering



#Data Preprocessing



#1. The skewed NUMERIC features are transformed by taking log(feature+1) to make the features more normal.

#To do this numpy function log1p is used

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])



#Check the new distribution after the log transformation

sns.distplot(train_df['SalePrice'] , fit=norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train_df['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot=plt)

plt.show()
#Log transform the skewed numeric features in both the train and test data



ntrain = train_df.shape[0]

ntest = test_df.shape[0]

y_train = train_df.SalePrice.values

all_data = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'],

                      test_df.loc[:,'MSSubClass':'SaleCondition']))



print(all_data)
from scipy.stats import skew



numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train_df[numeric_feats].apply(lambda x: skew(x.dropna()))

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])



print(all_data.shape)
#Imputing missing values



#PoolQC : data description says NA means "No Pool". That makes sense, given the huge ratio of missing value

#(+99%) and majority of houses have no Pool at all in general. 



all_data["PoolQC"] = all_data["PoolQC"].fillna("None")



#MiscFeature : data description says NA means "no misc feature"



all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")



#Alley : data description says NA means "no alley access"



all_data["Alley"] = all_data["Alley"].fillna("None")



#Fence : data description says NA means "no fence"



all_data["Fence"] = all_data["Fence"].fillna("None")



#FireplaceQu : data description says NA means "no fireplace"



all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")



#LotFrontage : Since the area of each street connected to the house property most likely have a similar 

#area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of 

#the neighborhood.



#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



#GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None



for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')

    

#GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in 

#such garage.)



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)

    

#BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely

#zero for having no basement



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

    

#BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical 

#basement-related features, NaN means that there is no basement.



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')

    

#MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 

#for the area and None for the type. 



all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)



#MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill 

#in missing values with 'RL'



all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])



#Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . 

#Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. 

#We can then safely remove it.



all_data = all_data.drop(['Utilities'], axis=1)



#Functional : data description says NA means typical



all_data["Functional"] = all_data["Functional"].fillna("Typ")



#Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing 

#value.



all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])



#KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for 

#the missing value in KitchenQual.



all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])



#Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just 

#substitute in the most common string



all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])



#SaleType : Fill in again with most frequent which is "WD"



all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])



#MSSubClass : Na most likely means No building class. We can replace missing values with None



all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")



#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()



print(all_data, all_data_na)
#More feature engineering



#There are a few numerical variables that are supposed to be transformed into categorical by nature



all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)



print(all_data)

#Label Encoding some categorical variables that may contain information in their ordering set



from sklearn.preprocessing import LabelEncoder



cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



# process columns, apply LabelEncoder to categorical features



for i in cols:

    label = LabelEncoder() 

    label.fit(list(all_data[i].values)) 

    all_data[i] = label.transform(list(all_data[i].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))
#Adding one more important feature



#Since area related features are very important to determine house prices, one more feature which 

#is the total area of basement, first and second floor areas of each house is added



# Adding total sqfootage feature

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#Getting dummy categorical features

all_data = pd.get_dummies(all_data)



print(all_data.shape)
train_X = all_data[:ntrain]

test_X = all_data[ntrain:]

y = train_df.SalePrice



print(train_X.shape, test_X.shape)
#Models



#Now the regularized linear regression models from the scikit learn module are used. Here, both Lasso

#and Rigde regularization techniques are used.



#The following code defines a function that returns the cross validation rmse error that helps in 

#evaluating the models and picking the best tuning parameters



from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, train_X, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)



model_ridge = Ridge()
#The main tuning parameter for Ridge model is alpha which is a regularization parameter that measures 

#how flexible our model is. As the regularization increases, the tendency of the model to overfit 

#decreases. However, the model might become too flexible that it fails to capture complex data



#Now lets try ridge regularization for different values of alpha



alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]



cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
#From the above curve, we can safely say that, as the value of alpha increases the rmse value increases

#which in turn means that when alpha is too large the regularization is too strong and the model cannot

#capture all of the data complexities. If alpha is too small (model is too flexible), the model starts to

#overfit. Therefore, an alpha value of 5 is the best choice as shown above.



cv_ridge.min()
#From the ridge regularization, we got rmsle value of 0.1269



#Now lets try Lasso model. Here we use the built-in Lasso CV to figure out the best alpha. The alphas in

#Lasso CV are really the inverse of the alphas in Ridge



model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train_X, y)



rmse_cv(model_lasso).mean()
#From the above results we can clearly see that Lasso has a better performance than Ridge in 

#regularization. Thus, we use Lasso for prediction on test data set.



#The coefficients obtained from the Lasso regularization



coef = pd.Series(model_lasso.coef_, index = train_X.columns)



print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated " +  

      str(sum(coef == 0)) + " variables")
#As there are a lot of collinear features in the data set, we cannot conclude that all the features that

#Lasso selected are necessarily correct. Lasso can be run a few times on bootstrapped samples to check 

#the stability of the feature selection



#Looking at the most important coefficients of significant variables



import matplotlib



%matplotlib inline



imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])



matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
#Looking at the distribution of residuals: This shows us the performance of the model



matplotlib.rcParams['figure.figsize'] = (7.0, 7.0)



pred = pd.DataFrame({"pred":model_lasso.predict(train_X), "true":y})

pred["residuals"] = pred["true"] - pred["pred"]

pred.plot(x = "pred", y = "residuals",kind = "scatter")
#Prediction on the test test



#Lets add an XGBoost model to this linear model to see if the model scores can be improved



import xgboost as xgb



train_d = xgb.DMatrix(train_X, label = y)

test_d = xgb.DMatrix(test_X)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, train_d,  num_boost_round=500, early_stopping_rounds=100)



model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
#The parameters were tuned using xgb.cv



model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)

model_xgb.fit(train_X, y)
xgb_preds = np.expm1(model_xgb.predict(test_X))

lasso_preds = np.expm1(model_lasso.predict(test_X))
print(xgb_preds, lasso_preds)
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
#We could take the weighted average of the uncorrelated results to improve the model score. However, in

#this case it does not help much



preds = 0.6*lasso_preds + 0.4*xgb_preds

print(preds)
pred_data = pd.DataFrame({"id":test_df.Id, "SalePrice":preds})

print(pred_data)

pred_data.to_csv("XGB_Lasso.csv", index = False)
#Finding the r2 value from the lasso



from sklearn.metrics import r2_score



r2_train_lasso = r2_score(y, model_lasso.predict(train_X))

print(r2_train_lasso)
#Model for comparison: Random Forest algorithm



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

import math

from scipy.stats import skew

import collections



df = pd.read_csv('../input/train.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')

tdf = pd.read_csv('../input/test.csv', sep='\s*,\s*', encoding="utf-8-sig", engine='python')



salePrice = df['SalePrice']



df.fillna(df.mean(), inplace=True)

TotalBsmtSFMean = df['TotalBsmtSF'].mean()

df.loc[df['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = np.round(TotalBsmtSFMean).astype(int)



tdf.fillna(tdf.mean(), inplace=True)

TTotalBsmtSFMean = tdf['TotalBsmtSF'].mean()

tdf.loc[tdf['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = np.round(TotalBsmtSFMean).astype(int)



y = df['SalePrice']

df = df.drop('SalePrice', axis=1)

X = df

tX = tdf



train_num = len(X)

dataset = pd.concat(objs=[X, tX], axis=0)



#log transform skewed numeric features:

numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index



skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



dataset[skewed_feats] = np.log1p(dataset[skewed_feats])



dataset_preprocessed = pd.get_dummies(dataset)

train_preprocessed = dataset_preprocessed[:train_num]

test_preprocessed = dataset_preprocessed[train_num:]



X_train, X_test, y_train, y_test = train_test_split(train_preprocessed, y, test_size=0.3, random_state=0)



rmse_est = {}

for est in range(360,550,20):

    model = RandomForestRegressor(n_estimators=est, n_jobs=-1)

    model.fit(X_train, y_train)

    predictions = np.array(model.predict(X_test))

    rmse = math.sqrt(np.mean((np.array(y_test) - predictions)**2))

    imp = sorted(zip(X.columns, model.feature_importances_), key=lambda tup: tup[1], reverse=True)

    print ("RMSE: {0} - est: {1}".format(str(rmse), est))

    rmse_est[rmse]= est



d = collections.OrderedDict(sorted(rmse_est.items()))

print ('generating file')

model = RandomForestRegressor(n_estimators=list(d.items())[0][1], n_jobs=-1)

model.fit(train_preprocessed, y)

y_test_pred = model.predict(test_preprocessed)

submission = pd.DataFrame({"Id": test_preprocessed["Id"],"SalePrice": y_test_pred})

submission.to_csv("RandomForest.csv", index = False)
print(rmse, submission)
X = df

train_num = len(X)

train_preprocessed = dataset_preprocessed[:train_num]



r2_train_RDF = r2_score(salePrice, model.predict(train_preprocessed))

print(r2_train_RDF)
from sklearn.metrics import mean_squared_error

from math import sqrt



rms = sqrt(mean_squared_error(salePrice, model.predict(train_preprocessed)))

print(rms)
#Model for comaprison: Linear Regression



import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score



# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(train_X, y)



# Make predictions using the testing set

y_pred = regr.predict(test_X)

ypred = abs(y_pred)



reg_pred_data = pd.DataFrame({"id":test_df.Id, "SalePrice":ypred})



reg_pred_data.to_csv("Linear_reg.csv", index = False)



# The coefficients

print('Coefficients: \n', regr.coef_)



# Explained variance score: 1 is perfect prediction

print('R-square: %.3f' % r2_score(y, regr.predict(train_X)))