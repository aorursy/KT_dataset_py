import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV



import os



# hide warnings

import warnings

warnings.filterwarnings('ignore')
house = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

house.head()
#Print Data Shape

print ("Train data shape:", house.shape)

print ("Test data shape:", test.shape)
#check some stats of SalePrice

house.SalePrice.describe()
# summary of the dataset:

print(house.info())
# Inspect the different columsn in the dataset



house.columns
house.describe()
# Now drop the  'Id' colum since it's unnecessary for  the prediction process.

#house.drop(['Id'], axis=1, inplace=True)

#test.drop(['Id'], axis=1, inplace=True)
#Seperate Numerical and Categorical Feature



num_fea = house.dtypes[house.dtypes!='object'].index

print('Numerical features:', len(num_fea))





cat_fea = house.dtypes[house.dtypes=='object'].index

print('Categorical features:', len(cat_fea))



num_fea_test = test.dtypes[test.dtypes!='object'].index

print('Numerical features:', len(num_fea_test))





cat_fea_test = test.dtypes[test.dtypes=='object'].index

print('Categorical features:', len(cat_fea_test))

#Understand the Variance of Sale Price in log scale 



house['SalePrice_log'] = np.log(house['SalePrice'])



sns.distplot(house['SalePrice_log'])

plt.title('Distribution of Sales Price with log')
#check some relationship of Variable  with Sale Price



plt.scatter(house.GrLivArea, house.SalePrice)
# Deleting outliers Manually

house = house[house.GrLivArea < 4500]

house.reset_index(drop=True, inplace=True)
#check the scatter plot again 



plt.scatter(house.GrLivArea, house.SalePrice)
#check the shape of house again after removing outlier data

house.shape
#again check for Outlier with IQR



plt.boxplot(house['GrLivArea'])

Q1 = house['GrLivArea'].quantile(0.25)

Q3 = house['GrLivArea'].quantile(0.95)

Q3

IQR = Q3 - Q1

IQR

house = house[(house['GrLivArea'] >= Q1 - 1.5*IQR) & (house['GrLivArea'] <= Q3 + 1.5*IQR)]
#check the scatter plot again 



plt.scatter(house.TotalBsmtSF, house.SalePrice)
#again check for Outlier with IQR



plt.boxplot(house['TotalBsmtSF'])

Q1 = house['TotalBsmtSF'].quantile(0.25)

Q3 = house['TotalBsmtSF'].quantile(0.95)

Q3

IQR = Q3 - Q1

IQR

house = house[(house['TotalBsmtSF'] >= Q1 - 1.5*IQR) & (house['TotalBsmtSF'] <= Q3 + 1.5*IQR)]
#Scatter plot of OverallQual with Sale Price



plt.scatter(house.OverallQual, house.SalePrice)
#Scatter plot of Garage Area vs Sale Price



plt.scatter(house.GarageArea, house.SalePrice)
#again check for Outlier with IQR



plt.boxplot(house['GarageArea'])

Q1 = house['GarageArea'].quantile(0.25)

Q3 = house['GarageArea'].quantile(0.95)

Q3

IQR = Q3 - Q1

IQR

house = house[(house['GarageArea'] >= Q1 - 1.5*IQR) & (house['GarageArea'] <= Q3 + 1.5*IQR)]
#Scatter plots of GarageCars with Sale Price



plt.scatter(house.GarageCars, house.SalePrice)
#To check the most correlated feature



cor_mat= house[:].corr()

cor=cor_mat.sort_values(['SalePrice'],ascending=False)

print("The most correlated/important features (numeric) for the target are :")

cor.SalePrice
#Overall Quality vs Sale Price (Univariate Analysis)

variable = 'OverallQual'

data = pd.concat([house['SalePrice'], house[variable]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=variable, y="SalePrice", data=data)

# Total Rooms abobe Ground vs Sale Price

sns.boxplot(x=house['TotRmsAbvGrd'], y=house['SalePrice'])
# Year Built vs Sale Price

sns.boxplot(x=house['YearBuilt'], y=house['SalePrice'])
#We have derived some more Feature from Raw Attribute 



#Total SF area 



house['TotalSF']=house['TotalBsmtSF'] + house['1stFlrSF'] + house['2ndFlrSF']



#Total Bathrooms



house['Total_Bathrooms'] = (house['FullBath'] + (0.5 * house['HalfBath']) +

                               house['BsmtFullBath'] + (0.5 * house['BsmtHalfBath']))

#Total POrch SF



house['Total_porch_sf'] = (house['OpenPorchSF'] + house['3SsnPorch'] +

                              house['EnclosedPorch'] + house['ScreenPorch'] +

                              house['WoodDeckSF'])



#We have derived some more Feature from Raw Attribute 



#Total SF area 



test['TotalSF']=test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']



#Total Bathrooms



test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) +

                               test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))

#Total POrch SF



test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] +

                              test['EnclosedPorch'] + test['ScreenPorch'] +

                              test['WoodDeckSF'])
#Age of House

house["Age_of_house"] = house['YrSold'] - house['YearBuilt'] 

test["Age_of_house"] = house['YrSold'] - house['YearBuilt'] 
#Check for Missing Value Percentage 



def missing_data():

    total = house.isnull().sum().sort_values(ascending=False)

    percent = (house.isnull().sum()/house.isnull().count()).sort_values(ascending=False)

    missing_data= pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return  missing_data

missing_data = missing_data()
missing_data.head(20)
#Impute Missing Value based on Data Understanding 



 # PoolQC : data description says NA means "No Pool"



house['PoolQC'].replace(np.nan,'No Pool', inplace=True)



 # MiscFeature : data description says NA means "None"



house['MiscFeature'].replace(np.nan,'None', inplace=True)



  # Alley : data description says NA means "no alley access"

house['Alley'].replace(np.nan,'No alley access', inplace=True)



#Fence: data description says NA means "No Fence"



house['Fence'].replace(np.nan,'No Fence', inplace=True)



#FireplaceQu: data description says NA means "No Fireplace"



house['FireplaceQu'].replace(np.nan,'No Fireplace', inplace=True)
#Impute Missing Value based on Data Understanding 



 # PoolQC : data description says NA means "No Pool"



test['PoolQC'].replace(np.nan,'No Pool', inplace=True)



 # MiscFeature : data description says NA means "None"



test['MiscFeature'].replace(np.nan,'None', inplace=True)



  # Alley : data description says NA means "no alley access"

test['Alley'].replace(np.nan,'No alley access', inplace=True)



#Fence: data description says NA means "No Fence"



test['Fence'].replace(np.nan,'No Fence', inplace=True)



#FireplaceQu: data description says NA means "No Fireplace"



test['FireplaceQu'].replace(np.nan,'No Fireplace', inplace=True)


def missing_data():

    total = house.isnull().sum().sort_values(ascending=False)

    percent = (house.isnull().sum()/house.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return  missing_data

missing_data = missing_data()



missing_data.head(20)
house.shape


#LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , lets fill missing values by the median LotFrontage of the neighborhood

house["LotFrontage"] = house.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.mean()))


#LotFrontage : Since the area of each street connected to the test property most likely have a similar area to other tests in its neighborhood , lets fill missing values by the median LotFrontage of the neighborhood

test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.mean()))
#check if house has garage 



house["has_garage"] = house["GarageCond"].apply(lambda x: 0 if pd.isnull(x) else 1)



#GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None



for col in ("GarageType", "GarageFinish", "GarageQual","GarageCond"):

    

    house[col] = house[col].fillna('None')



    

    #fill "GarageYrBlt", "GarageCars","GarageArea with 0

    

for col in ("GarageYrBlt", "GarageCars","GarageArea"):

    house[col] = house[col].fillna(0)

#check if test has garage 



test["has_garage"] = test["GarageCond"].apply(lambda x: 0 if pd.isnull(x) else 1)



#GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None



for col in ("GarageType", "GarageFinish", "GarageQual","GarageCond"):

    

    test[col] = test[col].fillna('None')



    

    #fill "GarageYrBlt", "GarageCars","GarageArea with 0

    

for col in ("GarageYrBlt", "GarageCars","GarageArea"):

    test[col] = test[col].fillna(0)

#Fill BSMT variable with None



for col in ("BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType2", "BsmtFinType1"):

    house[col] = house[col].fillna('None')

    

    #Fill BSMT variable with None



for col in ("BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType2", "BsmtFinType1"):

    test[col] = test[col].fillna('None')

#check again for missing data



def missing_data():

    total = house.isnull().sum().sort_values(ascending=False)

    percent = (house.isnull().sum()/house.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return  missing_data

missing_data = missing_data()



missing_data.head(20)
#Fill MasVnrType : NA most likely means no veneer and ifthere is no Vnr that area will be 0 . Lets Impute accordingly



house["MasVnrType"] = house["MasVnrType"].fillna("None")

house["MasVnrArea"] = house["MasVnrArea"].fillna(0)
#Fill MasVnrType : NA most likely means no veneer and ifthere is no Vnr that area will be 0 . Lets Impute accordingly



test["MasVnrType"] = test["MasVnrType"].fillna("None")

test["MasVnrArea"] = test["MasVnrArea"].fillna(0)
#Impute missing value of Electrical Variable with mode 



house['Electrical'] = house['Electrical'].fillna(house['Electrical'].mode()[0])



test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
#set(final.columns).symmetric_difference(test_final.columns)
house['Heating'].value_counts()
house['Electrical'].replace('Mix','SBrkr', inplace=True)

house['Condition2'].replace(['RRNn','RRAn','RRAe'],'Norm', inplace=True)

house['Exterior1st'].replace(['ImStucc','Stone'],'VinylSd', inplace=True)

house=house[house.Exterior2nd!='Other']



house['GarageQual'].replace('Fa','TA',inplace=True)

house=house[house.RoofMatl!='Metal']

house=house[house.RoofMatl!='Roll']

house=house[house.RoofMatl!='Membran']

house=house[house.Heating!='OthW']



#Lets check for Percentage of missing value again 



def missing_data():

    total = house.isnull().sum().sort_values(ascending=False)

    percent = (house.isnull().sum()/house.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return  missing_data

missing_data = missing_data()

missing_data.head(20)
#Lets check for Percentage of missing value again 



def missing_data():

    total = test.isnull().sum().sort_values(ascending=False)

    percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return  missing_data

missing_data = missing_data()

missing_data.head(20)
#Now check for some categorical variable and the relative Variance of information



house.Utilities.value_counts()
#Not much variance here so we will drop Utililities Variable 



house.drop(['Utilities'], axis=1, inplace=True)

test.drop(['Utilities'], axis=1, inplace=True)
#Now check for relative Variance of PoolQC



house.PoolQC.value_counts()



#Not much variance here so we will drop PoolQC Variable 



house.drop(['PoolQC'], axis=1, inplace=True)



test.drop(['PoolQC'], axis=1, inplace=True)



#Now check for relative Variance of MiscFeature



house.MiscFeature.value_counts()



#Not much variance here so we will drop PoolQC Variable 



house.drop(['MiscFeature'], axis=1, inplace=True)

test.drop(['MiscFeature'], axis=1, inplace=True)
house.drop(['Heating'], axis=1, inplace=True)

test.drop(['Heating'], axis=1, inplace=True)
# Some numerical features are actually really categories

house = house.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},

                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",

                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}

                      })
# Some numerical features are actually really categories

test = test.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},

                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",

                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}

                      })
#check for shape 



house.shape
test.shape
#Lets check for correlaton one more 



def correlation():  

    corr = house.corr()

    plt.figure(figsize=(15,10))

    plt.title('Overall Corellation')

    sns.heatmap(corr, annot=False, linewidths=0.5, cmap = 'coolwarm')

correlation()
# all numeric (float and int) variables in the dataset

house_numeric = house.select_dtypes(include=['float64', 'int64'])

house_numeric.head()
num_fea = house.dtypes[(house.dtypes=='float64') | (house.dtypes=='int64')].index

print('Numerical features:', len(num_fea))

cat_fea = house.dtypes[house.dtypes=='object'].index

print('Categorical features:', len(cat_fea))



print('Numerical features:', num_fea)
num_fea_test = test.dtypes[(test.dtypes=='float64') | (test.dtypes=='int64')].index

print('Numerical features:', len(num_fea_test))

cat_fea_test = test.dtypes[test.dtypes=='object'].index

print('Categorical features:', len(cat_fea_test))





print('Numerical features:', num_fea_test)
# subset all categorical variables

house_categorical = house.select_dtypes(include=['object'])

house_categorical.head()

# subset all categorical variables

test_categorical = test.select_dtypes(include=['object'])

test_categorical.head()

# Categorical boolean mask

categorical_feature_mask = house.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = house.columns[categorical_feature_mask].tolist()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

house[categorical_cols] = house[categorical_cols].apply(lambda col: labelencoder.fit_transform(col.astype(str)))
# Categorical boolean mask

categorical_feature_mask = test.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = test.columns[categorical_feature_mask].tolist()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

test[categorical_cols] = test[categorical_cols].apply(lambda col: labelencoder.fit_transform(col.astype(str)))
test.head()
# convert into dummies

#house_dummies = pd.get_dummies(house_categorical, drop_first=True)

#house_dummies.head()
# convert into dummies

#test_dummies = pd.get_dummies(test_categorical, drop_first=True)

#test_dummies.head()
# drop categorical variables 

#house_final=house.drop(list(house_categorical.columns), axis=1)
# drop categorical variables 

#test_final=test.drop(list(test_categorical.columns), axis=1)
# concat dummy variables with X

#house_final = pd.concat([house_final, house_dummies], axis=1)
# concat dummy variables with X

#test_final = pd.concat([test_final,test_dummies], axis=1)
house.shape
test.shape
# Separating Response Variable 



y = house['SalePrice']

#Separating independent variable 



final=house.drop(['SalePrice','SalePrice_log'], axis=1)
final.shape
# Train Test split



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(final,y, test_size = 0.07, random_state = 10)



print("The shape of Target set is :"+str(X_train.shape))

print("The shape of feature set is :"+str(y_train.shape))

print("The shape of Target set is :"+str(X_test.shape))

print("The shape of feature set is :"+str(y_test.shape))
# Standardize numerical features

from sklearn.preprocessing import StandardScaler

numerical_features = final.select_dtypes(exclude = ["object"]).columns

stdSc = StandardScaler()

X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])

X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])
test.head()
#Check Model Validation Metric for different values of alpha



from sklearn.linear_model import Ridge

from sklearn import metrics



print("Evaluating for different alpha values")

for alpha_val in [1,10,20,50,80,100,200,300,400,500,600,700]:

    ridger = Ridge(alpha=alpha_val).fit(X_train,y_train)

    r2train = ridger.score(X_train,y_train)

    r2test  = ridger.score(X_test,y_test)

    print("Alpha : {}\ R2 score training set : {:.2f} R2 score test set : {:.2f} Non null Coeff : {}\n".format(alpha_val,r2train,r2test,np.sum(ridger.coef_!=0)))
#check for Lasso Regression 



from sklearn.linear_model import Lasso



print("Trying out different alpha values : ")

for lasso_alpha_val in [1,10,100,200,400,500,800,1000,2000]:

    lassor = Lasso(alpha=lasso_alpha_val).fit(X_train,y_train)

    r2sctrain = lassor.score(X_train,y_train)

    r2sctest = lassor.score(X_test,y_test)

    print("Alpha : {}\ R2 score training set : {:.2f} R2 score test set : {:.2f} Non null Coeff : {}\n".format(lasso_alpha_val,r2sctrain,r2sctest,np.sum(lassor.coef_!=0)))
#Model 2 Ridge regression with 5 folds CV



# list of alphas to tune

params = {'alpha': [1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100,200,250,300,400,500,700, 1000,2000]}





ridge = Ridge()



# cross validation

folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train, y_train) 



#To check R2 Meric 



model2_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'r2', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model2_cv.fit(X_train, y_train) 
#Print cv results for neg_mean_absolute_error metric 



cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=3000]

cv_results.head()



#Print cv results for r2 metric



cv2_results = pd.DataFrame(model2_cv.cv_results_)

cv2_results = cv2_results[cv2_results['param_alpha']<=3000]
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
alpha = 100

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)
ridge.coef_
#Mapping coefficients with Feature 



coef = pd.Series(ridge.coef_, index = X_train.columns)

print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the" +  str(sum(coef == 0)) + " variables")

coef.sort_values(ascending=False)
# plotting r2 with alpha

plt.figure(figsize=(16,6))



plt.plot(cv2_results["param_alpha"], cv2_results["mean_test_score"])

plt.plot(cv2_results["param_alpha"], cv2_results["mean_train_score"])

plt.xlabel('number of features')

plt.ylabel('r-squared')

plt.title("Optimal Number of Features")

plt.legend(['test score', 'train score'], loc='upper left')


#Lets Perform Lasso Regression



lasso = Lasso()



params = {'alpha': [1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100,200,300,400,500,700, 1000,2000]}



# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train, y_train) 



lasso = Lasso()



# cross validation with R2

model1_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'r2', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model1_cv.fit(X_train, y_train) 
#Print cv results for neg_mean_absolute_error metric



cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results.head()



#Print cv results for r2 metric



cv1_results = pd.DataFrame(model1_cv.cv_results_)

cv1_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
# plotting r2  score with alpha

plt.figure(figsize=(16,6))



plt.plot(cv1_results["param_alpha"], cv1_results["mean_test_score"])

plt.plot(cv1_results["param_alpha"], cv1_results["mean_train_score"])

plt.xlabel('number of features')

plt.ylabel('r-squared')

plt.title("Optimal Number of Features")

plt.legend(['test score', 'train score'])
alpha =300



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 
lasso.coef_
#Mapping lasso coefficients with Feature column



coef = pd.Series(lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the  " +  str(sum(coef == 0)) + " variables")

coef.sort_values(ascending=False)
from sklearn.metrics import mean_squared_error

pred=lasso.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,pred)))
#Best Number of coefficients



params_dict={'alpha':[1,10,100,150,200,300,400,500,700,1000,2000]}

lasso_CV=GridSearchCV(estimator=Lasso(),param_grid=params_dict,scoring='neg_mean_squared_error',cv=5)

lasso_CV.fit(X_train,y_train)

pred=lasso_CV.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,pred)))



lasso_CV.best_params_
#Best Number of coefficients



params_dict={'alpha':[1,10,100,150,200,250,300,400,500,700,1000,2000]}

lasso_CV=GridSearchCV(estimator=Lasso(),param_grid=params_dict,scoring='neg_mean_squared_error',cv=5)

lasso_CV.fit(X_train,y_train)

pred=lasso_CV.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,pred)))



lasso_CV.best_params_
#Model evalution



from sklearn.metrics import mean_squared_error

pred=ridge.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,pred)))
#Best Number of coefficients model 1



params_dict={'alpha':[1,10,50,100,120,150,200,300,400,500,700,1000,2000]}

ridge_CV=GridSearchCV(estimator=Ridge(),param_grid=params_dict,scoring='neg_mean_squared_error',cv=5)

ridge_CV.fit(X_train,y_train)

pred=ridge_CV.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,pred)))



ridge_CV.best_params_
#Best Number of coefficients model 2



params_dict={'alpha':[1,10,50,80,100,120,150,200,300,400,500,700,1000,2000]}

ridge_CV=GridSearchCV(estimator=Ridge(),param_grid=params_dict,scoring='neg_mean_squared_error',cv=5)

ridge_CV.fit(X_train,y_train)

pred=ridge_CV.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,pred)))



ridge_CV.best_params_

from sklearn import ensemble

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error, r2_score
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)
clf_pred=clf.predict(X_test)

clf_pred= clf_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))

print('MSE:', metrics.mean_squared_error(y_test, clf_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,clf_pred, c= 'brown')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(y_test,label ='Test')

plt.plot(clf_pred, label = 'predict')

plt.show()
from sklearn.tree import DecisionTreeRegressor

dtreg = DecisionTreeRegressor(random_state = 100)

dtreg.fit(X_train, y_train)
dtr_pred = dtreg.predict(X_test)

dtr_pred= dtr_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, dtr_pred))

print('MSE:', metrics.mean_squared_error(y_test, dtr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,dtr_pred,c='green')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
from sklearn.svm import SVR

svr = SVR(kernel = 'rbf')

svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)

svr_pred= svr_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))

print('MSE:', metrics.mean_squared_error(y_test, svr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,svr_pred, c='red')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(y_test,label ='Test')

plt.plot(svr_pred, label = 'predict')

plt.show()
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 500, random_state = 0)

rfr.fit(X_train, y_train)
rfr_pred= rfr.predict(X_test)

rfr_pred = rfr_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, rfr_pred))

print('MSE:', metrics.mean_squared_error(y_test, rfr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,rfr_pred, c='orange')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(y_test,label ='Test')

plt.plot(rfr_pred, label = 'predict')

plt.show()
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [4,8,10],

    'min_samples_leaf': range(100, 400, 200),

    'min_samples_split': range(200, 500, 200),

    'n_estimators': [100,200,300], 

    'max_features': [5,10]

}

# Create a based model

rf = RandomForestRegressor()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1,verbose = 1)
# Fit the grid search to the data

grid_search.fit(X_train, y_train)
# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
# model with the best hyperparameters

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestRegressor(bootstrap=True,

                             max_depth=10,

                             min_samples_leaf=100, 

                             min_samples_split=100,

                             max_features=10,

                             n_estimators=100)
# fit

rfc.fit(X_train,y_train)
# predict

predictions = rfc.predict(X_test)

predictions = predictions.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=10,

                              learning_rate=0.01, n_estimators=5000,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



model_lgb.fit(X_train,y_train)
lgb_pred = model_lgb.predict(X_test)

lgb_pred = lgb_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, lgb_pred))

print('MSE:', metrics.mean_squared_error(y_test, lgb_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lgb_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,lgb_pred, c='orange')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(y_test,label ='Test')

plt.plot(lgb_pred, label = 'predict')

plt.show()
test.head()
#Lets check for Percentage of missing value again 



def missing_data():

    total = test.isnull().sum().sort_values(ascending=False)

    percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return  missing_data

missing_data = missing_data()

missing_data.head(20)
for col in ("Age_of_house", "BsmtHalfBath", "Total_Bathrooms","BsmtFullBath","BsmtFinSF1","TotalBsmtSF","BsmtFinSF2","BsmtUnfSF","TotalSF"):

    

    test[col] = test[col].fillna(0)
set(final.columns).symmetric_difference(test.columns)
set(X_train.columns).symmetric_difference(test.columns)
test1=test.copy(deep=True)

test2=test.copy(deep=False)
test1.head()
numerical_features_test = test.select_dtypes(exclude = ["object"]).columns

numerical_features_test

test.loc[:, numerical_features_test] = stdSc.transform(test.loc[:, numerical_features_test])
pred_test=model_lgb.predict(test)

cat_features_test = test.select_dtypes(include = ["object"]).columns

cat_features_test
submission = pd.DataFrame({'Id': test1.Id, 'SalePrice':pred_test })



submission.to_csv('submission.csv', index=False)

submission.head()