#import necessary libraries

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet, Lasso, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#Reading in training data and getting shape

train = pd.read_csv("../input/train.csv")



train_shape = train.shape[0]



train.shape
#Reading in test data and getting shape

test = pd.read_csv("../input/test.csv")

test.shape
#checking for outliers in the data

sns.scatterplot('GrLivArea', 'SalePrice', hue='OverallQual', data=train);

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#filter dataset

train[(train['SalePrice'] < 300000) & (train['GrLivArea'] > 4000)]
#dropping outliers and rechecking the shape of the data- saving the shape to use to split the data later

train.drop(train[(train['SalePrice'] < 300000) & \

                 (train['GrLivArea'] > 4000)].index,inplace=True)

train_shape = train.shape[0]

train.shape
#Ensuring the outliers have been removed

sns.scatterplot('GrLivArea', 'SalePrice', hue='OverallQual', data=train)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#checking the distribution of the of the sale prices to see if it needs to be normalised

plt.figure(figsize=(8, 5))

sns.distplot(train['SalePrice'])

plt.title('SalePrice Distribution')

plt.ylabel('Frequency')

plt.xticks(rotation=45)
#Normalising the sale price because it was skewed, and assigning it to variable y

y = train.SalePrice

y = np.log1p(y)

len(y)
#Plotting the results to view the transformation

plt.figure(figsize=(8, 5))

sns.distplot(y)

plt.title('Normalized SalePrice Distribution')

plt.ylabel('Frequency')

plt.xticks(rotation=45)
#computing the correlations of all the features against the sale price

corr = pd.DataFrame(train.corr(method='pearson')['SalePrice'])

corr1 = train.corr().drop('Id')



# Top 10 Heatmap

k = 10 #number of variables for heatmap

cols = corr1.nlargest(k, 'SalePrice')['SalePrice'].index

to_plot = np.corrcoef(train[cols].values.T)

plt.figure(figsize=(10, 10))

sns.set(font_scale=1.25)

sns.heatmap(to_plot, cbar=True, annot=True, square=True, fmt='.2f',

            annot_kws={'size': 9}, yticklabels=cols.values, 

            xticklabels=cols.values, cmap='coolwarm')
all_columns = test.columns

train = train[all_columns]



data = pd.concat([train, test])

data.shape
#count of missing values per variable



#sns.set_style("whitegrid")

plt.figure(figsize=(10, 8))

missing = data.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(ascending=False, inplace=True)

missing.plot.bar()

#missing
#Filling null values with appropriate code as seen in data description doc

nones = [

    'PoolQC', 'MiscFeature', 'Alley','Fence', 

    'FireplaceQu', 'GarageType','GarageFinish',

    'GarageQual','GarageCond','BsmtQual','BsmtCond',

    'BsmtExposure','BsmtFinType1','BsmtFinType2'

]



for none in nones:

    data[none].fillna('NA',inplace=True)
#This feature used 'None' and 'TA' instead of NA

data.MasVnrType.fillna('None', inplace=True)

data.KitchenQual.fillna('TA', inplace=True)
#Filling nulls in numerical data with zero

naughts = [

    'GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1',

    'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath',

    'BsmtHalfBath','MasVnrArea' 

]



for naught in naughts:

    data[naught].fillna(0, inplace=True)
#we've filled nulls in some of of the catagorical data using the mode value

modes = [

    'MSZoning','Exterior1st','Exterior2nd',

    'SaleType','Electrical','Functional'

]



for m in modes:

    data[m].fillna(data[m].mode()[0], inplace=True)
sns.distplot(data['LotFrontage'],kde=True,bins=70,color='b')
data['LotFrontage'] = data.groupby(['LotArea','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
sns.distplot(data['LotFrontage'],kde=True,bins=70,color='b')
#Check the data again to ensure there are no nulls left

data.isnull().sum().sum()
data['Utilities'].value_counts()
data.drop('Utilities',axis=1, inplace=True)
#classifying all catagorica data 

boxs = [

    'MSSubClass', 'MoSold', 'GarageYrBlt', 'LotConfig', 'Neighborhood', 

    'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 

    'MasVnrArea', 'Foundation', 'SaleCondition', 'SaleType', 'Exterior2nd'

]

for box in boxs:

    data[box] = data[box].astype('category')
#Simplifying some of the ordinal features

#    data['OverallQual_binned'] = data.OverallQual.replace({1:1, 2:1, 3:1, # bad

#                                                            4:2, 5:2, 6:2, # ok

#                                                            7:3, 8:3, 9:4, 10:4 # good, great

#                                                           })

#    data['OverallCond_binned'] = data.OverallCond.replace({1:1, 2:1, 3:1, 

#                                                            4:2, 5:2, 6:2, 

#                                                            7:3, 8:3, 9:4, 10:4

#                                                           })
#data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

#data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)



#data['TotBathrooms'] = data.FullBath + (data.HalfBath*0.5) + data.BsmtFullBath + (data.BsmtHalfBath*0.5)



#basements = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF']

#data['TotalBsmt'] = data[basements].sum(axis=1)
#Nominal data

nominals = [

    'MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood','Condition1',

    'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl', 'Exterior1st','Exterior2nd',

    'MasVnrType','Foundation','Heating','CentralAir','GarageType','MiscFeature','SaleType',

    'SaleCondition','MoSold','YrSold'

]
#Ordinal data

ordinals = [

    'LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond',

    'BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','Electrical',

    'KitchenQual','Functional','FireplaceQu','GarageFinish','GarageQual',

    'GarageCond','PavedDrive','PoolQC','Fence'

]
#encoding ordinal data with a regular label encoder

from sklearn.preprocessing import LabelEncoder

for ordinal in ordinals:

    lab = LabelEncoder()

    lab.fit(data[ordinal])

    data[ordinal] = lab.transform(data[ordinal])
#Splitting catagorical and numerical features

categorical_features = data.select_dtypes(include = ["object", "category"]).columns

numerical_features = data.select_dtypes(exclude = ["object", "category"]).columns

print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

data_num = data[numerical_features]

data_cat = data[categorical_features]
data_num.isnull().values.sum(), data_cat.isnull().values.sum()
#dummy encoding catagorical data

data_cat = pd.get_dummies(data_cat)

print(data_cat.shape)
#Normalising skewed features for accuracy

from scipy.stats import skew



skewness = data_num.apply(skew)

skewness = skewness[abs(skewness) > 0.5]

print("There are {} skewed numerical features to log transform".format(skewness.shape[0]))

skewed_features = skewness.index

data_num[skewed_features] = data_num[skewed_features].applymap(np.log1p)
#skewness = skewness[abs(skewness) > 0.75]



#from scipy.special import boxcox1p

#skewed_features = skewness.index

#lam = 0.15

#for feat in skewed_features:

    #all_data[feat] += 1

#    data[feat] = boxcox1p(data[feat], lam)
#joining the numerical and catagorical data, abd scaling features. Splitting the test and train sets

data = pd.concat([data_num, data_cat], axis=1)



scaler = StandardScaler()

data[numerical_features] = scaler.fit_transform(data[numerical_features])



X_train = data[:train_shape]

X_test = data[train_shape:]

print(X_train.shape, X_test.shape)
from sklearn.model_selection import GridSearchCV,learning_curve, cross_val_score, KFold
#Declaring kfold variable

kfold = KFold(n_splits=20, random_state=42, shuffle=True)
#Function to measure RMSE

#n_folds = 20



#def rmsle_cv(model):

#    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)

#    rmse= np.sqrt(-cross_val_score(model, X_train.values, y, scoring="neg_mean_squared_error", cv = kf))

#    return(rmse)
#Lasso model 

lasso = LassoCV(alphas=[

    0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.003, 0.007, 

    0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.2, 0.4, 0.6, 

    0.8, 1, 1.2

                ], random_state=1, n_jobs=-1, verbose=1)



lasso.fit(X_train, y)

alpha = lasso.alpha_

alphas = lasso.alphas_

mse = lasso.mse_path_

print("Optimized Alpha:", alpha)



plt.plot(mse)



lasso = LassoCV(alphas=alpha * np.linspace(0.5,1.5,20),

                cv = kfold, random_state = 1, n_jobs = -1)

lasso.fit(X_train, y)

alpha = lasso.alpha_

coeffs = lasso.coef_

intercpt = lasso.intercept_

print("Final Alpha:", alpha)

print("Intercept:", intercpt )
#print("Lasso mean score:", rmsle_cv(lasso).mean())

#print("Lasso std:", rmsle_cv(lasso).std())
lasso_4 = np.expm1(lasso.predict(X_test))

lasso_preds = pd.DataFrame(dict(SalePrice=lasso_4, Id=test.Id))
#elastic net model

elnet = ElasticNetCV(alphas = [

    0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.003, 0.007,

    0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.2, 

    0.4, 0.6, 0.8, 1, 1.2

                        ] 

                ,l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1]

                ,cv = kfold, random_state = 1, n_jobs = -1)



elnet.fit(X_train, y)

alpha = elnet.alpha_

alphas = elnet.alphas_

ratio = elnet.l1_ratio_

print("Optimized Alpha:", alpha)

print("Optimized l1_ratio:", ratio)



elnet = ElasticNetCV(alphas = alpha * np.linspace(0.5,1.5,20),

                     l1_ratio = ratio * np.linspace(0.9,1.3,6), 

                     cv = kfold, random_state = 1, n_jobs = -1)

elnet.fit(X_train, y)

coeffs2 = elnet.coef_

coeffs2 = elnet.intercept_

alpha = elnet.alpha_

ratio = elnet.l1_ratio_



print("Final Alpha:", alpha)

print("Final l1_ratio:", ratio)
#print("ElasticNet mean score:", rmsle_cv(elnet).mean())

#print("ElasticNet std:", rmsle_cv(elnet).std())
ElNet_2 = np.expm1(elnet.predict(X_test))

#ElNet_preds = pd.DataFrame(dict(SalePrice=ElNet_2, Id=test.Id))
#simple stacked model- Averaging the models together for a more accurate prediction

stacked_sub = (lasso_4 + ElNet_2)/2
stacked_preds = pd.DataFrame(dict(SalePrice=stacked_sub, Id=test.Id))

stacked_preds.to_csv('stacked_submission.csv', index=False)

#ElNet_preds.to_csv('ElNet_1.csv')

#lasso_preds.to_csv('Lasso_4.csv',index=False)
model = ['Linear Regression','Decision Tree', 'Lasso', 'Ridge', 'Elastic-Net', 'Lasso + E-Net']

score = ['0.46485', '0.21068', '0.12131', '0.12301', '0.12216', '0.12042']



df = pd.DataFrame(list(zip(model, score)), 

               columns =['Model', 'Score']) 

df 