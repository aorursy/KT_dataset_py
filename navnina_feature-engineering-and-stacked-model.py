%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')



import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

import math as ma

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics

from plotly import plotly

import plotly.offline as offline

import plotly.graph_objs as go

offline.init_notebook_mode()

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from collections import Counter

from sklearn.metrics import accuracy_score

from sklearn import model_selection

from scipy import stats

from scipy.stats import norm, skew



from scipy.sparse import hstack

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn import linear_model

import pylab

from matplotlib.pyplot import figure

font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 12}



plt.rc('font', **font)
data_train = pd.read_csv('../input/train.csv')
data_train.head()
data_test = pd.read_csv('../input/test.csv')

data_test.head()
data_test.shape
data_train.shape
train_ID = data_train['Id']

test_ID = data_test['Id']
data_train.drop("Id", axis = 1, inplace = True)

data_test.drop("Id", axis = 1, inplace = True)
y_tr = data_train['SalePrice']

data_train.drop(['SalePrice'], axis=1, inplace=True)

X_tr=data_train

X_test=data_test

print(X_tr.shape)

print(y_tr.shape)

print(X_test.shape)
# First we fit a normal distribution on our Y values

sns.distplot(y_tr , fit=norm, hist_kws={ "linewidth": 3,"alpha": 1, "color": "g"},kde_kws={"color": "g", "lw": 3});



# Calculate the mean and sigma

(mu, sigma) = norm.fit(y_tr)

print( '\n mu = {:.2f} and sigma = {:.2f} \n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



# Taking log(1+y_tr)

y_tr = np.log1p(((y_tr)))



#Check the new distribution 

sns.distplot(y_tr , fit=norm, hist_kws={ "linewidth": 3,"alpha": 1, "color": "g"},kde_kws={"color": "g", "lw": 3},);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(y_tr)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))





#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

X_tr.info()
#training data



# Filling MSZoing with most frequent value

X_tr['MSZoning']=X_tr['MSZoning'].fillna(X_tr.MSZoning.mode()[0])



# LotFrontage can be categorised based on the neighnourhood, so taking median of the neighbourhood

X_tr["LotFrontage"] =X_tr["LotFrontage"].fillna(X_tr.groupby(['Neighborhood'])["LotFrontage"].median()[0])



X_tr["Alley"] = X_tr["Alley"].fillna("None")



X_tr["MasVnrType"] = X_tr["MasVnrType"].fillna("None")



X_tr["MasVnrArea"] = X_tr["MasVnrArea"].fillna(0)



X_tr["BsmtQual"] = X_tr["BsmtQual"].fillna("None")

X_tr["BsmtCond"] = X_tr["BsmtCond"].fillna("None")

X_tr["BsmtExposure"] = X_tr["BsmtExposure"].fillna("None")

X_tr["BsmtFinType1"] = X_tr["BsmtFinType1"].fillna("None")

X_tr["BsmtFinType2"] = X_tr["BsmtFinType2"].fillna("None")



X_tr["FireplaceQu"] = X_tr["FireplaceQu"].fillna("None")



X_tr["GarageType"] = X_tr["GarageType"].fillna("None")

X_tr["GarageFinish"] = X_tr["GarageFinish"].fillna("None")

X_tr["GarageYrBlt"] = X_tr["GarageYrBlt"].fillna(0)

X_tr["GarageQual"] = X_tr["GarageQual"].fillna("None")

X_tr["GarageCond"] = X_tr["GarageCond"].fillna("None")



X_tr["PoolQC"] = X_tr["PoolQC"].fillna("None")



X_tr["Fence"] = X_tr["Fence"].fillna("None")

X_tr["MiscFeature"] = X_tr["MiscFeature"].fillna("None")



X_tr['Electrical']=X_tr['Electrical'].fillna(X_tr.Electrical.mode()[0])

X_tr.isnull().values.any()
X_test.info()
# As I mentioned before I am filling with most common value in The TRAINING data

X_test['MSZoning']=X_test['MSZoning'].fillna(X_tr.MSZoning.mode()[0])



# Group by neighbourhood, median of the training data

X_test["LotFrontage"] =X_test["LotFrontage"].fillna(X_tr.groupby(['Neighborhood'])["LotFrontage"].median()[0])



X_test["Alley"] = X_test["Alley"].fillna("None")



X_test["MasVnrType"] = X_test["MasVnrType"].fillna("None")



X_test["MasVnrArea"] = X_test["MasVnrArea"].fillna(0)



X_test["BsmtQual"] = X_test["BsmtQual"].fillna("None")

X_test["BsmtCond"] = X_test["BsmtCond"].fillna("None")

X_test["BsmtExposure"] = X_test["BsmtExposure"].fillna("None")

X_test["BsmtFinType1"] = X_test["BsmtFinType1"].fillna("None")

X_test["BsmtFinType2"] = X_test["BsmtFinType2"].fillna("None")

X_test["TotalBsmtSF"] = X_test["TotalBsmtSF"].fillna(0)

X_test["BsmtFinSF1"] = X_test["BsmtFinSF1"].fillna(0)

X_test["BsmtFinSF2"] = X_test["BsmtFinSF2"].fillna(0)

X_test["BsmtUnfSF"] = X_test["BsmtUnfSF"].fillna(0)

X_test["BsmtFullBath"] = X_test["BsmtFullBath"].fillna(0)

X_test["BsmtHalfBath"] = X_test["BsmtHalfBath"].fillna(0)



X_test["FireplaceQu"] = X_test["FireplaceQu"].fillna("None")



X_test["GarageYrBlt"] = X_test["GarageYrBlt"].fillna(0)

X_test["GarageArea"] = X_test["GarageArea"].fillna(0)

X_test["GarageType"] = X_test["GarageType"].fillna("None")

X_test["GarageFinish"] = X_test["GarageFinish"].fillna("None")

X_test["GarageQual"] = X_test["GarageQual"].fillna("None")

X_test["GarageCond"] = X_test["GarageCond"].fillna("None")

X_test["GarageCars"] = X_test["GarageCars"].fillna("None")



X_test["PoolQC"] = X_test["PoolQC"].fillna("None")



X_test["Fence"] = X_test["Fence"].fillna("None")

X_test["MiscFeature"] = X_test["MiscFeature"].fillna("None")



X_test['MSZoning']=X_test['MSZoning'].fillna(X_tr.MSZoning.mode()[0])



X_test['Utilities']=X_test['Utilities'].fillna(X_tr.Utilities.mode()[0])



X_test['Exterior1st']=X_test['Exterior1st'].fillna(X_tr.Exterior1st.mode()[0])

X_test['Exterior2nd']=X_test['Exterior2nd'].fillna(X_tr.Exterior2nd.mode()[0])







X_test['KitchenQual']=X_test['KitchenQual'].fillna(X_tr.KitchenQual.mode()[0])

X_test['Functional']=X_test['Functional'].fillna("Typ")



X_test['SaleType']=X_test['SaleType'].fillna(X_tr.SaleType.mode()[0])

X_test.isnull().values.any()
X_tr['HasGarage'] = X_tr['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

X_test['HasGarage'] = X_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)



X_tr['HasPool'] = X_tr['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

X_test['HasPool'] = X_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)



X_tr['Has2ndFloor'] = X_tr['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

X_test['Has2ndFloor'] = X_test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)



X_tr['HasBsmt'] = X_tr['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

X_test['HasBsmt'] = X_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)



X_tr['HasFireplace'] = X_tr['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

X_test['HasFireplace'] = X_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
# I found out that taking log(1+total_area) really improves the result as total_area is a skewed feature

X_tr["Total_area"] = np.log1p(X_tr["OpenPorchSF"]+X_tr["WoodDeckSF"]+X_tr["MasVnrArea"]

                            +X_tr["TotalBsmtSF"]+X_tr["GrLivArea"]+X_tr["1stFlrSF"]+X_tr["2ndFlrSF"]+X_tr["GarageArea"]

                            +data_train["PoolArea"])



#Check the new distribution 

sns.distplot(X_tr['Total_area'] , fit=norm, hist_kws={ "linewidth": 3,"alpha": 1, "color": "g"},kde_kws={"color": "g", "lw": 3});



# mean and std

(mu, sigma) = norm.fit(X_tr['Total_area'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
fig, ax = plt.subplots()

ax.scatter(x = X_tr['Total_area'], y=y_tr, marker='x', color='red')

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('Total_area', fontsize=13)

plt.show()
index_outlier=X_tr[(X_tr['Total_area']>9.5)].index

X_tr = X_tr.drop(index_outlier)

y_tr = y_tr.drop(index_outlier)

print(X_tr.shape)

print(y_tr.shape)
index_outlier=X_tr[y_tr<10.75].index

X_tr = X_tr.drop(index_outlier)

y_tr = y_tr.drop(index_outlier)

print(X_tr.shape)

print(y_tr.shape)
fig, ax = plt.subplots()

ax.scatter(x = X_tr['Total_area'], y=y_tr, marker='x', color='red')

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('Total_area', fontsize=13)

plt.show()
X_test["Total_area"] = np.log1p(X_test["OpenPorchSF"]+X_test["WoodDeckSF"]+X_test["MasVnrArea"]

                            +X_test["TotalBsmtSF"]+X_test["GrLivArea"]+X_test["1stFlrSF"]+X_test["2ndFlrSF"]+X_test["GarageArea"]

                            +X_test["PoolArea"])
X_tr.drop(columns=[ 'LotFrontage','OpenPorchSF','WoodDeckSF','MasVnrArea','TotalBsmtSF','GrLivArea',

                         '1stFlrSF','2ndFlrSF','GarageArea','PoolArea'                     

                        ], axis=1, inplace=True)
X_test.drop(columns=[ 'LotFrontage','OpenPorchSF','WoodDeckSF','MasVnrArea','TotalBsmtSF','GrLivArea',

                         '1stFlrSF','2ndFlrSF','GarageArea','PoolArea'                     

                        ], axis=1, inplace=True)
Quality_train=X_tr[['ExterQual','HeatingQC','KitchenQual','FireplaceQu','BsmtQual','GarageQual']]

X_tr['Quality']=(Quality_train.mode(axis=1))[0]



Quality_test=X_test[['ExterQual','HeatingQC','KitchenQual','FireplaceQu','BsmtQual','GarageQual']]

X_test['Quality']=(Quality_test.mode(axis=1))[0]
X_tr.drop(columns=[ 'ExterQual','HeatingQC','KitchenQual','FireplaceQu','BsmtQual','GarageQual'                    

                        ], axis=1, inplace=True)

X_test.drop(columns=[ 'ExterQual','HeatingQC','KitchenQual','FireplaceQu','BsmtQual','GarageQual'                    

                        ], axis=1, inplace=True)
Condition_train=X_tr[['ExterCond','GarageCond','BsmtCond']]

X_tr['Condition']=(Condition_train.mode(axis=1))[0]



Condition_test=X_test[['ExterCond','GarageCond','BsmtCond']]

X_test['Condition']=(Condition_test.mode(axis=1))[0]
X_tr.drop(columns=[ 'ExterCond','GarageCond','BsmtCond'                    

                        ], axis=1, inplace=True)

X_test.drop(columns=[ 'ExterCond','GarageCond','BsmtCond'                    

                        ], axis=1, inplace=True)
X_tr['Total_bath']=X_tr['BsmtFullBath']+X_tr['FullBath']+X_tr['FullBath']

X_test['Total_bath']=X_test['BsmtFullBath']+X_test['FullBath']+X_test['HalfBath']



X_tr.drop(columns=[ 'BsmtFullBath','HalfBath','FullBath'                   

                        ], axis=1, inplace=True)

X_test.drop(columns=[ 'BsmtFullBath','HalfBath','FullBath'                    

                        ], axis=1, inplace=True)

fig, ax = plt.subplots()

ax.scatter(x = X_tr['LotArea'], y = y_tr,marker='x', color='red')

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('Lot_area', fontsize=13)

plt.show()
X_tr['LotArea'].skew()
X_tr['LotArea']=np.log1p(X_tr['LotArea'])

X_test['LotArea']=np.log1p(X_test['LotArea'])
X_tr['LotArea'].skew()
fig, ax = plt.subplots()

ax.scatter(x = X_tr['LotArea'], y = y_tr,marker='x', color='red')

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('Lot_area', fontsize=13)

plt.show()
index_outlier=X_tr[(X_tr['LotArea']>11)].index

X_tr = X_tr.drop(index_outlier)

y_tr = y_tr.drop(index_outlier)

print(X_tr.shape)

print(y_tr.shape)
X_tr['biggerhouse_area'] = X_tr['Total_area'].apply(lambda x: 1 if x > 8.5 else 0)

X_tr['smallerhouse_area'] = X_tr['Total_area'].apply(lambda x: 1 if x < 8 else 0)
X_test['biggerhouse_area'] = X_test['Total_area'].apply(lambda x: 1 if x > 8.5 else 0)

X_test['smallerhouse_area'] = X_test['Total_area'].apply(lambda x: 1 if x < 8 else 0)
X_tr['biggerhouse_lot'] = X_tr['LotArea'].apply(lambda x: 1 if x > 10.5 else 0)

X_tr['smallerhouse_lot'] = X_tr['LotArea'].apply(lambda x: 1 if x < 8.5 else 0)
X_test['biggerhouse_lot'] = X_test['LotArea'].apply(lambda x: 1 if x > 10.5 else 0)

X_test['smallerhouse_lot'] = X_test['LotArea'].apply(lambda x: 1 if x < 8.5 else 0)
X_tr['biggerhouse'] = X_tr['biggerhouse_area'] | X_tr['biggerhouse_lot']

X_tr['smallerhouse'] = X_tr['smallerhouse_area'] | X_tr['smallerhouse_lot']



X_test['biggerhouse'] = X_test['biggerhouse_area'] | X_test['biggerhouse_lot']

X_test['smallerhouse'] = X_test['smallerhouse_area'] | X_test['smallerhouse_lot']
X_tr.drop(columns=[ 'biggerhouse_area','smallerhouse_area','biggerhouse_lot','smallerhouse_lot'], axis=1, inplace=True)



X_test.drop(columns=['biggerhouse_area','smallerhouse_area','biggerhouse_lot','smallerhouse_lot'], axis=1, inplace=True)
print(X_tr.shape)

print(X_test.shape)
X_tr.head()
import math

import numpy as np

import pandas as pd

import seaborn as sns

import scipy.stats as ss

import matplotlib.pyplot as plt

from collections import Counter



def convert(data, to):

    converted = None

    if to == 'array':

        if isinstance(data, np.ndarray):

            converted = data

        elif isinstance(data, pd.Series):

            converted = data.values

        elif isinstance(data, list):

            converted = np.array(data)

        elif isinstance(data, pd.DataFrame):

            converted = data.as_matrix()

    elif to == 'list':

        if isinstance(data, list):

            converted = data

        elif isinstance(data, pd.Series):

            converted = data.values.tolist()

        elif isinstance(data, np.ndarray):

            converted = data.tolist()

    elif to == 'dataframe':

        if isinstance(data, pd.DataFrame):

            converted = data

        elif isinstance(data, np.ndarray):

            converted = pd.DataFrame(data)

    else:

        raise ValueError("Unknown data conversion: {}".format(to))

    if converted is None:

        raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data),to))

    else:

        return converted

    





def cramers_v(x, y):

    

    confusion_matrix = pd.crosstab(x,y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))







def correlation_ratio(categories, measurements):

    

    categories = convert(categories, 'array')

    measurements = convert(measurements, 'array')

    fcat, _ = pd.factorize(categories)

    cat_num = np.max(fcat)+1

    y_avg_array = np.zeros(cat_num)

    n_array = np.zeros(cat_num)

    for i in range(0,cat_num):

        cat_measures = measurements[np.argwhere(fcat == i).flatten()]

        n_array[i] = len(cat_measures)

        y_avg_array[i] = np.average(cat_measures)

    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)

    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))

    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))

    if numerator == 0:

        eta = 0.0

    else:

        eta = np.sqrt(numerator/denominator)

    return eta





def associations(dataset,y, nominal_columns=None, mark_columns=False, theil_u=False, plot=True,

                          return_results = False, **kwargs):

    

    dataset["SalePrice"]=y

    columns = dataset.columns

    if nominal_columns is None:

        nominal_columns = list()

    elif nominal_columns == 'all':

        nominal_columns = columns

    corr = pd.DataFrame(index=columns, columns=columns)

    for i in range(0,len(columns)):

        for j in range(i,len(columns)):

            if i == j:

                corr[columns[i]][columns[j]] = 1.0

            else:

                if columns[i] in nominal_columns:

                    if columns[j] in nominal_columns:

                        if theil_u:

                            corr[columns[j]][columns[i]] = theils_u(dataset[columns[i]],dataset[columns[j]])

                            corr[columns[i]][columns[j]] = theils_u(dataset[columns[j]],dataset[columns[i]])

                        else:

                            cell = cramers_v(dataset[columns[i]],dataset[columns[j]])

                            corr[columns[i]][columns[j]] = cell

                            corr[columns[j]][columns[i]] = cell

                    else:

                        cell = correlation_ratio(dataset[columns[i]], dataset[columns[j]])

                        corr[columns[i]][columns[j]] = cell

                        corr[columns[j]][columns[i]] = cell

                else:

                    if columns[j] in nominal_columns:

                        cell = correlation_ratio(dataset[columns[j]], dataset[columns[i]])

                        corr[columns[i]][columns[j]] = cell

                        corr[columns[j]][columns[i]] = cell

                    else:

                        cell, _ = ss.pearsonr(dataset[columns[i]], dataset[columns[j]])

                        corr[columns[i]][columns[j]] = cell

                        corr[columns[j]][columns[i]] = cell

    corr.fillna(value=np.nan, inplace=True)

    if mark_columns:

        marked_columns = ['{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col) for col in columns]

        corr.columns = marked_columns

        corr.index = marked_columns

    if plot:

        plt.subplots(figsize=(15,15))

        sns.heatmap(corr, vmax=0.9, square=True)

        plt.show()

    if return_results:

        return abs(corr["SalePrice"][:]).sort_values(ascending=False)
corr=associations(X_tr,y_tr,nominal_columns=['MSZoning','Street','Alley','LotShape' ,'LandContour','Utilities','LotConfig',

                                         'LandSlope','Neighborhood' ,'Condition1','Condition2' ,'BldgType','HouseStyle' ,    

                                        'RoofStyle','RoofMatl' ,'Exterior1st', 'Exterior2nd', 'MasVnrType' ,

                                         'Foundation','BsmtExposure','BsmtFinType1',

                                         'BsmtFinType2','Heating','CentralAir','Electrical',

                                         'Functional','GarageType','GarageFinish',

                                         'PavedDrive','Fence','MiscFeature','SaleType','SaleCondition','Quality','Condition',

                                             'biggerhouse','smallerhouse','HasGarage','PoolQC','HasPool','Has2ndFloor','HasBsmt',

                                             'HasFireplace'

                                         ],plot=False,return_results = True)

corr
X_tr.drop(columns=[ 'BsmtFinType2','Heating','LandContour','Alley','Functional',

                         'LotConfig','EnclosedPorch','KitchenAbvGr','ScreenPorch','Condition2','RoofMatl','RoofMatl','MiscFeature',

                         'MSSubClass','MoSold','Street','3SsnPorch','LandSlope','OverallCond','LowQualFinSF','YrSold','MiscVal',

                         'BsmtHalfBath','Utilities','BsmtFinSF2','BedroomAbvGr','Exterior2nd','GarageType','GarageYrBlt','BsmtFinSF1',

                         'BsmtUnfSF','BsmtFinSF1','YearBuilt','SalePrice','HasBsmt','Fence','Has2ndFloor','PoolQC','HasPool'

                        ], axis=1, inplace=True)



X_test.drop(columns=[ 'BsmtFinType2','Heating','LandContour','Alley','Functional',

                         'LotConfig','EnclosedPorch','KitchenAbvGr','ScreenPorch','Condition2','RoofMatl','RoofMatl','MiscFeature',

                         'MSSubClass','MoSold','Street','3SsnPorch','LandSlope','OverallCond','LowQualFinSF','YrSold','MiscVal',

                         'BsmtHalfBath','Utilities','BsmtFinSF2','BedroomAbvGr','Exterior2nd','GarageType','GarageYrBlt','BsmtFinSF1',

                         'BsmtUnfSF','BsmtFinSF1','YearBuilt','HasBsmt','Fence','Has2ndFloor','PoolQC','HasPool'                     

                        ], axis=1, inplace=True)



print(X_tr.shape)

print(X_test.shape)
associations(X_tr,y_tr, nominal_columns=['MSZoning','Street','Alley','LotShape' ,'LandContour','Utilities','LotConfig',

                                         'LandSlope','Neighborhood' ,'Condition1','Condition2' ,'BldgType','HouseStyle' ,    

                                        'RoofStyle','RoofMatl' ,'Exterior1st', 'Exterior2nd', 'MasVnrType' ,

                                         'Foundation','BsmtExposure','BsmtFinType1',

                                         'BsmtFinType2','Heating','CentralAir','Electrical',

                                         'Functional','GarageType','GarageFinish',

                                         'PavedDrive','Fence','MiscFeature','SaleType','SaleCondition','Quality','Condition','HasGarage','PoolQC',

                                         'HasPool','Has2ndFloor','HasBsmt','HasFireplace'

                                         ],plot=False, return_results = True)
from sklearn.preprocessing import RobustScaler

Total_area_scalar = RobustScaler()

Total_area_scalar.fit(X_tr['Total_area'].values.reshape(-1,1)) # finding the mean and standard deviation of this data

#print(f"Mean : {Total_area_scalar.mean_[0]}, Standard deviation : {np.sqrt(Total_area_scalar.var_[0])}")



# Now standardize the data with above maen and variance.

Tr_Total_area_standardized = Total_area_scalar.transform(X_tr['Total_area'].values.reshape(-1, 1))

Test_Total_area_standardized = Total_area_scalar.transform(X_test['Total_area'].values.reshape(-1, 1))



print("Shape of matrix after standarsation")

print(Tr_Total_area_standardized.shape)

print(Test_Total_area_standardized.shape)



LotArea_scalar = RobustScaler()

LotArea_scalar.fit(X_tr['LotArea'].values.reshape(-1,1)) # finding the mean and standard deviation of this data

#print(f"Mean : {Total_area_scalar.mean_[0]}, Standard deviation : {np.sqrt(Total_area_scalar.var_[0])}")



# Now standardize the data with above maen and variance.

Tr_LotArea_standardized = LotArea_scalar.transform(X_tr['LotArea'].values.reshape(-1, 1))

Test_LotArea_standardized = LotArea_scalar.transform(X_test['LotArea'].values.reshape(-1, 1))



print("Shape of matrix after standarsation")

print(Tr_LotArea_standardized.shape)

print(Test_LotArea_standardized.shape)

def One_hot_encoding_tr(col):

    my_counter = Counter()

    for word in col.values:

        my_counter.update(word.split())



    col_dict = dict(my_counter)

    sorted_col_dict = dict(sorted(col_dict.items(), key=lambda kv: kv[1])) # sort categories in desc order as a dictionary



    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(vocabulary=list(sorted_col_dict.keys()), lowercase=False, binary=True)

    vectorizer.fit(col.values)

    print(vectorizer.get_feature_names())

    

    Tr_col_one_hot = vectorizer.transform(col.values)

    return Tr_col_one_hot



def One_hot_encoding_test(col,test_col):

    my_counter = Counter()

    for word in col.values:

        my_counter.update(word.split())



    col_dict = dict(my_counter)

    sorted_col_dict = dict(sorted(col_dict.items(), key=lambda kv: kv[1])) # sort categories in desc order as a dictionary



    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(vocabulary=list(sorted_col_dict.keys()), lowercase=False, binary=True)

    vectorizer.fit(col.values)

    print(vectorizer.get_feature_names())

    

    Tr_col_one_hot = vectorizer.transform(col.values)

    Test_col_one_hot = vectorizer.transform(test_col.values)

    return Test_col_one_hot

Tr_MSZoning_one_hot=One_hot_encoding_tr(X_tr['MSZoning'])

Test_MSZoning_one_hot=One_hot_encoding_test(X_tr['MSZoning'],X_test['MSZoning'])

print("Shape of matrix after one hot encoding")

print(Tr_MSZoning_one_hot.shape)

print(Test_MSZoning_one_hot.shape)
Tr_LotShape_one_hot=One_hot_encoding_tr(X_tr['LotShape'])

Test_LotShape_one_hot=One_hot_encoding_test(X_tr['LotShape'],X_test['LotShape'])

print("Shape of matrix after one hot encoding")

print(Tr_LotShape_one_hot.shape)

print(Test_LotShape_one_hot.shape)
Tr_Neighborhood_one_hot=One_hot_encoding_tr(X_tr['Neighborhood'])

Test_Neighborhood_one_hot=One_hot_encoding_test(X_tr['Neighborhood'],X_test['Neighborhood'])

print("Shape of matrix after one hot encoding")

print(Tr_Neighborhood_one_hot.shape)

print(Test_Neighborhood_one_hot.shape)
Tr_Condition1_one_hot=One_hot_encoding_tr(X_tr['Condition1'])

Test_Condition1_one_hot=One_hot_encoding_test(X_tr['Condition1'],X_test['Condition1'])

print("Shape of matrix after one hot encoding")

print(Tr_Condition1_one_hot.shape)

print(Test_Condition1_one_hot.shape)
Tr_BldgType_one_hot=One_hot_encoding_tr(X_tr['BldgType'])

Test_BldgType_one_hot=One_hot_encoding_test(X_tr['BldgType'],X_test['BldgType'])

print("Shape of matrix after one hot encoding")

print(Tr_BldgType_one_hot.shape)

print(Test_BldgType_one_hot.shape)
Tr_HouseStyle_one_hot=One_hot_encoding_tr(X_tr['HouseStyle'])

Test_HouseStyle_one_hot=One_hot_encoding_test(X_tr['HouseStyle'],X_test['HouseStyle'])

print("Shape of matrix after one hot encoding")

print(Tr_HouseStyle_one_hot.shape)

print(Test_HouseStyle_one_hot.shape)
X_tr['OverallQual'] = X_tr['OverallQual'].astype(str)

X_test['OverallQual'] = X_test['OverallQual'].astype(str)

Tr_OverallQual_one_hot=One_hot_encoding_tr(X_tr['OverallQual'])

Test_OverallQual_one_hot=One_hot_encoding_test(X_tr['OverallQual'],X_test['OverallQual'])

print("Shape of matrix after one hot encoding")

print(Tr_OverallQual_one_hot.shape)

print(Test_OverallQual_one_hot.shape)
X_tr['YearRemodAdd'] = X_tr['YearRemodAdd'].astype(str)

X_test['YearRemodAdd'] = X_test['YearRemodAdd'].astype(str)

Tr_YearRemodAdd_one_hot=One_hot_encoding_tr(X_tr['YearRemodAdd'])

Test_YearRemodAdd_one_hot=One_hot_encoding_test(X_tr['YearRemodAdd'],X_test['YearRemodAdd'])

print("Shape of matrix after one hot encoding")

print(Tr_YearRemodAdd_one_hot.shape)

print(Test_YearRemodAdd_one_hot.shape)
Tr_RoofStyle_one_hot=One_hot_encoding_tr(X_tr['RoofStyle'])

Test_RoofStyle_one_hot=One_hot_encoding_test(X_tr['RoofStyle'],X_test['RoofStyle'])

print("Shape of matrix after one hot encoding")

print(Tr_RoofStyle_one_hot.shape)

print(Test_RoofStyle_one_hot.shape)
Tr_Exterior1st_one_hot=One_hot_encoding_tr(X_tr['Exterior1st'])

Test_Exterior1st_one_hot=One_hot_encoding_test(X_tr['Exterior1st'],X_test['Exterior1st'])

print("Shape of matrix after one hot encoding")

print(Tr_Exterior1st_one_hot.shape)

print(Test_Exterior1st_one_hot.shape)
Tr_MasVnrType_one_hot=One_hot_encoding_tr(X_tr['MasVnrType'])

Test_MasVnrType_one_hot=One_hot_encoding_test(X_tr['MasVnrType'],X_test['MasVnrType'])

print("Shape of matrix after one hot encoding")

print(Tr_MasVnrType_one_hot.shape)

print(Test_MasVnrType_one_hot.shape)
Tr_Foundation_one_hot=One_hot_encoding_tr(X_tr['Foundation'])

Test_Foundation_one_hot=One_hot_encoding_test(X_tr['Foundation'],X_test['Foundation'])

print("Shape of matrix after one hot encoding")

print(Tr_Foundation_one_hot.shape)

print(Test_Foundation_one_hot.shape)
Tr_BsmtExposure_one_hot=One_hot_encoding_tr(X_tr['BsmtExposure'])

Test_BsmtExposure_one_hot=One_hot_encoding_test(X_tr['BsmtExposure'],X_test['BsmtExposure'])

print("Shape of matrix after one hot encoding")

print(Tr_BsmtExposure_one_hot.shape)

print(Test_BsmtExposure_one_hot.shape)
Tr_BsmtFinType1_one_hot=One_hot_encoding_tr(X_tr['BsmtFinType1'])

Test_BsmtFinType1_one_hot=One_hot_encoding_test(X_tr['BsmtFinType1'],X_test['BsmtFinType1'])

print("Shape of matrix after one hot encoding")

print(Tr_BsmtFinType1_one_hot.shape)

print(Test_BsmtFinType1_one_hot.shape)
Tr_CentralAir_one_hot=One_hot_encoding_tr(X_tr['CentralAir'])

Test_CentralAir_one_hot=One_hot_encoding_test(X_tr['CentralAir'],X_test['CentralAir'])

print("Shape of matrix after one hot encoding")

print(Tr_CentralAir_one_hot.shape)

print(Test_CentralAir_one_hot.shape)
Tr_Electrical_one_hot=One_hot_encoding_tr(X_tr['Electrical'])

Test_Electrical_one_hot=One_hot_encoding_test(X_tr['Electrical'],X_test['Electrical'])

print("Shape of matrix after one hot encoding")

print(Tr_Electrical_one_hot.shape)

print(Test_Electrical_one_hot.shape)
X_tr['TotRmsAbvGrd'] = X_tr['TotRmsAbvGrd'].astype(str)

X_test['TotRmsAbvGrd'] = X_test['TotRmsAbvGrd'].astype(str)

Tr_TotRmsAbvGrd_one_hot=One_hot_encoding_tr(X_tr['TotRmsAbvGrd'])

Test_TotRmsAbvGrd_one_hot=One_hot_encoding_test(X_tr['TotRmsAbvGrd'],X_test['TotRmsAbvGrd'])

print("Shape of matrix after one hot encoding")

print(Tr_TotRmsAbvGrd_one_hot.shape)

print(Test_TotRmsAbvGrd_one_hot.shape)
X_tr['Fireplaces'] = X_tr['Fireplaces'].astype(str)

X_test['Fireplaces'] = X_test['Fireplaces'].astype(str)

Tr_Fireplaces_one_hot=One_hot_encoding_tr(X_tr['Fireplaces'])

Test_Fireplaces_one_hot=One_hot_encoding_test(X_tr['Fireplaces'],X_test['Fireplaces'])

print("Shape of matrix after one hot encoding")

print(Tr_Fireplaces_one_hot.shape)

print(Test_Fireplaces_one_hot.shape)
Tr_GarageFinish_one_hot=One_hot_encoding_tr(X_tr['GarageFinish'])

Test_GarageFinish_one_hot=One_hot_encoding_test(X_tr['GarageFinish'],X_test['GarageFinish'])

print("Shape of matrix after one hot encoding")

print(Tr_GarageFinish_one_hot.shape)

print(Test_GarageFinish_one_hot.shape)
X_tr['GarageCars'] = X_tr['GarageCars'].astype(str)

X_test['GarageCars'] = X_test['GarageCars'].astype(str)

Tr_GarageCars_one_hot=One_hot_encoding_tr(X_tr['GarageCars'])

Test_GarageCars_one_hot=One_hot_encoding_test(X_tr['GarageCars'],X_test['GarageCars'])

print("Shape of matrix after one hot encoding")

print(Tr_GarageCars_one_hot.shape)

print(Test_GarageCars_one_hot.shape)
Tr_PavedDrive_one_hot=One_hot_encoding_tr(X_tr['PavedDrive'])

Test_PavedDrive_one_hot=One_hot_encoding_test(X_tr['PavedDrive'],X_test['PavedDrive'])

print("Shape of matrix after one hot encoding")

print(Tr_PavedDrive_one_hot.shape)

print(Test_PavedDrive_one_hot.shape)
Tr_SaleType_one_hot=One_hot_encoding_tr(X_tr['SaleType'])

Test_SaleType_one_hot=One_hot_encoding_test(X_tr['SaleType'],X_test['SaleType'])

print("Shape of matrix after one hot encoding")

print(Tr_SaleType_one_hot.shape)

print(Test_SaleType_one_hot.shape)
Tr_SaleCondition_one_hot=One_hot_encoding_tr(X_tr['SaleCondition'])

Test_SaleCondition_one_hot=One_hot_encoding_test(X_tr['SaleCondition'],X_test['SaleCondition'])

print("Shape of matrix after one hot encoding")

print(Tr_SaleCondition_one_hot.shape)

print(Test_SaleCondition_one_hot.shape)
Tr_Quality_one_hot=One_hot_encoding_tr(X_tr['Quality'])

Test_Quality_one_hot=One_hot_encoding_test(X_tr['Quality'],X_test['Quality'])

print("Shape of matrix after one hot encoding")

print(Tr_Quality_one_hot.shape)

print(Test_Quality_one_hot.shape)
Tr_Condition_one_hot=One_hot_encoding_tr(X_tr['Condition'])

Test_Condition_one_hot=One_hot_encoding_test(X_tr['Condition'],X_test['Condition'])

print("Shape of matrix after one hot encoding")

print(Tr_Condition_one_hot.shape)

print(Test_Condition_one_hot.shape)
X_tr['Total_bath'] = X_tr['Total_bath'].astype(str)

X_test['Total_bath'] = X_test['Total_bath'].astype(str)

Tr_Total_bath_one_hot=One_hot_encoding_tr(X_tr['Total_bath'])

Test_Total_bath_one_hot=One_hot_encoding_test(X_tr['Total_bath'],X_test['Total_bath'])

print("Shape of matrix after one hot encoding")

print(Tr_Total_bath_one_hot.shape)

print(Test_Total_bath_one_hot.shape)
X_tr['smallerhouse'] = X_tr['smallerhouse'].astype(str)

X_test['smallerhouse'] = X_test['smallerhouse'].astype(str)

Tr_smallerhouse_one_hot=One_hot_encoding_tr(X_tr['smallerhouse'])

Test_smallerhouse_one_hot=One_hot_encoding_test(X_tr['smallerhouse'],X_test['smallerhouse'])

print("Shape of matrix after one hot encoding")

print(Tr_smallerhouse_one_hot.shape)

print(Test_smallerhouse_one_hot.shape)
X_tr['biggerhouse'] = X_tr['biggerhouse'].astype(str)

X_test['biggerhouse'] = X_test['biggerhouse'].astype(str)

Tr_biggerhouse_one_hot=One_hot_encoding_tr(X_tr['biggerhouse'])

Test_biggerhouse_one_hot=One_hot_encoding_test(X_tr['biggerhouse'],X_test['biggerhouse'])

print("Shape of matrix after one hot encoding")

print(Tr_biggerhouse_one_hot.shape)

print(Test_biggerhouse_one_hot.shape)
X_tr['HasGarage'] = X_tr['HasGarage'].astype(str)

X_test['HasGarage'] = X_test['HasGarage'].astype(str)

Tr_HasGarage_one_hot=One_hot_encoding_tr(X_tr['HasGarage'])

Test_HasGarage_one_hot=One_hot_encoding_test(X_tr['HasGarage'],X_test['HasGarage'])

print("Shape of matrix after one hot encoding")

print(Tr_HasGarage_one_hot.shape)

print(Test_HasGarage_one_hot.shape)
X_tr['HasFireplace'] = X_tr['HasFireplace'].astype(str)

X_test['HasFireplace'] = X_test['HasFireplace'].astype(str)

Tr_HasFireplace_one_hot=One_hot_encoding_tr(X_tr['HasFireplace'])

Test_HasFireplace_one_hot=One_hot_encoding_test(X_tr['HasFireplace'],X_test['HasFireplace'])

print("Shape of matrix after one hot encoding")

print(Tr_HasFireplace_one_hot.shape)

print(Test_HasFireplace_one_hot.shape)
train = hstack((Tr_Total_area_standardized,

                Tr_MSZoning_one_hot,Tr_LotShape_one_hot,Tr_Neighborhood_one_hot,Tr_Condition1_one_hot,Tr_BldgType_one_hot,

                Tr_HouseStyle_one_hot,Tr_OverallQual_one_hot,Tr_YearRemodAdd_one_hot,Tr_RoofStyle_one_hot,Tr_Exterior1st_one_hot,

                Tr_MasVnrType_one_hot,Tr_Foundation_one_hot,Tr_BsmtExposure_one_hot,Tr_BsmtFinType1_one_hot,Tr_CentralAir_one_hot,

                Tr_Electrical_one_hot,Tr_TotRmsAbvGrd_one_hot,Tr_Fireplaces_one_hot,Tr_GarageFinish_one_hot,Tr_GarageCars_one_hot,

                Tr_PavedDrive_one_hot,Tr_SaleType_one_hot,Tr_SaleCondition_one_hot,Tr_Quality_one_hot,

                Tr_Condition_one_hot,Tr_Total_bath_one_hot,Tr_LotArea_standardized,Tr_biggerhouse_one_hot,Tr_smallerhouse_one_hot,

                Tr_HasGarage_one_hot,Tr_HasFireplace_one_hot

             

                )) 



test = hstack((Test_Total_area_standardized,

                Test_MSZoning_one_hot,Test_LotShape_one_hot,Test_Neighborhood_one_hot,Test_Condition1_one_hot,Test_BldgType_one_hot,

                Test_HouseStyle_one_hot,Test_OverallQual_one_hot,Test_YearRemodAdd_one_hot,Test_RoofStyle_one_hot,Test_Exterior1st_one_hot,

                Test_MasVnrType_one_hot,Test_Foundation_one_hot,Test_BsmtExposure_one_hot,Test_BsmtFinType1_one_hot,Test_CentralAir_one_hot,

                Test_Electrical_one_hot,Test_TotRmsAbvGrd_one_hot,Test_Fireplaces_one_hot,Test_GarageFinish_one_hot,Test_GarageCars_one_hot,

                Test_PavedDrive_one_hot,Test_SaleType_one_hot,Test_SaleCondition_one_hot,Test_Quality_one_hot,

                Test_Condition_one_hot,Test_Total_bath_one_hot,Test_LotArea_standardized,Test_biggerhouse_one_hot,Test_smallerhouse_one_hot,

               Test_HasGarage_one_hot,Test_HasFireplace_one_hot

             

                 )) 





print(train.shape)

print(test.shape)
from sklearn.svm import SVR

svr = SVR(kernel='precomputed')





kernel_train =  np.dot(train, train.T)  # linear kernel

kernel_train=kernel_train.todense()



svr.fit(kernel_train, y_tr)



#kernel_test = np.dot(X_test, X_train[svc.support_, :].T)

kernel_test = np.dot(test,train.T)

kernel_test=kernel_test.todense()

y_pred_svr= svr.predict(kernel_train)



y_test_pred_svr= svr.predict(kernel_test)

y_pred_svr = np.expm1(y_pred_svr)

y_tr_svr=np.expm1(y_tr)

import math

def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
rmsle(np.array(y_tr_svr),y_pred_svr)
plt.scatter(np.array(y_tr_svr), y_pred_svr)

plt.xlabel("Prices: $Y_i$")

plt.ylabel("Predicted prices: $\hat{Y}_i$")

plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

plt.show()
import xgboost as xgb



xgb = xgb.XGBRegressor(colsample_bytree=0.2,gamma=0.0, learning_rate=0.01,max_depth=4, min_child_weight=1.5,  n_estimators=7200,reg_alpha=0.9, reg_lambda=0.6, subsample=0.2, seed=42,silent=0)             



xgb.fit(train, y_tr)



y_pred_xgb = xgb.predict(train)



y_test_pred_xgb = xgb.predict(test)



y_pred_xgb = np.expm1(y_pred_xgb)

y_tr_xgb=np.expm1(y_tr)
rmsle(np.array(y_tr_xgb),y_pred_xgb)
plt.scatter(np.array(y_tr_xgb), y_pred_xgb)

plt.xlabel("Prices: $Y_i$")

plt.ylabel("Predicted prices: $\hat{Y}_i$")

plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

plt.show()
from sklearn.ensemble import RandomForestRegressor

rf_grid = RandomForestRegressor(n_estimators=1000,

                                max_depth=60, 

                                max_features=150,

                                min_samples_leaf=10,

                                random_state=1)

rf_grid.fit(train, y_tr)

y_pred_rf = rf_grid.predict(train)

y_test_pred_rf = rf_grid.predict(test)
y_pred_rf = np.expm1(y_pred_rf)

y_tr_rf=np.expm1(y_tr)
rmsle(np.array(y_tr_rf),y_pred_rf)
from sklearn.ensemble import GradientBoostingRegressor

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
GBoost.fit(train, y_tr)

y_pred_gb = GBoost.predict(train)

y_test_pred_gb = GBoost.predict(test)
y_pred_gb= np.expm1(y_pred_gb)

y_tr_gb=np.expm1(y_tr)
rmsle(np.array(y_tr_gb),y_pred_gb)
plt.scatter(np.array(y_tr_gb), y_pred_gb)

plt.xlabel("Prices: $Y_i$")

plt.ylabel("Predicted prices: $\hat{Y}_i$")

plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

plt.show()
svr_rbf = SVR(kernel='rbf',C=0.5, gamma=0.01)
svr_rbf.fit(train, y_tr)

y_pred_rbf = svr_rbf.predict(train)

y_test_pred_rbf = svr_rbf.predict(test)
y_pred_rbf= np.expm1(y_pred_rbf)

y_tr_rbf=np.expm1(y_tr)
rmsle(np.array(y_tr_rbf),y_pred_rbf)
plt.scatter(np.array(y_tr_rbf), y_pred_rbf)

plt.xlabel("Prices: $Y_i$")

plt.ylabel("Predicted prices: $\hat{Y}_i$")

plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

plt.show()
y_test_pred= pd.DataFrame()

y_test_pred['svr']=np.expm1(y_test_pred_svr)

y_test_pred['xgb']=np.expm1(y_test_pred_xgb)

y_test_pred['rf']=np.expm1(y_test_pred_rf)

y_test_pred['gb']=np.expm1(y_test_pred_gb)

y_test_pred['rbf']=np.expm1(y_test_pred_rbf)

y_test_pred['skews']=y_test_pred.iloc[:,0:5].skew(axis=1)

y_test_pred['minimums']=y_test_pred.iloc[:,0:5].min(axis=1)

y_test_pred['maxs']=y_test_pred.iloc[:,0:5].max(axis=1)

y_test_pred['medians']=y_test_pred.iloc[:,0:5].median(axis=1)



y_test_pred.head()
y_test_pred['final'] = np.where((y_test_pred.skews) > 1.7, (y_test_pred.minimums), (y_test_pred.medians))

y_test_pred['final'] = np.where((y_test_pred.skews) < -1.7, (y_test_pred.maxs), (y_test_pred.final))

y_test_pred_final=y_test_pred['final']
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = y_test_pred_final

sub.to_csv('submission.csv',index=False)