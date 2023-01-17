import numpy as np #for linear algebra and scientific computing

import pandas as pd #data analysis and manipulation



# Input data files are available in the read-only "../input/" directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#data visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split #split into training and testing data

from sklearn.metrics import mean_squared_error #RMSE for evaluation

from sklearn.model_selection import GridSearchCV #for exhaustive grid search(hyperparameter tuning)



#encoders for categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import category_encoders as ce
cars_Train = pd.read_csv("/kaggle/input/used-cars-price-prediction/train-data.csv", index_col=0)

cars_Test = pd.read_csv("/kaggle/input/used-cars-price-prediction/test-data.csv", index_col=0)
cars = cars_Train.copy()
cars.head()
cars.info()
cars.describe()
#1   Location

plt.xticks(rotation = 90)

sns.countplot(cars.Location)
#2   Year

plt.xticks(rotation = 90)

sns.countplot(cars.Year)
#3   Kilometers_Driven

sns.distplot(cars[cars.Kilometers_Driven<500000].Kilometers_Driven, kde=False)
#setting the outliers as nan

cars.loc[cars.Kilometers_Driven > 400000, "Kilometers_Driven"] = np.nan
#4   Fuel_Type

sns.countplot(cars.Fuel_Type)
#5   Transmission

sns.countplot(cars.Transmission, palette="Reds_r")
#6   Owner_Type

sns.countplot(cars.Owner_Type, order=['First','Second','Third','Fourth & Above'])
#7   Mileage

print("Null values:", cars.Mileage.isnull().sum())

print("Outliers:",(cars.Mileage==0).sum())
#Removing units and extracting numerical data from mileage

cars.Mileage = cars.Mileage.str.split(expand=True)[0].astype("float64")
#set the outliers as null

cars[cars.Mileage==0].Mileage = np.nan
sns.distplot(cars.Mileage, kde=False)
#7   Engine

print("Total null values:",cars.Engine.isnull().sum())

cars[cars.Engine.isnull()].head()
#Removing units and extracting numerical data from Engine

cars.Engine = cars.Engine.str.split(expand=True)[0].astype("float64")
sns.distplot(cars.Engine, kde=False)
#8 Power

print("Total null values:",cars.Power.isnull().sum())

cars[cars.Power.isnull()].head()
#Removing units and extracting numerical data from Power

cars.Power = cars.Power.apply(lambda s: np.nan if "null" in str(s) else s).str.split(expand=True)[0].astype("float64")
sns.distplot(cars.Power, kde=False)
#9 Seats

print("Total null values:",cars.Seats.isnull().sum())

cars[cars.Seats.isnull()].head()
cars.loc[cars.Seats<1,"Seats"] = np.nan
sns.distplot(cars.Seats, kde=False)
#10 New_Price

print("Total null values:",cars.New_Price.isnull().sum())

cars[cars.New_Price.isnull()].head()
cars.New_Price = cars.New_Price.apply(lambda s: float(s.split()[0])*100 if "Cr" in str(s) else str(s).split()[0]).astype("float64")
print("Total null values:",cars.New_Price.isnull().sum())

sns.distplot(cars.New_Price, kde=False)
#sns.pairplot(cars)
sns.heatmap(cars.corr(), cmap="coolwarm")
carnames = cars.Name.str.split(expand=True)[[0,1,2]]
carnames.rename(columns={0:'Brand',1:'Model',2:'Type'}, inplace=True)
cars = cars.join(carnames)

cars = cars.drop("Name", axis=1)
from itertools import combinations



object_cols = cars.select_dtypes("object").columns

low_cardinality_cols = [col for col in object_cols if cars[col].nunique() < 15]

low_cardinality_cols.append("Brand")

interactions = pd.DataFrame(index=cars.index)



# Iterate through each pair of features, combine them into interaction features

for features in combinations(low_cardinality_cols,2):

    

    new_interaction = cars[features[0]].map(str)+"_"+cars[features[1]].map(str)

    

    encoder = LabelEncoder()

    interactions["_".join(features)] = encoder.fit_transform(new_interaction)
cars = cars.join(interactions) #append to the dataset
cars.head(5)
# cars.info()
features = cars.drop(["Price"], axis=1)

target = cars["Price"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)
X_train.isnull().sum()
num_cols = X_train.drop('New_Price',1).select_dtypes("number")

null_num_cols = num_cols.columns[num_cols.isnull().any()]



for cols in null_num_cols:

    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train.groupby('Brand')[cols].transform('mean'))

    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train[cols].mean())



    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test.groupby('Brand')[cols].transform('mean'))

    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test[cols].mean())
# #Binning #didn't provide improvement to results

# #Year

# X_train=X_train.drop('Year',1).join(pd.cut(X_train.Year, range(1996,2021,4), False, range(6)).astype('int64'))

# X_test=X_test.drop('Year',1).join(pd.cut(X_test.Year, range(1996,2021,4), False, range(6)).astype('int64'))



# #Kilometers_Driven

# X_train=X_train.drop('Kilometers_Driven',1).join(pd.cut(X_train.Kilometers_Driven, range(0,300001,10000), labels= range(30)).astype('int64'))

# X_test=X_test.drop('Kilometers_Driven',1).join(pd.cut(X_test.Kilometers_Driven, range(0,300001,10000), labels= range(30)).astype('int64'))
cars.select_dtypes("object").nunique()
OHE_cat_features = ["Fuel_Type","Transmission", "Location", "Owner_Type", "Brand"]

OH_encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')



OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[OHE_cat_features]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[OHE_cat_features]))



OH_cols_train.index = X_train.index

OH_cols_test.index = X_test.index



OH_cols_train.columns = OH_encoder.get_feature_names(OHE_cat_features)

OH_cols_test.columns = OH_encoder.get_feature_names(OHE_cat_features)



X_train_enc = X_train.join(OH_cols_train)

X_test_enc = X_test.join(OH_cols_test)
X_train_enc.drop(OHE_cat_features, axis=1, inplace = True)

X_test_enc.drop(OHE_cat_features, axis=1, inplace = True)
target_cat_features = X_train_enc.select_dtypes('object').columns

target_enc = ce.TargetEncoder(cols=target_cat_features)

target_enc.fit(X_train[target_cat_features], y_train)

X_train_enc = X_train_enc.join(target_enc.transform(X_train[target_cat_features]).add_suffix('_enc'))

X_test_enc = X_test_enc.join(target_enc.transform(X_test[target_cat_features]).add_suffix('_enc'))
object_cols = X_train_enc.select_dtypes('object')

X_train_enc.drop(object_cols, axis=1, inplace = True)

X_test_enc.drop(object_cols, axis=1, inplace = True)
# X_train_enc=X_train_enc.astype('int64')

# X_test_enc=X_test_enc.astype('int64')
X_train_enc.info()
pcorr = X_train_enc.join(y_train).corr()

imp_corr_cols = pcorr[['Price']][pcorr['Price']>-0.25].iloc[:-1].index



X_train_enc = X_train_enc[imp_corr_cols]

X_test_enc = X_test_enc[imp_corr_cols]
from xgboost import XGBRegressor
base_xgbr = XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist')

base_xgbr.fit(X_train_enc,y_train)



base_xgbr.score(X_test_enc,y_test) #r2 score
yhat_xgbr = base_xgbr.predict(X_test_enc)

print(mean_squared_error(y_test, yhat_xgbr, squared=False))

sns.kdeplot(y_test)

sns.kdeplot(yhat_xgbr)
feat_imp = pd.DataFrame(base_xgbr.feature_importances_, index=X_train_enc.columns)

sns.heatmap(feat_imp, cmap='Purples')
impfeat = pd.Series(base_xgbr.feature_importances_, index=X_train_enc.columns)

impcols = impfeat[impfeat>0.005].index

X_train_enc = X_train_enc[impcols]

X_test_enc = X_test_enc[impcols]
sns.heatmap(X_train_enc.join(y_train).corr()[['Price']], cmap='Reds')
#testing xgbr model

# param_grid = {

#     "learning_rate": [0.05],

#     "max_depth": [6,8,10,12],

#     "min_child_weight": [5],

#     "n_estimators": [350,400,450,500],

#     "subsample": [0.55]

# }

# gscv = GridSearchCV(estimator=base_xgbr, param_grid=param_grid, n_jobs=-1, verbose=5, cv=4)
# gscv.fit(X_train_enc, y_train)
#the best params from the given parameter grid

# gscv.best_params_

# gscv.score(X_test_enc,y_test) #r2 score
# tuned_xgbr = XGBRegressor(objective = 'reg:squarederror',

#                     learning_rate = 0.05, max_depth = 12, min_child_weight = 5,

#                     n_estimators = 500, subsample = 0.55)

# tuned_xgbr.fit(X_train_enc,y_train)



# tuned_xgbr.score(X_test_enc,y_test) #r2 score
# yhat_xgbr = tuned_xgbr.predict(X_test_enc)

# print(mean_squared_error(y_test, yhat_xgbr, squared=False))

# sns.kdeplot(y_test)

# sns.kdeplot(yhat_xgbr)
from lightgbm import LGBMRegressor
base_lgbmr = LGBMRegressor()
base_lgbmr.fit(X_train_enc, y_train)

base_lgbmr.score(X_test_enc,y_test)
yhat_lgbmr = base_lgbmr.predict(X_test_enc)

print(mean_squared_error(y_test, yhat_lgbmr, squared=False))

sns.kdeplot(y_test)

sns.kdeplot(yhat_lgbmr)
#feature importance

#pd.Series(base_lgbmr.feature_importances_, index=X_train_enc.columns)
base_lgbmr.get_params()
#initial grid search

param_grid = {

    "learning_rate": [0.15],

    "max_depth": [5,8,10,12],

    "min_child_weight": [3,5,6,8],

    "n_estimators": [300,500,800,1000,1200],

    "num_leaves": [20,25,40,50],

    "subsample": [0.3,0.5]

}

# gscv_lgbm = GridSearchCV(estimator=base_lgbmr, param_grid=param_grid, n_jobs=-1, verbose=5, cv=4)
# gscv_lgbm.fit(X_train_enc, y_train)
# gscv_lgbm.best_params_
# gscv_lgbm.score(X_test_enc,y_test) #r2 score
param_grid2 = {

    "learning_rate": [0.15],

    "max_depth": [8],

    "n_estimators": [1500,1800],

    "num_leaves": [25,27],

    'reg_alpha': [0,0.001,0.01],

    'reg_lambda': [0,0.001,0.01]

}

gscv_lgbm2 = GridSearchCV(estimator=base_lgbmr, param_grid=param_grid2, n_jobs=-1, verbose=5, cv=4)
gscv_lgbm2.fit(X_train_enc, y_train)
print(gscv_lgbm2.best_params_)

print(gscv_lgbm2.score(X_test_enc,y_test)) #r2 score
tuned_lgbmr = LGBMRegressor(**gscv_lgbm2.best_params_)

tuned_lgbmr.fit(X_train_enc, y_train)

tuned_lgbmr.score(X_test_enc,y_test)
yhat_lgbmr = tuned_lgbmr.predict(X_test_enc)

print(mean_squared_error(y_test, yhat_lgbmr, squared=False))

sns.kdeplot(y_test)

sns.kdeplot(yhat_lgbmr)
# Custom Label Encoder for handling unknown values

class LabelEncoderExt(object):

    def __init__(self):

        """

        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]

        Unknown will be added in fit and transform will take care of new item. It gives unknown class id

        """

        self.label_encoder = LabelEncoder()

        # self.classes_ = self.label_encoder.classes_



    def fit(self, data_list):

        """

        This will fit the encoder for all the unique values and introduce unknown value

        :param data_list: A list of string

        :return: self

        """

        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])

        self.classes_ = self.label_encoder.classes_



        return self



    def transform(self, data_list):

        """

        This will transform the data_list to id list where the new values get assigned to Unknown class

        :param data_list:

        :return:

        """

        new_data_list = list(data_list)

        for unique_item in np.unique(data_list):

            if unique_item not in self.label_encoder.classes_:

                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]



        return self.label_encoder.transform(new_data_list)

X_train = cars_Train.drop('Price',1)

y_train = cars.Price

X_test = cars_Test
carnames = X_train.Name.str.split(expand=True)[[0,1,2]]

carnames_test = X_test.Name.str.split(expand=True)[[0,1,2]]



carnames.rename(columns={0:'Brand',1:'Model',2:'type'}, inplace=True)

carnames_test.rename(columns={0:'Brand',1:'Model',2:'type'}, inplace=True)



X_train = X_train.join(carnames)

X_train = X_train.drop("Name", axis=1)

X_test = X_test.join(carnames_test)

X_test = X_test.drop("Name", axis=1)
object_cols = X_train.select_dtypes("object").columns

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 15]

low_cardinality_cols.append("Brand")

interactions = pd.DataFrame(index=X_train.index)

interactions_test = pd.DataFrame(index=X_test.index)



for features in combinations(low_cardinality_cols,2):

    

    new_interaction = X_train[features[0]].map(str)+"_"+X_train[features[1]].map(str)

    new_interaction_test = X_test[features[0]].map(str)+"_"+X_test[features[1]].map(str)

    

    encoder = LabelEncoderExt()

    encoder.fit(new_interaction)

    interactions["_".join(features)] = encoder.transform(new_interaction)

    interactions_test["_".join(features)] = encoder.transform(new_interaction_test)
X_train = X_train.join(interactions)

X_test = X_test.join(interactions_test)
num_cols = X_train.drop('New_Price',1).select_dtypes("number")

null_num_cols = num_cols.columns[num_cols.isnull().any()]



for cols in null_num_cols:

    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train.groupby('Brand')[cols].transform('mean'))

    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train[cols].mean())



    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test.groupby('Brand')[cols].transform('mean'))

    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test[cols].mean())
num_cols = X_train.select_dtypes("number")

null_num_cols = num_cols.columns[num_cols.isnull().any()]



for cols in null_num_cols:

    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train.groupby('Brand')[cols].transform('mean'))

    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train[cols].mean())



    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test.groupby('Brand')[cols].transform('mean'))

    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test[cols].mean())
OHE_cat_features = ["Fuel_Type","Transmission", "Location", "Owner_Type", "Brand"]

OH_encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')



OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[OHE_cat_features]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[OHE_cat_features]))



OH_cols_train.index = X_train.index

OH_cols_test.index = X_test.index



OH_cols_train.columns = OH_encoder.get_feature_names(OHE_cat_features)

OH_cols_test.columns = OH_encoder.get_feature_names(OHE_cat_features)



X_train_enc = X_train.join(OH_cols_train)

X_test_enc = X_test.join(OH_cols_test)
X_train_enc.drop(OHE_cat_features, axis=1, inplace = True)

X_test_enc.drop(OHE_cat_features, axis=1, inplace = True)
target_cat_features = X_train_enc.select_dtypes('object').columns

target_enc = ce.TargetEncoder(cols=target_cat_features)

target_enc.fit(X_train[target_cat_features], y_train)

X_train_enc = X_train_enc.join(target_enc.transform(X_train[target_cat_features]).add_suffix('_enc'))

X_test_enc = X_test_enc.join(target_enc.transform(X_test[target_cat_features]).add_suffix('_enc'))
object_cols = X_train_enc.select_dtypes('object')

X_train_enc.drop(object_cols, axis=1, inplace = True)

X_test_enc.drop(object_cols, axis=1, inplace = True)
pcorr = X_train_enc.join(y_train).corr()

imp_corr_cols = pcorr[['Price']][pcorr['Price']>-0.25].iloc[:-1].index



X_train_enc = X_train_enc[imp_corr_cols]

X_test_enc = X_test_enc[imp_corr_cols]
xgbr = XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist')

xgbr.fit(X_train_enc,y_train)
impfeat = pd.Series(xgbr.feature_importances_, index=X_train_enc.columns)

impcols = impfeat[impfeat>0.005].index

X_train_enc = X_train_enc[impcols]

X_test_enc = X_test_enc[impcols]
lgbmr = LGBMRegressor(**gscv_lgbm2.best_params_)



lgbmr.fit(X_train_enc, y_train)
preds_test = lgbmr.predict(X_test_enc)
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)