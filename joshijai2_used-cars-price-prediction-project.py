import numpy as np #for linear algebra and scientific computing

import pandas as pd #data analysis and manipulation
#data visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split #split into training and testing data

#encoders for categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import category_encoders as ce
cars_Train = pd.read_excel("../input/used-cars-dataset/Data_Train.xlsx")

cars_Test = pd.read_excel("../input/used-cars-dataset/Data_Test.xlsx")
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

sns.boxplot(cars.Kilometers_Driven)
#setting the outlier as nan

cars.loc[cars.Kilometers_Driven > 1000000, "Kilometers_Driven"] = np.nan
#4   Fuel_Type

sns.countplot(cars.Fuel_Type)
#5   Transmission

sns.countplot(cars.Transmission, palette="Reds_r")
#6   Owner_Type

sns.countplot(cars.Owner_Type, order=['First','Second','Third','Fourth & Above'])
#7   Mileage

cars[cars.Mileage.isnull()]
#extract numerical data from mileage

cars.Mileage = cars.Mileage.str.split(expand=True)[0].astype("float64")
#let's set these missing values as 0

cars.Mileage.fillna(0, inplace=True)

#We should also make another column with rows corresponding to missing values set as 1 else 0. But here it wouldn't provide significant improvement to our model.
cars.info()
sns.distplot(cars.Mileage, kde=False)
#7   Engine

print("Total null values:",cars.Engine.isnull().sum())

cars[cars.Engine.isnull()].head()
cars.Engine = cars.Engine.str.split(expand=True)[0].astype("float64")
sns.distplot(cars.Engine, kde=False)
#8 Power

print("Total null values:",cars.Power.isnull().sum())

cars[cars.Power.isnull()].head()
cars.Power = cars.Power.apply(lambda s: np.nan if "null" in str(s) else s).str.split(expand=True)[0].astype("float64")
sns.distplot(cars.Power, kde=False)
#9 Seats

print("Total null values:",cars.Seats.isnull().sum())

cars[cars.Seats.isnull()].head()
cars.loc[cars.Seats<1,"Seats"] = np.nan
sns.distplot(cars.Seats, kde=False)
#sns.pairplot(cars)
sns.heatmap(cars.corr(), cmap="coolwarm")
#extracting brand and model names from name

carnames = cars.Name.str.split(expand=True)[[0,1,2,3,4]]
carnames.rename(columns={0:'Brand',1:'Model',2:'Type',3:'Subtype',4:'Subtype_2'}, inplace=True)
cars = cars.join(carnames)

cars = cars.drop("Name", axis=1)
#creating new features using combinations of categorical columns

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
cars = cars.join(interactions)
cars.sample(5)
cars.info()
features = cars.drop(["Price"], axis=1)

target = cars["Price"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)
#let's see those missing values

X_train.isnull().sum()
#We need to fill missing values in Engine, Power and Seats only. Missing values in Subtype and Subtype_2 would provide a feature.

num_cols = X_train.select_dtypes("number")

null_num_cols = num_cols.columns[num_cols.isnull().any()]



for cols in null_num_cols:

    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train.groupby('Brand')[cols].transform('mean'))

    X_train.loc[:,cols] = X_train.loc[:,cols].fillna(X_train[cols].mean())



    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test.groupby('Brand')[cols].transform('mean'))

    X_test.loc[:,cols] = X_test.loc[:,cols].fillna(X_test[cols].mean())
#let's check again

print(X_train.isnull().sum())
cars.select_dtypes("object").nunique()
#Encoding the categorical columns

# Create the encoder itself

# Fit the encoder using the categorical features and target

# Transform the features and join to dataframe
#One-hot encoding for fuel_type and transmission

OHE_cat_features = ["Fuel_Type","Transmission"]

OH_encoder = OneHotEncoder(drop='first', sparse=False)



OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[OHE_cat_features]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[OHE_cat_features]))



OH_cols_train.index = X_train.index

OH_cols_test.index = X_test.index



OH_cols_train.columns = OH_encoder.get_feature_names(OHE_cat_features)

OH_cols_test.columns = OH_encoder.get_feature_names(OHE_cat_features)



X_train_enc = X_train.join(OH_cols_train)

X_test_enc = X_test.join(OH_cols_test)
OH_encoder.get_feature_names(OHE_cat_features)
#Label encoding for Owner_type

label_cat_features = ["Owner_Type"]

le = LabelEncoder()

X_train_enc.loc[:,label_cat_features] = le.fit_transform(X_train_enc.loc[:,label_cat_features[0]])

X_test_enc.loc[:,label_cat_features] = le.transform(X_test_enc.loc[:,label_cat_features[0]])

#Target encoding for Other categorical columns

target_cat_features = ["Location", "Brand", "Model", "Type","Subtype", "Subtype_2"]

target_enc = ce.TargetEncoder(cols=target_cat_features)

target_enc.fit(X_train[target_cat_features], y_train)

X_train_enc = X_train_enc.join(target_enc.transform(X_train[target_cat_features]).add_suffix('_enc'))

X_test_enc = X_test_enc.join(target_enc.transform(X_test[target_cat_features]).add_suffix('_enc'))
object_cols = X_train_enc.select_dtypes('object')

X_train_enc.drop(object_cols, axis=1, inplace = True)

X_test_enc.drop(object_cols, axis=1, inplace = True)
print(X_train_enc.info())

print(X_test_enc.info())
from sklearn.metrics import mean_squared_error #RMSE for evaluation

from sklearn.inspection import permutation_importance #for evaluation of feature importance
from xgboost import XGBRegressor
#base model

xgbr = XGBRegressor(objective='reg:squarederror')

xgbr.fit(X_train_enc,y_train)

yhat_xgbr = xgbr.predict(X_test_enc)

xgbr.score(X_test_enc,y_test) #r2 score
#feature importance

#for i in zip(X_train_enc.columns,list(map(str,xgbr.feature_importances_))):

#    print(i)

feature_importance = xgbr.feature_importances_

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

fig = plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, np.array(X_train_enc.columns)[sorted_idx])

plt.title('Feature Importance')



result = permutation_importance(xgbr, X_test_enc, y_test, n_repeats=10, random_state=0)

sorted_idx = result.importances_mean.argsort()

plt.subplot(1, 2, 2)

plt.boxplot(result.importances[sorted_idx].T,

            vert=False, labels=np.array(X_train_enc.columns)[sorted_idx])

plt.title("Permutation Importance (test set)")

fig.tight_layout()

plt.show()
print(mean_squared_error(y_test, yhat_xgbr, squared=False))

sns.kdeplot(y_test)

sns.kdeplot(yhat_xgbr)
#now we will try to improve from this model 

xgbr.get_params
from sklearn.model_selection import GridSearchCV
param_grid = {

    "learning_rate": [0.05,0.1,0.15],

    "max_depth": [5,8,10],

    "min_child_weight": [3,5],

    "subsample": [0.5,0.6],

    "n_estimators": [100, 150, 200]

}

gscv = GridSearchCV(estimator=xgbr, param_grid=param_grid, n_jobs=-1, verbose=5)
gscv.fit(X_train_enc,y_train)
gscv.score(X_test_enc, y_test)
gscv.best_params_
#improved model after hyperparameter tuning using gridsearchcv

xgbr = XGBRegressor(objective='reg:squarederror',learning_rate= 0.05, max_depth= 8,

                    min_child_weight= 5, n_estimators= 200, subsample=0.6)

xgbr.fit(X_train_enc,y_train)

yhat_xgbr = xgbr.predict(X_test_enc)

xgbr.score(X_test_enc,y_test) #r2 score
print(mean_squared_error(y_test, yhat_xgbr, squared=False))

sns.kdeplot(y_test)

sns.kdeplot(yhat_xgbr)