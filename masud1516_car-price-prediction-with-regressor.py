# Libaries for data read

import numpy as np

import pandas as pd

from datetime import datetime



# Libaries for visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Libaries for model

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score



# other's

import warnings
# Let's hide Warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")

df.head()
# Let's check the missing value 

df.isnull().sum() # There is no missing value feature
# Let's Final Dataset without car name feature

dataset = df.drop(["Car_Name"], axis=1)

dataset.head()
# Let's create a current year feature

current_year = datetime.now().year

dataset["Current_year"] = current_year

dataset.head()
# Let's create Number of years the car uses 

dataset["Num_Years"] = dataset["Current_year"] - dataset["Year"]

dataset.head()
# Let's drop unnecessary features like- "Year", "Current_year"

dataset.drop(["Year", "Current_year"], axis=1, inplace=True)

dataset.head()
# Let's find categorical features 

cate_features = [feature for feature in dataset.columns if dataset[feature].dtypes == "O"]

cate_features
# Let's create a function to do "One Hot Encoding" of Categorical Features 

encode = pd.get_dummies(dataset[cate_features], drop_first=True)

# concatanet with dataset

data = pd.concat([dataset, encode], axis=1)

data.head()
# Make final dataset for model

final_dataset = data.drop(cate_features, axis=1)

final_dataset.head()
final_dataset.corr()
corr = final_dataset.corr()

corr_features = corr.index

plt.figure(figsize=(15,10))

sns.heatmap(final_dataset[corr_features].corr(), annot=True)
# Lets divide X & Y dataset

X = final_dataset.iloc[:, 1:]

y = final_dataset.iloc[:, 0]
# Features Importance 

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()

model.fit(X, y)

print(model.feature_importances_)
# Let's find best 5 features by using plot

feature_importance = pd.Series(model.feature_importances_, index=X.columns)

feature_importance.nlargest(5).plot(kind='barh')

plt.show()
# Let's split train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
# create model and train the model with Random Forest Regressor

cpp_model = RandomForestRegressor()

cpp_model.fit(X_train, y_train)
# Let's test the model

y_pred = cpp_model.predict(X_test)

y_pred
# Let's compare test result with actual

pred_dataset = pd.DataFrame({"Actual_Data": y_test, "Predict_Data": y_pred})

pred_dataset.head()
# Model score

cpp_model.score(X_train, y_train)
# r2 Score of Model

R2Score = r2_score(y_test, y_pred)

R2Score
cpp_xgbr_model = XGBRegressor()

cpp_xgbr_model.fit(X_train, y_train)

y_pred_xgb = cpp_xgbr_model.predict(X_test)

y_pred_xgb
# Let's compare test result with actual

xgb_pred_dataset = pd.DataFrame({"Actual_Data": y_test, "Predict_Data": y_pred_xgb})

xgb_pred_dataset.head()
# Model score

cpp_xgbr_model.score(X_train, y_train)
# r2 Score of Model

r2_score(y_test, y_pred_xgb)
cscore = cross_val_score(cpp_xgbr_model, X_train, y_train.ravel(), cv=5)

cscore.mean()