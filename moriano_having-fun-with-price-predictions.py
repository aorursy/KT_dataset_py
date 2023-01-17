# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
raw = pd.read_csv("../input/autos.csv", encoding='cp1252')
raw.head()
raw.dtypes
date_columns = ["dateCreated", "lastSeen"]
# A date looks like => "2016-04-07 03:16:57"
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
raw = pd.read_csv("../input/autos.csv", parse_dates=date_columns, date_parser=dateparse, encoding='cp1252')
raw.dtypes
raw.head()
raw.describe()
clean = raw.copy()
clean.drop(["nrOfPictures", "postalCode", "monthOfRegistration"], axis=1, inplace=True)
clean.plot(y="kilometer", kind="hist", figsize=(10, 7), bins=10, title="Km for cars...")
# Lets also check how many DIFFERENT values we have there...
set(clean["kilometer"])
clean["powerPS"].describe()
# It is obvious, we cannot have cars with ZERO power, so lets get rid of them
print("Size before removing cars with 0 power", len(clean))
clean = clean[clean["powerPS"] > 0]
print("Size after removing cars with 0 power", len(clean))
clean["powerPS"].describe()
clean = clean[clean["powerPS"] >= 60]
print("Size after removing cars with less than 60hp", len(clean))
clean = clean[clean["powerPS"] <= 1000]
print("Size after removing cars with more than 1000hp", len(clean))

clean.plot(y="powerPS", kind='hist', bins=25, figsize=(15, 7), title='HP for cars')
DEFAULT_POWER_PS = clean["powerPS"].mean()
clean["yearOfRegistration"].describe()
max(clean["dateCreated"])
clean = clean[clean["yearOfRegistration"] <= 2016]
print("Rows after removing cars whose registration was AFTER 2016 => ", len(clean))
clean["yearOfRegistration"].sort_values(ascending=True)
clean[clean["yearOfRegistration"] == 1910]
clean = clean[clean["yearOfRegistration"] > 1910]
clean["yearOfRegistration"].describe()
clean[clean["yearOfRegistration"] == 1930]
clean = clean[clean["yearOfRegistration"] > 1930]
clean["yearOfRegistration"].describe()
clean[clean["yearOfRegistration"] < 1946]
clean.plot(y="yearOfRegistration", kind='hist', bins=35, figsize=(10, 7), title="Cars and their registration years")
clean["yearOfRegistration"].describe()
DEFAULT_YEAR_OF_REGISTRATION = clean["yearOfRegistration"].mean()
clean["price"].describe()
print("Rows before discarding cars cheaper than 380 euros", len(clean))
clean = clean[clean["price"] >= 380]
print("Rows after discarding cars cheaper than 380 euros", len(clean))
clean["price"].sort_values(ascending=False)
clean = clean[clean["price"] <= 100000]
print("Rows after discarding cars valued at more than 100000 euros", len(clean))
clean["price"].describe()
clean["name"].describe()
set(clean["name"])
# We cannot get much from this... we will just drop it
del clean["name"]
clean["seller"].describe()
set(clean["seller"])
print("Rows with private seller => ", len(clean[clean["seller"] == "privat"]))
print("Rows with dealer seller => ", len(clean[clean["seller"] != "privat"]))
del clean["seller"]
del clean["abtest"]
clean["vehicleType"].describe()
set(clean["vehicleType"])
clean["vehicleType"].value_counts()
# Lets plot it... just for fun
clean["vehicleType"].value_counts().plot(kind='bar', title="cars by type")
DEFAULT_VEHICLE_TYPE = "limousine"
print("Rows without a vehicle type", clean["vehicleType"].isna().sum())
print("Total number of rows", len(clean))
default_vehicle_type = "limousine"
print("Rows before droping cars without type", len(clean))
clean = clean.dropna(subset=["vehicleType"])
print("Rows after droping cars without type", len(clean))
set(clean["gearbox"])
clean["gearbox"].describe()
clean["gearbox"].value_counts()
print("Rows without a gearbox type", clean["gearbox"].isna().sum())

print("Rows before droping cars without gearbox", len(clean))
clean = clean.dropna(subset=["gearbox"])
print("Rows after droping cars without gearbox", len(clean))
clean["manual"] = np.where(clean["gearbox"] == "manuell", 1, 0)
del clean["gearbox"]

DEFAULT_MANUAL = 1
clean["model"].describe()
del clean["model"]
clean["fuelType"].describe()
set(clean["fuelType"])
clean["fuelType"].value_counts()
clean["fuelType"].value_counts().plot(kind='bar', title='Cars by fuel type')
print("Rows without a fuel type", clean["fuelType"].isna().sum())
clean = clean.dropna(subset=["fuelType"])  # Lets drop them...
DEFAULT_FUEL_TYPE = "benzin"
clean["brand"].describe()
clean["brand"].value_counts()
clean["brand"].value_counts().plot(kind='bar', figsize=(20, 5), title='Cars by brand')
DEFAULT_BRAND = "volkswagen"
clean["notRepairedDamage"].describe()
clean["notRepairedDamage"].value_counts()
clean["notRepairedDamage"].value_counts().plot(kind='pie', title='Cars with damage')
print("Rows without a notRepairedDamage spec", clean["notRepairedDamage"].isna().sum())
clean = clean.dropna(subset=["notRepairedDamage"])  # Lets drop them...
clean["has_damage"] = np.where(clean["notRepairedDamage"] == "ja", 1, 0)
del clean["notRepairedDamage"]
DEFAULT_HAS_DAMAGE = 0
del clean["dateCreated"]
del clean["dateCrawled"]
del clean["offerType"]
del clean["lastSeen"]
clean.dtypes
clean = pd.concat([clean, pd.get_dummies(clean["fuelType"], prefix="fuelType")], axis=1)
clean = pd.concat([clean, pd.get_dummies(clean["vehicleType"], prefix="vehicleType")], axis=1)
clean = pd.concat([clean, pd.get_dummies(clean["brand"], prefix="brand")], axis=1)
clean.drop(["fuelType", "vehicleType", "brand"], axis=1, inplace=True)
clean.head()
print("Cheap cars (less than 2000 euros)", len(clean[clean["price"] <= 2000]))
print("Non - Cheap cars (more than 2000 euros)", len(clean[clean["price"] > 2000]))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y = clean["price"]
del clean["price"]
scaler.fit(clean)
clean_normalized = pd.DataFrame(scaler.transform(clean), columns=clean.columns)
clean_normalized.head()
clean_normalized.describe()
print("We have a total of ", clean_normalized.shape[0], " columns")
np.random.seed(42)
from sklearn.model_selection import train_test_split
X_normalized = clean_normalized
X_train, X_not_train, y_train, y_not_train = train_test_split(X_normalized, y, test_size=0.3)
X_validation, X_test, y_validation, y_test = train_test_split(X_not_train, y_not_train, test_size=0.5)
print("Original size was", X_normalized.shape[0])
print("X_train", X_train.shape[0])
print("X_validation", X_validation.shape[0])
print("X_test", X_test.shape[0])
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
# Lets put everything together
def evaluate_model(model, X_train, y_train, X_validation, y_validation):
    y_train_prediction = model.predict(X_train)
    y_validation_prediction = model.predict(X_validation)
    print("train error\t\t\t", mean_squared_error(y_train, y_train_prediction))
    print("validation error\t\t", mean_squared_error(y_validation, y_validation_prediction))
    print("train error (absolute))\t\t", mean_absolute_error(y_train, y_train_prediction))
    print("validation error (absolute))\t", mean_absolute_error(y_validation, y_validation_prediction))
    
evaluate_model(linear_model, X_train, y_train, X_validation, y_validation)
from sklearn.neural_network import MLPRegressor
nn_model = MLPRegressor(hidden_layer_sizes=(128,), # Pretty much a random value here to start playing with
                        max_iter=150, # Some base line 
                        verbose=True, # Well, I am just too eager :)
                        random_state=42, # We want repeatable experiments
                        alpha=0.0 #  No regularization by default... IF we have overfitting, then we will look into it
                       )
nn_model.fit(X_train, y_train )
pd.DataFrame(nn_model.loss_curve_, columns=["loss"]).plot(title='Learning curve for the training set. loss='+str(nn_model.loss_), 
                                                          figsize=(10, 5))
evaluate_model(nn_model, X_train, y_train, X_validation, y_validation)
y_train_predictions = nn_model.predict(X_train)
y_validation_predictions = nn_model.predict(X_validation)
from sklearn.metrics import accuracy_score
def evaluate_final_model(y_true, y_pred):
    df = pd.DataFrame()
    df["true"] = np.where(y_true <= 2000, 1, 0)
    df["pred"] = np.where(y_pred <= 2000, 1, 0)
    
    return accuracy_score(df["true"], df["pred"]), f1_score(df["true"], df["pred"])    
print("NN Accuracy for training data %.5f, f1 score %.5f" % evaluate_final_model(y_train, nn_model.predict(X_train)))
print("NN Accuracy for validation data %.5f, f1 score %.5f" %  evaluate_final_model(y_validation, nn_model.predict(X_validation)))
print("Linear Accuracy for training data  %.5f, f1 score %.5f" % evaluate_final_model(y_train, linear_model.predict(X_train)))
print("Linear Accuracy for validation data  %.5f, f1 score %.5f" % evaluate_final_model(y_validation, linear_model.predict(X_validation)))
from  sklearn.ensemble import RandomForestRegressor
model_forest = RandomForestRegressor(random_state=42)

model_forest.fit(X_train, y_train)
print("Random forest Accuracy for training data  %.5f, f1 score %.5f" % evaluate_final_model(y_train, model_forest.predict(X_train)))
print("Random forestAccuracy for validation data  %.5f, f1 score %.5f" % evaluate_final_model(y_validation, model_forest.predict(X_validation)))
from sklearn.model_selection import GridSearchCV

param_grid = { 
    'n_estimators': [10, 25, 50, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [None, 2, 4, 8, 16],
    'criterion' :['mse']
}

grid_search = GridSearchCV(estimator=model_forest, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Random forest (grid search) Accuracy for training data %.5f, f1 score %.5f" % evaluate_final_model(y_train, grid_search.predict(X_train)))
print("Random forest (grid search) Accuracy for validation data %.5f, f1 score %.5f" % evaluate_final_model(y_validation, grid_search.predict(X_validation)))
models = {"linear": linear_model, 
         "neural_network": nn_model,
         "random_forest": model_forest}
train_ensemble = pd.DataFrame()


train_ensemble["true"] = np.where(y_train <= 2000, 1, 0)

for model_name, model in models.items():
    predictions = model.predict(X_train)
    train_ensemble[model_name] = np.where(predictions <= 2000, 1, 0)
    
train_ensemble["total"] = train_ensemble["linear"] + train_ensemble["neural_network"] + train_ensemble["random_forest"]
print("Accuracy", accuracy_score(train_ensemble["true"], np.where(train_ensemble["total"] >= 1, 1, 0)))
print("F1 ", f1_score(train_ensemble["true"], np.where(train_ensemble["total"] >= 1, 1, 0)))
print("Accuracy ", accuracy_score(train_ensemble["true"], np.where(train_ensemble["total"] >= 2, 1, 0)))
print("F1 ", f1_score(train_ensemble["true"], np.where(train_ensemble["total"] >= 2, 1, 0)))
print("Accuracy ", accuracy_score(train_ensemble["true"], np.where(train_ensemble["total"] >= 3, 1, 0)))
print("F1 ", f1_score(train_ensemble["true"], np.where(train_ensemble["total"] >= 3, 1, 0)))
print("Random forest, test results Accuracy %.5f, F1 %.5f " % evaluate_final_model(y_test, model_forest.predict(X_test)))
print("linear model, test results Accuracy %.5f, F1 %.5f " % evaluate_final_model(y_test, linear_model.predict(X_test)))
print("neural network, test results Accuracy %.5f, F1 %.5f " % evaluate_final_model(y_test, nn_model.predict(X_test)))
def prepare_data(to_predict):
    DEF
