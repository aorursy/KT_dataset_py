import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

HOUSING_PATH = "/kaggle/input/usa-housing/"
FILE_NAME = "USA_Housing.csv"

def load_data(housing_path = HOUSING_PATH, file_name = FILE_NAME):
    csv_path = os.path.join(housing_path, file_name)
    return pd.read_csv(csv_path)

housing = load_data()

housing.head()
housing.info()
# This will turn the labels into integers instead of floats
housing["income_cat"] = pd.cut(housing["Avg. Area Income"],
                               bins=[0, 40000, 60000, 80000, 90000, 110000],
                               labels=[1, 2, 3, 4, 5])

pd.cut(housing["Avg. Area Income"],
             bins=[0, 40000, 60000, 80000, 90000, 110000]).value_counts()

housing["income_cat"].value_counts()
housing["income_cat"].value_counts() / len(housing) *100
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Drop the 'income_cat' so that the data is back to its original state
for set in (strat_train_set,strat_test_set):
    set.drop(["income_cat"],axis=1,inplace=True)
housing = strat_train_set.copy()
housing
corr_matrix = housing.corr()
corr_matrix["Price"].sort_values(ascending=False)
import matplotlib.pyplot as plt

attributes = ["Price", "Avg. Area Income", "Avg. Area House Age", "Area Population"]
pd.plotting.scatter_matrix(housing[attributes],figsize=(12,8))
plt.show()

housing.plot(kind="scatter", x="Avg. Area Income", y="Price", alpha=0.5)
plt.show()
housing = strat_train_set.drop("Price", axis=1)
housing_labels = strat_train_set["Price"].copy()
housing
housing_labels
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("Address",axis=1)
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
# Returns a Numpy array
X = imputer.transform(housing_num)
# Turn the array into a Pandas DataFrame
housing_tr = pd.DataFrame(X,columns=housing_num.columns)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])

housing_tr = pipeline.fit_transform(housing_num)
housing_tr
from sklearn.ensemble import RandomForestRegressor

my_model = RandomForestRegressor()
my_model.fit(housing_tr, housing_labels)

# Get some data and some labels. That last 10 instances
some_data = housing_num.iloc[:10]
some_labels = housing_labels.iloc[:10]

some_data_prepared = pipeline.transform(some_data)
from sklearn.metrics import mean_squared_error

housing_predictions = my_model.predict(housing_tr)
my_model_mse = mean_squared_error(housing_labels,housing_predictions)
my_model_rmse = np.sqrt(my_model_mse)

"Model Root Mean Squared Error:", my_model_rmse
from sklearn.model_selection import cross_val_score

my_model_scores = cross_val_score(RandomForestRegressor(),
                  housing_tr,housing_lables,scoring="neg_mean_squared_error",cv=10)
my_model_rmse_scores = np.sqrt(-my_model_scores)

def display_scores(scores):
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

display_scores(my_model_rmse_scores)
# Set prediction price and actual price variables
predictions = np.ceil(my_model.predict(some_data_prepared))
actuals = np.ceil(list(some_labels))

for p, a in list(zip(predictions, actuals)):
    percentage_diff = (a - p) / a * 100
    percentage_diff = round(percentage_diff,2)
    print("Prediction:", int(p), "Actual:", int(a), "Percentage difference:", percentage_diff)
