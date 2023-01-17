import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline



diamonds = pd.read_csv("../input/diamonds/diamonds.csv")
diamonds.head()
diamonds.info()
diamonds = diamonds.drop("Unnamed: 0", axis = 1)



diamonds["price"] = diamonds["price"].astype(float)



diamonds.head()
diamonds["cut"].value_counts()
diamonds["color"].value_counts()
diamonds["clarity"].value_counts()
diamonds.describe()
diamonds.hist(bins = 50, figsize = (20, 15))

plt.show()
corr_matrix = diamonds.corr()



plt.subplots(figsize = (10, 8))

sns.heatmap(corr_matrix, annot = True)

plt.show()
diamonds["carat"].hist(bins = 50)

plt.show()
diamonds["carat_cat"] = np.ceil(diamonds["carat"] / 0.35)



diamonds["carat_cat"].where(diamonds["carat_cat"] < 5, 5.0, inplace = True)
diamonds["carat_cat"].value_counts()
diamonds["carat_cat"].hist()

plt.show()
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)



for train_index, test_index in split.split(diamonds, diamonds["carat_cat"]):

    strat_train_set = diamonds.loc[train_index]

    strat_test_set = diamonds.loc[test_index]
for set in (strat_train_set, strat_test_set):

    set.drop(["carat_cat"], axis = 1, inplace = True)
diamonds = strat_train_set.copy()

diamonds.head()
sns.pairplot(diamonds[["price", "carat", "cut"]], hue = "cut", height = 5)

plt.show()

sns.barplot(x = "carat", y = "cut", data = diamonds)

plt.show()

sns.barplot(x = "price", y = "cut", data = diamonds)

plt.show()
sns.pairplot(diamonds[["price", "carat", "color"]], hue = "color", height = 5)

plt.show()

sns.barplot(x = "carat", y = "color", data = diamonds)

plt.show()

sns.barplot(x = "price", y = "color", data = diamonds)

plt.show()
sns.pairplot(diamonds[["price", "carat", "clarity"]], hue = "clarity", height = 5)

plt.show()

sns.barplot(x = "carat", y = "clarity", data = diamonds)

plt.show()

sns.barplot(x = "price", y = "clarity", data = diamonds)

plt.show()
diamonds = strat_train_set.drop("price", axis = 1)



diamond_labels = strat_train_set["price"].copy()



diamonds_num = diamonds.drop(["cut", "color", "clarity"], axis = 1)

diamonds_num.head()
from sklearn.preprocessing import StandardScaler



num_scaler = StandardScaler()

diamonds_num_scaled = num_scaler.fit_transform(diamonds_num)



pd.DataFrame(diamonds_num_scaled).head()
diamonds_cat = diamonds[["cut", "color", "clarity"]]

diamonds_cat.head()

from sklearn.preprocessing import OneHotEncoder



cat_encoder = OneHotEncoder()

diamonds_cat_encoded = cat_encoder.fit_transform(diamonds_cat)



pd.DataFrame(diamonds_cat_encoded.toarray()).head()
from sklearn.compose import ColumnTransformer



num_attribs = list(diamonds_num)

cat_attribs = ["cut", "color", "clarity"]



pipeline = ColumnTransformer([

    ("num", StandardScaler(), num_attribs), 

    ("cat", OneHotEncoder(), cat_attribs) 

])
diamonds_ready = pipeline.fit_transform(diamonds)



pd.DataFrame(diamonds_ready).head()
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from random import randint



X_test = strat_test_set.drop("price", axis = 1)

y_test = strat_test_set["price"].copy()



models_rmse = []

cvs_rmse_mean = []

tests_rmse = []

tests_accuracy = []

models = []



def display_model_performance(model_name, model, diamonds = diamonds_ready, labels = diamond_labels,

                              models_rmse = models_rmse, cvs_rmse_mean = cvs_rmse_mean, tests_rmse = tests_rmse,

                              tests_accuracy = tests_accuracy, pipeline = pipeline, X_test = X_test,

                              y_test = y_test, cv = True):



    model.fit(diamonds, labels)

    

    predictions = model.predict(diamonds)

    

    model_mse = mean_squared_error(labels, predictions)

    model_rmse = np.sqrt(model_mse)

    

    cv_score = cross_val_score(model, diamonds, labels, scoring = "neg_mean_squared_error", cv = 10)

    cv_rmse = np.sqrt(-cv_score)

    cv_rmse_mean = cv_rmse.mean()

    

    print("RMSE: %.4f" %model_rmse)

    models_rmse.append(model_rmse)

    

    print("CV-RMSE: %.4f" %cv_rmse_mean)

    cvs_rmse_mean.append(cv_rmse_mean)

    

    print("--- Test Performance ---")

    

    X_test_prepared = pipeline.transform(X_test)

    

    model.fit(X_test_prepared, y_test)

    

    test_predictions = model.predict(X_test_prepared)

    

    test_model_mse = mean_squared_error(y_test, test_predictions)

    test_model_rmse = np.sqrt(test_model_mse)

    print("RMSE: %.4f" %test_model_rmse)

    tests_rmse.append(test_model_rmse)

    

    test_accuracy = round(model.score(X_test_prepared, y_test) * 100, 2)

    print("Accuracy:", str(test_accuracy)+"%")

    tests_accuracy.append(test_accuracy)

    

    start = randint(1, len(y_test))

    some_data = X_test.iloc[start:start + 7]

    some_labels = y_test.iloc[start:start + 7]

    some_data_prepared = pipeline.transform(some_data)

    print("Predictions:\t", model.predict(some_data_prepared))

    print("Labels:\t\t", list(some_labels))

    

    models.append(model_name)

    

    plt.scatter(diamond_labels, model.predict(diamonds_ready))

    plt.xlabel("Actual")

    plt.ylabel("Predicted")

    x_lim = plt.xlim()

    y_lim = plt.ylim()

    plt.plot(x_lim, y_lim, "k--")

    plt.show()

    

    print("------- Test -------")

    plt.scatter(y_test, model.predict(X_test_prepared))

    plt.xlabel("Actual")

    plt.ylabel("Predicted")

    plt.plot(x_lim, y_lim, "k--")

    plt.show()
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression(normalize = True)

display_model_performance("Linear Regression", lin_reg)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators = 10, random_state = 42)

display_model_performance("Random Forest Regression", forest_reg)
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(random_state = 42)

display_model_performance("Decision Tree Regression", tree_reg)
compare_models = pd.DataFrame({ "Algorithms": models, "Models RMSE": models_rmse, "CV RMSE Mean": cvs_rmse_mean,

                              "Tests RMSE": tests_rmse, "Tests Accuracy": tests_accuracy })

compare_models.sort_values(by = "Tests Accuracy", ascending = False)
sns.barplot(x = "Tests Accuracy", y = "Algorithms", data = compare_models)

plt.show()
import pickle



with open('final_model.pkl', 'wb') as f:

    pickle.dump(tree_reg, f)