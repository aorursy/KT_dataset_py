import pandas as pd



cars_data = pd.read_csv("../input/car-data/car_data.csv", index_col = "car_name")
cars_data.shape
cars_data.head()
cars_data.describe().T
cars_data.groupby("cylinders").count()
cars_data = cars_data[cars_data.cylinders == 4]

cars_data = cars_data.drop("cylinders", axis = 1)

cars_data.head()
cars_data.corr()
knn_regression_data = cars_data.loc[:,["horsepower","weight", "mpg","displacement"]]

knn_regression_data.head()
import numpy as np



knn_regression_data = (knn_regression_data - np.min(knn_regression_data))/(np.max(knn_regression_data) - np.min(knn_regression_data))

knn_regression_data.describe().T
knn_regression_data.dtypes
knn_independent = knn_regression_data.drop("displacement", axis = 1)

knn_dependent = knn_regression_data["displacement"] # I want estimate to acceleration
from sklearn.model_selection import train_test_split



independent_train, independent_test, dependent_train, dependent_test = train_test_split(

    knn_independent, 

    knn_dependent, 

    test_size = 0.10, 

    random_state = 20)
from sklearn.neighbors import KNeighborsRegressor



knn_model = KNeighborsRegressor().fit(independent_train, dependent_train)

predicted_values = knn_model.predict(independent_test)
predict_df = pd.DataFrame({"Dependent_Test" : dependent_test, "Dependent_Predicted" : predicted_values})

predict_df.head()
predict_df = (predict_df*(np.max(cars_data.displacement) - np.min(cars_data.displacement))) + np.min(cars_data.displacement)

predict_df.head()
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np



print("Mean Squared Error = ", mean_squared_error(predict_df.Dependent_Predicted, predict_df.Dependent_Test))

print("Root Mean Squared Error = ", np.sqrt(mean_squared_error(predict_df.Dependent_Predicted, predict_df.Dependent_Test)))
r2_score(predict_df.Dependent_Predicted, predict_df.Dependent_Test)
from sklearn.model_selection import GridSearchCV

import numpy as np



knn_params = {"n_neighbors" : np.arange(1,11,1)}

knn = KNeighborsRegressor()

knn_cv_model = GridSearchCV(knn, knn_params, cv = 10)

knn_cv_model.fit(independent_train, dependent_train)
knn_cv_model.best_params_["n_neighbors"]
knn_model = KNeighborsRegressor(n_neighbors = knn_cv_model.best_params_["n_neighbors"]).fit(independent_train, dependent_train)

predicted_values = knn_model.predict(independent_test)
predict_df = pd.DataFrame({"Dependent_Test" : dependent_test, "Dependent_Predicted" : predicted_values})
predict_df = (predict_df*(np.max(cars_data.displacement) - np.min(cars_data.displacement))) + np.min(cars_data.displacement)

predict_df.head()
print("Mean Squared Error = ", mean_squared_error(predict_df.Dependent_Test, predict_df.Dependent_Predicted))

print("Root Mean Squared Error = ", np.sqrt(mean_squared_error(predict_df.Dependent_Test, predict_df.Dependent_Predicted)))
r2_score(predict_df.Dependent_Test, predict_df.Dependent_Predicted)
from sklearn.model_selection import cross_val_score



MSE = []

MSE_CV = []



for k in range(10):

    k = k + 1

    knn_model = KNeighborsRegressor(n_neighbors = k).fit(independent_train, dependent_train)

    y_pred = knn_model.predict(independent_test)

    mse = mean_squared_error(y_pred, dependent_test)

    mse_cv = -1 * cross_val_score(knn_model, independent_train,dependent_train, cv = 10,

                         scoring = "neg_mean_squared_error").mean()

    MSE.append(mse)

    MSE_CV.append(mse_cv)

    print("k =", k, "MSE :", mse, "MSE_CV:", mse_cv)
import matplotlib.pyplot as plt



plt.plot(np.arange(1,11,1), MSE)

plt.plot(np.arange(1,11,1), MSE_CV)

plt.xlabel("Value of K for KNN")

plt.ylabel("Testing Accurracy");