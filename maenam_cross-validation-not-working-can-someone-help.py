import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor
def randomforest(training_set, validation_set):



    target = training_set["SalePrice"].values



    # Prepare the data first

    all_data = pd.concat((training_set, validation_set))

    all_data = all_data.fillna(all_data.median())

    all_data = pd.get_dummies(all_data)

    training_set = all_data[0:len(target)]

    validation_set = all_data[len(target):]

    training_set.drop("SalePrice", axis=1)

    validation_set.drop("SalePrice", axis=1)



    forest = RandomForestRegressor(min_samples_split=3, n_estimators=500, random_state=1)



    training_features = training_set.ix[:, 1:].values

    forest.fit(training_features, target)



    # Make a prediction

    validation_features = validation_set.ix[:, 1:].values

    my_prediction = forest.predict(validation_features)



    # Save it in a dataframe

    id = np.array(validation_set["Id"]).astype(int)

    my_solution = pd.DataFrame(my_prediction, id, columns=["SalePrice"])

    return my_solution
def testsolution(my_solution, validation_set):

    prediction = my_solution["SalePrice"].values

    reality = validation_set["SalePrice"].values

    n = len(reality)



    error = np.sqrt((1/n) * sum((np.log(prediction + 1) - np.log(reality + 1))**2))

    return error
train = pd.read_csv("../input/train.csv")

n = 5

errors = np.zeros(n)

for k in range(0, n):

    print(n - k)

    train = train.reindex(np.random.permutation(train.index))

    validation_data = train[0:146]

    training_data = train[146:1460]



    solution = randomforest(training_data, validation_data)

    errors[k] = testsolution(solution, validation_data)



print(errors)