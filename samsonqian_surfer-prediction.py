import numpy as np 

import pandas as pd 



import seaborn as sns

from matplotlib import pyplot as plt

sns.set_style("whitegrid")

%matplotlib inline



from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import make_scorer, accuracy_score 

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import GridSearchCV





import warnings

warnings.filterwarnings("ignore")



import os
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
print("These are all the columns for the data: ", train.columns)

print("There are " + str(len(train.columns)) + " columns.")
train.describe()
train.corr()
train.isnull().sum()
test.isnull().sum()
# Drop useless features from data

train = train.drop(["id", "name", "favorite_color"], axis=1)

testing = test.drop(["id", "name", "favorite_color"], axis=1)
le = LabelEncoder()

le.fit(train["city"])



train["city"] = le.transform(train["city"])

testing["city"] = le.transform(testing["city"])
def one_hot(data, col):

    for value in set(data[col]):

        encode = (data[col] == value).astype("int")

        data[value] = encode

    return data.drop([col], axis=1)

        

train = one_hot(train, "city")

testing = one_hot(testing, "city")
def scale(data, method="standard"):

    """

    Scales the inputted data by standardizing its values.

  

    :param data: data to be scaled

    :param method: how to scale data

    :return: standardized version of the data

    """

    scaler = StandardScaler() if method=="standard" else MinMaxScaler()

    arr = np.array(data).reshape(-1, 1)

    return scaler.fit_transform(arr)





columns = ["age", "salary"]



for column in columns:

    train[column] = scale(train[column])

    testing[column] = scale(testing[column])
train.head()
def prepare_data(df, valid_split=0.2):

    """

    Splits and returns the training and validation sets for the data. 



    :param df: preprocessed dataset

    :returns: training sets and validation sets

    """

    X_train = df.drop("surf", axis=1) # define training features set

    y_train = df["surf"] # define training label set

    # we will use 20% of the trainig data as validation data

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_split, random_state=0) # X_valid and y_valid are the validation sets

    return X_train, X_valid, y_train, y_valid





X_train, X_valid, y_train, y_valid = prepare_data(train)
def train_and_evaluate(model, parameters, train_X, train_y, valid_X, valid_y):

    """

    Trains and evalutaes a model based on training/validation data.

  

    :param model: model to fit

    :param parameters: model parameters

    :param train_X: training features

    :param train_y: training label

    :param valid_X: validation features

    :param valid_y: validation label

    :return: accuracy of model on validation set

    """

    grid_search = GridSearchCV(model, parameters, scoring=make_scorer(accuracy_score))

    grid_search.fit(train_X, train_y)

    model = grid_search.best_estimator_

    model.fit(train_X, train_y)

    

    predictions = model.predict(valid_X)

    return model, accuracy_score(valid_y, predictions)





knn_params = {"n_neighbors": [3, 5, 10, 15], "weights": ["uniform", "distance"], "algorithm": ["auto", "ball_tree", "kd_tree"],

              "leaf_size": [20, 30, 50]}

knn, knn_acc = train_and_evaluate(KNeighborsClassifier(), knn_params, X_train, y_train, X_valid, y_valid)



gnb_params = {}

gnb, gnaivebayes_acc = train_and_evaluate(GaussianNB(), gnb_params, X_train, y_train, X_valid, y_valid)



bnb_params = {}

bnb, bnaivebayes_acc = train_and_evaluate(BernoulliNB(), bnb_params, X_train, y_train, X_valid, y_valid)
model_performances = pd.DataFrame({

    "Model": ["K Nearest Neighbors", "Gaussian Naive Bayes", "Bernoulli Naive Bayes"],

    "Accuracy": [knn_acc, gnaivebayes_acc, bnaivebayes_acc]

})



model_performances.sort_values(by="Accuracy", ascending=False)
model = gnb.fit(train.drop(["surf"], axis=1), train["surf"])

preds = model.predict(testing)



predictions = pd.DataFrame({

        "id": test["id"],

        "surf": preds

    })

    

predictions.to_csv("predictions.csv", index=False)

predictions