import numpy as np

import pandas as pd



from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
# Read all data from input file

data = pd.read_csv("../input/train.csv")
# Overview of columns and their data types

print(data.dtypes)



print()

print("Number of passengers in training data set:")

print(len(data))



# Let's also look at some examples

data.head()
# What about missing data?

# The columns "Age", "Cabin", and "Embarked" are missing data

print("Missing values by column:")

print(data.loc[:, data.isnull().any()].isnull().sum())



# Overview of numerical columns

data.describe()
# Naive way to deal with missing values... 

data["AgeCompleted"] = data["Age"]

data.loc[np.isnan(data["AgeCompleted"]), "AgeCompleted"] = np.mean(data["Age"])
# Checking that there are only two values in the Sex column

print("Genders in data:")

print(sorted(data["Sex"].unique()))

print()



# Create 'Title' column to hold title values derived from names

print("Different titles in data:")

data["Title"] = data["Name"].apply(lambda s: s.split(',')[1].split('.')[0].strip())

print(data["Title"].value_counts())

print()



print("Sample of ticket numbers in data:")

print(data["Ticket"].unique()[:10])
# Sex

data["IsFemale"] = 0

data.loc[data["Sex"] == "female", "IsFemale"] = 1



# There is one person on the 'T' deck, but other than that, all people are on decks A to G

decks = list("ABCDEFG")



for deck in decks:

    data["Deck" + deck] = 0

    data["Deck" + deck] = data["Cabin"].apply(lambda s: 1 if isinstance(s, str) and s.startswith(deck) else 0)



# Embarked

ports = list("CQS")



for port in ports:

    data["Port" + port] = 0

    data["Port" + port] = data["Embarked"].apply(lambda s: int(s == port))



data.head()
features = [

    #'PassengerId', 

    #'Survived', 

    'Pclass', 

    #'Name', 

    #'Sex', 

    #'Age', 

    'SibSp', 

    'Parch', 

    #'Ticket', 

    'Fare', 

    #'Cabin', 

    #'Embarked', 

    'AgeCompleted',

    #'Title', 

    'IsFemale', 

    'DeckA', 'DeckB', 'DeckC', 'DeckD', 'DeckE', 'DeckF', 'DeckG',

    'PortC', 'PortQ', 'PortS'

]
# Travelling with family?

data["WithFamily"] = 0

data.loc[(data["SibSp"] > 0) | (data["Parch"] > 0), "WithFamily"] = 1



# Family size

data["FamilySize"] = data["SibSp"] + data["Parch"]



# Age group

data["IsChild"] = 0

data.loc[data["Age"] < 10, "IsChild"] = 1

data["IsYoung"] = 0

data.loc[(data["Age"] >= 10) & (data["Age"] <= 20), "IsYoung"] = 1

data["IsAdult"] = 0

data.loc[data["Age"] > 20, "IsAdult"] = 1



features += ["WithFamily", "FamilySize", "IsChild", "IsYoung", "IsAdult"]
print("We now have a total of", len(features), "features.")



# Unscaled data

X = data[features].copy()

y = np.array(np.transpose([data["Survived"]]))



# Split into training and validation sets

train_idx = np.random.rand(len(X)) < 0.8



X_train = X[train_idx]

y_train = y[train_idx]

train_size = len(X_train)

X_val = X[~train_idx]

y_val = y[~train_idx]

val_size = len(X_val)



print()

print("Training set size:", np.shape(X_train))

print("Validation set size:", np.shape(X_val))
# Scale both sets based on scaling obtained from training data

scaler = StandardScaler()

scaler.fit(X_train)



scaled_X_train = scaler.transform(X_train)

scaled_X_val = scaler.transform(X_val)



# The mean on the training set will be close to zero

# The mean on the validation should be as well, but probably not as close

print(np.mean(scaled_X_train))

print(np.mean(scaled_X_val))
def error(predictions, y):

    real = y.ravel()

    total = np.shape(predictions)[0]

    correct = np.sum(predictions == real)

    return 1 - (correct / total)
# We want to use a suitable amount of regularization to prevent overfitting

# so we iterate over different Lambda values and choose whatever gets us

# the best performance on the validation set.

LAMBDAS = [1e-5, 1e-3, 1e-1, 1, 3, 10, 30]



num_features = len(data.columns)



training_errors = []

validation_errors = []



smallest_validation_error = np.inf

best_classifier = None



for LAMBDA in LAMBDAS:

    classifier = MLPClassifier(

        solver='lbfgs',

        alpha=LAMBDA,  # Regularization parameter

        hidden_layer_sizes=(num_features, num_features),  # Two hidden layers

        random_state=1

    )

    classifier.fit(scaled_X_train, y_train.ravel())

    

    pred_train = classifier.predict(scaled_X_train)

    pred_val = classifier.predict(scaled_X_val)

    

    err_train = error(pred_train, y_train)

    training_errors.append(err_train)

    err_val = error(pred_val, y_val)

    validation_errors.append(err_val)

    

    if err_val < smallest_validation_error:

        best_classifier = classifier

        smallest_validation_error = err_val
import matplotlib.pyplot as plt

import seaborn as sns



print("Validation error when lambda = 10:", validation_errors[LAMBDAS.index(10)])



plt.figure()

plt.plot(LAMBDAS, training_errors, label="Training error")

plt.plot(LAMBDAS, validation_errors, label="Validation error")

plt.legend()

plt.xscale('log')

plt.axvline(10, color='red', linestyle=":")

plt.show()