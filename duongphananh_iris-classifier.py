# General imports



import numpy as np # I don't think I used numpy at all, but I'll keep it here

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

%pylab inline
# Import scikit-learn libraries here



from sklearn.model_selection import train_test_split

"""

This is for splitting the training and testing data



If you train your model on all of the data,

it might get really good at predicting, but only on that dataset

because it can simply memorize the data. By splitting the data

into a train, and a test set, we can score the model on the test set

and get a more reflective result compare to real world data

"""

from sklearn.ensemble import RandomForestClassifier

"""

A RandomForest model is a cluster of DecisionTree models,

committing votes amongst its trees to determine the final results.

However, RandomForests tend to overfit, meaning being really good at

predicting only on that dataset, so tuning its hyperparameters is necessary.

"""

from sklearn.model_selection import GridSearchCV

"""

That brings us to GridSearchCV. This lets us look through various combinations

of hyperparameters and determine the best one.

"""
# Load the data, and set the "Id" column as index

# .read_csv() allows us to parse through a .csv file and return a pandas Dataframe

data = pd.read_csv("../input/iris/Iris.csv", index_col="Id")

# The .head() method allows us to peek at the first 5 rows of a Dataframe

# If you want to look at the final rows, you can use .tail()

data.head()
# Load proper columns into X

# The columns used for the model to infer info is traditionally called X

X_labels = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

X = data[X_labels]

X.head()
# Target column is traditionally named y

y = data["Species"]

# Because it only contains one column, this is a pandas Series, not a Dataframe

y.head()
# The .describe method allows us to see a Dataframe's statistical values

data.describe()
# The .unique() methods return an array of unique values in a Series

y.unique()
# The .groupby() function is quite interesting

# It lets us group rows depending on their unique values in a column

# Then perform a statistical operation on it like .median(), .std()

data.groupby(["Species"]).mean()
# X.corr() returns the correlation values of every pair of data in a Dataframe

sns.heatmap(X.corr(), annot=True)
sns.pairplot(data=data, hue="Species")
# We now divide the Length by the Width, then round it to the nearest hundred for the new ratio column

data["SepalWidthLengthRatioCm"] = round(data["SepalLengthCm"] / data["SepalWidthCm"], 2)

# Attach it to the X labels

X_labels.append("SepalWidthLengthRatioCm")

X = data[X_labels]

# Then drop the other two columns as they're no longer necessary

X = X.drop(["SepalLengthCm", "SepalWidthCm"], axis=1)
sns.swarmplot(data=data, x="Species", y="SepalWidthLengthRatioCm")
# We use a lambda along with ternary operators to get our results

data["SpeciesEncoded"] = data["Species"].apply(lambda x: 1 if x =="Iris-setosa" else 2 if x =="Iris-versicolor" else 3)

# Set y to the new encoded column

y = data["SpeciesEncoded"]

data.head()
data["SpeciesEncoded"].unique()
X.head()
y.head()
# The random_state is for the randomization process to be predictable

# Meaning it'll randomize the same for every run, ensuring a stable result

rf = RandomForestClassifier(random_state=0)
# We split the data here

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("There are {} samples in the training set and {} samples in the test set".format(X_train.shape[0], X_test.shape[0]))
grid_params  = {

    'bootstrap' : [True, False],

    'n_estimators' : list(range(10,101,10)),

    'criterion' : ['gini', 'entropy'],

    'min_samples_leaf' : list(range(1,10,2)),

    'max_features' : ['sqrt', 'log2']

}
grid_search = GridSearchCV(estimator=rf, param_grid=grid_params, cv=3, verbose=1)
grid_search.fit(X_train, y_train)
# Here are the best parameters that GridSearch found

grid_search.best_params_
final_model = grid_search

# It scored a ~97% accuracy, pretty good!

final_model.score(X_test, y_test)

"""

Be wary with high accuracy results though. 

Yes, it can mean that your model is really good,

or it can mean that your model is really good on that one dataset.

Whenever you are suspicious, try tuning the hyperparameters again,

or test it out on real data.

"""
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(final_model, X_test, y_test)