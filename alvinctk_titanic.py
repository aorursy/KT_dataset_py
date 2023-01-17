import os

print(os.listdir("../input"))
# import necessary libraries

from matplotlib import pyplot as plt

import pandas as pd

import numpy as np 
def read_data(filename):

    """

    Returns data dataframe

    """

    df = pd.read_csv(filename)

    rows, columns = df.shape

    name = filename.split("/")[-1].split(".")[0]

    

    print("{}'s row = {}".format(name, rows))

    print("{}'s column = {}".format(name, columns))

    

    return df



test = read_data("../input/test.csv")

train = read_data("../input/train.csv")

train.head()
def remove_border(plt, legend=None):

    if plt is None:

        return

    

    # Remove plot border

    for sphine in plt.gca().spines.values():

        sphine.set_visible(False)

        

    # Create legend and remove legend box

    if legend is None:

        plt.legend(frameon=False)

    else:

        plt.legend(legend, frameon=False)

    

    # Remove ticks 

    plt.tick_params(left=False, bottom=False) 
sex_pivot = train.pivot_table(index="Sex", values="Survived")

ax = sex_pivot.plot.barh()

ax.set_ylabel("Count")

remove_border(plt)

plt.show()
print(train["Age"].describe())
survived = train[train["Survived"] == 1]

died = train.drop(survived.index)



# Survived and died histogram in a single plot

for df, color in [(survived, "red"), (died,"blue")]:

    ax = df["Age"].plot.hist(alpha=.5, color=color, bins=50)

    ax.set_xlabel("Age")

    

remove_border(plt, legend=["Survived", "Died"])



plt.title("Survived and Died Histogram")

plt.show()



print(train["Age"].describe())
def process_age(df, cut_point=None, label_names=None):

    # Fill missing age with negative values to denote missing category

    df["Age"] = df["Age"].fillna(-0.5)

    

    # If cut point and label names are not provided, default ranges used.

    if cut_point is None and label_names is None:

        age_ranges = [("Missing", -1, 0), ("Infant", 0, 5), ("Child", 5, 12), 

                      ("Teenager", 12, 18), ("Young Adult", 18, 35), ("Adult", 35, 60),

                      ("Senior", 60, 100)]

    

        cut_points = [x for _, x, _ in age_ranges]

        cut_points.append(age_ranges[-1][-1])

        label_names = [labels for labels, *_ in age_ranges]

    

    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)

    return df



train = process_age(train)

train_original_columns = train.columns

test = process_age(test)

train.pivot_table(index="Age_categories", values="Survived").plot.barh()

remove_border(plt)

#train.pivot_table(index="Age_categories")["Survived"].plot.bar()

plt.show()

def create_dummies(df, column):

    return pd.concat([df, pd.get_dummies(df[column], prefix = column)], axis=1)



for col in ['Pclass', "Sex", "Age_categories"]:

    train = create_dummies(train, col)

    test = create_dummies(test, col)

train.columns
target = "Survived"

features = train.drop(columns=train_original_columns).columns

features
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear')

lr.fit(train[features], train[target])

lr
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(train[features], train[target], 

                                                    test_size=0.2, random_state=0)
from sklearn.metrics import accuracy_score

lr = LogisticRegression(solver='liblinear')

lr.fit(train_x, train_y)

predictions = lr.predict(test_x)

accuracy = accuracy_score(test_y, predictions)

accuracy
from sklearn.model_selection import cross_val_score

from numpy import mean

lr = LogisticRegression(solver='liblinear')

scores = cross_val_score(lr, train[features], train[target], cv=10)

accuracy = mean(scores)

print("accuracy = ", accuracy)

print("scores = ", scores)
lr = LogisticRegression(solver='liblinear')

lr.fit(train[features], train[target])

holdout_predictions = lr.predict(test[features])

holdout_predictions
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived":holdout_predictions})

# Avoid adding index column

submission.to_csv("gender_submission.csv", index=False)
import os

print(os.listdir("."))