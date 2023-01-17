import numpy as np

import pandas as pd

import os

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import itertools



# Importing Machine learning models library used for classification

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, confusion_matrix



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

df.head()
# Get the shape of dataset.

df.shape
# Get dataset information.

df.info()
# Identify columns with null values.

df.isnull().sum()
# Describe the dataset by basic statistical calculations.

df.describe()
# Feature correlation.

plt.figure(figsize=(14, 14))

plt.title("Credit Card Transactions features correlation plot (Pearson)")

corr = df.corr()

sns.heatmap(

    corr,

    xticklabels=corr.columns,

    yticklabels=corr.columns,

    linewidths=0.1,

    cmap="Blues",

)

plt.show()
LABELS = ["Legitimate (0)", "Fraud (1)"]

count_classes = pd.value_counts(df["Class"], sort=True)

count_classes.plot(kind="bar", rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")
Fraud = df[df["Class"] == 1]

Legitimate = df[df["Class"] == 0]

print("Number of Legitimate entries = {}".format(len(Legitimate)))

print("Number of Fraud entries = {}".format(len(Fraud)))
X = df.drop("Class", axis=1)

Y = df["Class"]



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



# Convert data to numpy arrays to be fed into algorithms.

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
num_neighbours = 7

n_est = 100
classifiers = {

    "Logisitic Regression": LogisticRegression(),

    "K Nearest Neighbors": KNeighborsClassifier(n_neighbors=num_neighbours),

    "Gaussian Naive Bays": GaussianNB(),

    "Decision Tree Classifier": DecisionTreeClassifier(),

}
Acc = {}
# Train classifiers.

print("Cross-Validation Scores for classifiers:-")

for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    cv_score = cross_val_score(classifier, X_train, y_train, cv=5)

    pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    Acc[key] = accuracy

    print("{}: {}".format(key, round(cv_score.mean() * 100.0, 2)))
for model, acc in Acc.items():

    print("Accuracy for {} = {}".format(model, acc * 100))
# Train a Decision Tree Classifier since it has the best accuracy on the dataset.

model = DecisionTreeClassifier()

model = model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Confusion matrix")

cm = confusion_matrix(y_test, predictions)

print(cm)