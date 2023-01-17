# Importing all necessary libraries.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head()
df.info()
df.describe()
print(df.columns)
df = df.sample(frac=0.1, random_state = 1)

print(df.shape)

print(df.describe())
df.hist(figsize = (20, 20))

plt.show()
# Determine number of fraud cases in dataset



Fraud = df[df['Class'] == 1]

Valid = df[df['Class'] == 0]



outlier_fraction = len(Fraud)/float(len(Valid))

print(outlier_fraction)



print('Fraud Cases: {}'.format(len(df[df['Class'] == 1])))

print('Valid Transactions: {}'.format(len(df[df['Class'] == 0])))
# Correlation matrix

import seaborn as sns

corrmat = df.corr()

fig = plt.figure(figsize = (12, 9))



sns.heatmap(corrmat, vmax = .8, square = True)

plt.show()
# Get all the columns from the dataFrame.

columns = df.columns.tolist()



# Filter the columns to remove data we do not want.

columns = [c for c in columns if c not in ["Class"]]



# Store the variable we'll be predicting on.

target = "Class"



X = df[columns]

Y = df[target]



# Print shapes.

print(X.shape)

print(Y.shape)
from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor



# Defining random states.

state = 1



# Defining outlier detection tools to be compared.

classifiers = {

    "Isolation Forest": IsolationForest(max_samples=len(X),

                                        contamination=outlier_fraction,

                                        random_state=state),

    "Local Outlier Factor": LocalOutlierFactor(

        n_neighbors=20,

        contamination=outlier_fraction)}
# Fit the model

plt.figure(figsize=(9, 7))

n_outliers = len(Fraud)





for i, (clf_name, clf) in enumerate(classifiers.items()):

    

    # fit the data and tag outliers

    if clf_name == "Local Outlier Factor":

        y_pred = clf.fit_predict(X)

        scores_pred = clf.negative_outlier_factor_

    else:

        clf.fit(X)

        scores_pred = clf.decision_function(X)

        y_pred = clf.predict(X)

    

    # Reshape the prediction values to 0 for valid, 1 for fraud. 

    y_pred[y_pred == 1] = 0

    y_pred[y_pred == -1] = 1

    

    n_errors = (y_pred != Y).sum()

    

    # Run classification metrics

    print('{}: {}'.format(clf_name, n_errors))

    print(accuracy_score(Y, y_pred))

    print(classification_report(Y, y_pred))
