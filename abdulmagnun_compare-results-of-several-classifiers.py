

#! -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics



sns.set_style("darkgrid")



% matplotlib inline



data = pd.read_csv('../input/oasis_longitudinal.csv')



# Fill missing fields

data = data.fillna(method='ffill')



# Drop unnecessary columns

for x in ['Subject ID', 'MRI ID', 'Visit']:

    data.drop(x, axis=1, inplace=True)



# Encode columns into integers

for column in data.columns:

    le = LabelEncoder()

    data[column] = le.fit_transform(data[column])
# Draw heatmap to show correlation

sns.heatmap(data.corr(), annot=True)
# Split training and test data

train, test = train_test_split(data, test_size=0.3)



# Load columns

train_X = train[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]

train_y = train.CDR

test_X = test[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]

test_y = test.CDR



# Pick a few classifiers, source: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

classifiers = [

    KNeighborsClassifier(2),

    KNeighborsClassifier(3),

    KNeighborsClassifier(6),

    KNeighborsClassifier(7),

    SVC(),

    SVC(kernel="linear", C=0.025),

    SVC(kernel="linear", C=0.01),

    SVC(kernel="linear", C=2),

    SVC(gamma=2, C=1),

    # too slow GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(alpha=1),

    AdaBoostClassifier(),

    GaussianNB(),

    LogisticRegression()]



# Print accuracy

data = []

labels = []



for i, model in enumerate(classifiers):

    model = model

    model.fit(train_X, train_y)

    prediction = model.predict(test_X)

    labels.append(str(model).split('(')[0])

    data.append([metrics.accuracy_score(prediction, test_y)])

plt.plot([i for i, e in enumerate(data)], data, 'ro'); plt.xticks([i for i, e in enumerate(labels)], [l[0:3] for l in labels])
