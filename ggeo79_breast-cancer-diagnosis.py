import numpy as np 

import pandas as pd 

df = pd.read_csv('../input/data.csv')

df.head()
# Check the data size

df.shape
df.isnull().sum()
df.drop(df.columns[32], axis=1, inplace=True)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

#%matplotlib inline



sns.pairplot(df, vars=df.columns[2:5], hue='diagnosis', markers=['o', 's'], size=3, kind='reg')

plt.show()
# Use dummies in order to map the diagnosis labels to 0 and 1

df_new = pd.get_dummies(df)

df_new.head()
cols = list(df_new.columns)

fig, ax = plt.subplots(figsize=(26,20))

# We take values[:, 1:] in order to ommit the id's.

cm = np.corrcoef(df_new.values[:, 1:].T)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':8}, yticklabels=cols[1:], xticklabels=cols[1:])

plt.show()
# Let's get back to the original data

df.head()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.metrics import accuracy_score



# Take every line except from the columns with the id's and the labelled data (benigh, malignant)

X = df.values[:, 2:].astype(float)

# The labelled data

y = df.values[:, 1]
# Prepare the models

lr = LogisticRegression(C=1000.0, random_state=0)

svm_linear = SVC(kernel='linear', C=1.0, random_state=0, probability=True)

svm_nonlinear = SVC(kernel='rbf', C=10.0, gamma = 0.1, random_state=0, probability=True)

random_forest = RandomForestClassifier(criterion='entropy', random_state=0)

gaussian_nb = GaussianNB()
labels = ['Logistic Regression', 'SVC linear', 'SVC non linear', 'Random Forest', 'Naive Bayes']

models = [lr, svm_linear, svm_nonlinear, random_forest, gaussian_nb]

# Apply cross validation score

for model, label in zip(models, labels):

    scores = model_selection.cross_val_score(model, X, y, cv=5, scoring='accuracy')

    print('Mean: {0}\t Std: {1}\t Label: {2}'.format(scores.mean(), scores.std(), label))
voting = VotingClassifier(estimators=[('lr', lr), ('svm_linear', svm_linear), ('svm_nonlinear', svm_nonlinear), 

                                      ('random_forest', random_forest), ('gaussian_nb', gaussian_nb)],

                        voting = 'soft', weights=[1,1,1,1,1])



labels = ['Logistic Regression', 'SVC linear', 'SVC non linear', 'Random Forest', 'Naive Bayes', 'Voting']

models = [lr, svm_linear, svm_nonlinear, random_forest, gaussian_nb, voting]



for model, label in zip(models, labels):

    scores = model_selection.cross_val_score(model, X, y, cv=5, scoring='accuracy')

    print('Mean: {0}\t Std: {1}\t Label: {2}'.format(scores.mean(), scores.std(), label))
# Split data into training and test datasets.

# Playing with the test size, I found that 20% gives the best results

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# Standardize the features in order to have properties of standar normal distribution (mean = 0, std = 1)

std_sc = StandardScaler()

X_train = std_sc.fit_transform(X_train)

X_test = std_sc.fit_transform(X_test)
# Train our model

random_forest.fit(X_train, y_train)

# Get the feature importances attribute

importances = random_forest.feature_importances_

indices = np.argsort(importances)[::-1]



# Plot feature importances

fig, axes = plt.subplots(figsize=(14, 10))

plt.bar(range(X_train.shape[1]), importances[indices], align='center')

plt.title('Feature Importances')

plt.xticks(range(X_train.shape[1]), df.columns[2:], rotation=90)

plt.show()
# Predict class

y_pred = random_forest.predict(X_test)

print('Accuracy: {0}'.format(accuracy_score(y_test, y_pred)))