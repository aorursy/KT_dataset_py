import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import matplotlib.pyplot as plt


%matplotlib inline

# get training & test csv files as a DataFrame
train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )



test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


# preview the data
train_df.head()
test_df.head()

# Encode Embarked using onehot encoding

test_df=test_df.join(pd.get_dummies(test_df.Embarked, prefix='Emb'))
train_df=train_df.join(pd.get_dummies(train_df.Embarked, prefix='Emb'))


test_df=test_df.join(pd.get_dummies(test_df.Sex, prefix='Sex'))
train_df=train_df.join(pd.get_dummies(train_df.Sex, prefix='Sex'))

# Lose the data we aren't interested in
test_df=test_df.drop(['Embarked','Sex','Name','Ticket','Cabin'], axis=1)
train_df=train_df.drop(['Embarked','Sex','Name','Ticket','Cabin'], axis=1)
train_df.describe()
# Fill in missing Age & Fair data in train, test datasets with median age of passengers found in training data set

# IMPROVEMENT OPP -- potential improvement would be to fill in median ages by 
# class and gender categories (sample code in Kaggle tutorial)
# or drop the rows altogether
median_age = train_df.Age.median(axis=0)
train_df.Age = train_df.Age.fillna(median_age)
test_df.Age = test_df.Age.fillna(median_age)

median_fare = train_df.Fare.median(axis=0)
train_df.Fare = train_df.Fare.fillna(median_fare)
test_df.Fare = test_df.Fare.fillna(median_fare)
y_train_orig = train_df.iloc[:,1].values


# Keep Class, Sex, Age, Relationships, Fare, Origin in model for now
X_train_orig = train_df.iloc[:,[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
X_test_orig = test_df.iloc[:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]


train_col = X_train_orig.columns
test_col = X_test_orig.columns
# Normalize data fields
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# sc.fit(X_train_orig)

X_test_orig = pd.DataFrame(sc.fit_transform(X_test_orig))
X_train_orig = pd.DataFrame(sc.fit_transform(X_train_orig))
X_test_orig.columns = test_col
X_train_orig.columns = train_col
# Find the features that really matter in data set using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
feat_labels = X_train_orig.columns
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train_orig, y_train_orig)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
importances

indices
# identify the list of top features

for f in range(X_train_orig.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
# Use only top features
X_train_orig = forest.transform(X_train_orig, threshold=.05)
X_test_orig = forest.transform(X_test_orig, threshold=.05)
X_train_orig
# Perform pre-processing to determine optimal data set size and tune model parameters
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=100.0, gamma=0.1, random_state=0)

# Determine optimal training data set size using learning curve methods
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=svm, X=X_train_orig, y=y_train_orig, 
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 0.9])
plt.show()
# Determine optimal parameters for machine learning model
from sklearn.learning_curve import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=svm, X=X_train_orig, y=y_train_orig, param_name='C',
                                            param_range=param_range, cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xscale('log')
plt.grid()
plt.xlabel('Parameter')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 0.85])
plt.show()
# Mathmatically determine optimal parameters
from sklearn.grid_search import GridSearchCV
param_grid = [{'C': param_range,
              'kernel': ['linear']},
             {'C': param_range,
             'gamma': param_range,
             'kernel': ['rbf']}]
gs = GridSearchCV(estimator=svm,
                 param_grid=param_grid,
                 scoring='accuracy',
                 cv=10, n_jobs=1)
gs = gs.fit(X_train_orig, y_train_orig)
print(gs.best_score_)
print(gs.best_params_)
# in addition to the original data sets for training (train_orig)and testing (test_orig)
# split train_orig data into training and testing sets randomly so we can obtain a practice test set with outcomes
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_orig, y_train_orig, test_size=0.25, random_state=0)
# Call SVC from scikitlearn library to train weights and run it on segmented data used for testing first 
# to see how accurate we can be
svm = SVC(kernel='rbf', C=100.0, gamma=0.1, random_state=0)
svm.fit(X_train, y_train)

# call algo to predict using test data set
y_pred = svm.predict(X_test)
no_samples = len(y_test)
print('Misclassified samples: %d of %d' % ((y_test != y_pred).sum() , no_samples))

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# Determine number of true-positives, false-positives, true-negatives, false-negatives to see if model can be 
# optimized
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[i]):
        ax.text(x=j, y=i,
               s=confmat[i,j],
               va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
# Use k-fold cross validation scorer as a better way to predict how robust our model will be against test data
from sklearn.cross_validation  import cross_val_score
scores = cross_val_score(estimator=svm, X=X_train, y=y_train, cv=10, n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
svmorig = SVC(kernel='rbf', C=100.0, gamma=0.1, random_state=0)
svmorig.fit(X_train_orig, y_train_orig)

# call algo to predict using test data set
y_pred_orig = svm.predict(X_test_orig)
y_pred_orig.sum()
output = test_df.PassengerId
output = pd.DataFrame(output)
# len(output)
predict = pd.DataFrame(y_pred_orig)
output = output.join(predict)
output.columns = ['PassengerId', 'Survived']
output
output.to_csv("../input/output.csv", index=False)