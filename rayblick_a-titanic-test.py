#Import modules

%matplotlib inline

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



# Classification modules

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn import cross_validation



# check for datasets

from subprocess import check_output

#print(check_output(["ls", "../../data"]).decode("utf8"))
# get csv files according to the output above

train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
# subset columns (note these do not include survivor column)

train_data = train_df.loc[:,['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]



# create labels for the training data

train_labels = train_df.loc[:,['Survived']] 
# convert gender to integers

train_data.Sex = preprocessing.LabelEncoder().fit_transform(train_data.Sex) 



# fill in missing values for Age by replacing NA to Mean 

train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
# create training data

train_subset = train_data[0:600]

train_sub_labels = train_labels[0:600]



# create test data 

test_subset = train_data[601:849]

test_sub_labels = train_labels[601:849]
# classification

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_subset, train_sub_labels['Survived'])

output = forest.predict(test_subset)
# Results

rf_crosstab = pd.crosstab(test_sub_labels['Survived'], 

                          output, rownames=['actual'],

                          colnames=['preds'], 

                          margins=True)



rf_accuracy = 100 * ((rf_crosstab[0][0] + rf_crosstab[1][1]) / rf_crosstab['All']['All'])



print('RF classification accuracy = {} %'.format(round(rf_accuracy)))
# classification

clf = SGDClassifier(alpha=0.001, loss="hinge",penalty='elasticnet', n_iter=100000)

clf.fit(train_subset, train_sub_labels['Survived'])

sgd_output = clf.predict(test_subset)
# Results

sgd_crosstab = pd.crosstab(test_sub_labels['Survived'], 

                           sgd_output, rownames=['actual'],

                           colnames=['preds'], 

                           margins=True)



sgd_accuracy = 100 * ((sgd_crosstab[0][0] + sgd_crosstab[1][1]) / sgd_crosstab['All']['All'])



print('SGD classification accuracy = {} %'.format(round(sgd_accuracy)))
# classification

clf = NearestCentroid()

clf.fit(train_subset, train_sub_labels['Survived'])

nn_output = clf.predict(test_subset)
# Results

nn_crosstab = pd.crosstab(test_sub_labels['Survived'], 

                          nn_output, 

                          rownames=['actual'],\

                          colnames=['preds'], 

                          margins=True)



nn_accuracy = 100 * ((nn_crosstab[0][0] + nn_crosstab[1][1]) / nn_crosstab['All']['All'])



print('NN classification accuracy = {} %'.format(round(nn_accuracy)))
clf = svm.SVC(kernel='linear', C=1).fit(train_subset, train_sub_labels['Survived'])

clf.score(test_subset, test_sub_labels['Survived'])
clf = svm.SVC(kernel='linear', C=1)

scores = cross_validation.cross_val_score(clf, train_data, train_labels['Survived'], cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
rfc = RandomForestClassifier(n_estimators = 1000)

scores = cross_validation.cross_val_score(rfc, train_data, train_labels['Survived'], cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# import, reduce and clean test data 

test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

test_data = test_df.loc[:,['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

test_data.Sex = preprocessing.LabelEncoder().fit_transform(test_data.Sex) 

test_data["Age"].fillna(test_data["Age"].mean(), inplace=True)

test_data["Fare"].fillna(test_data["Fare"].mean(), inplace=True) # additional process required for Fare
# Random Forest classification

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data, train_labels['Survived'])

rf_predict = forest.predict(test_data)
# Create PassengerID column (test passengerID's start at 892)

pID = [x for x in range(892,(892+len(rf_predict)),1)]

pID = pd.DataFrame(pID, columns=['PassengerID'])
# merge dataframes

rf_predict = pd.DataFrame(rf_predict, columns=['Survived'])

output = pID.merge(rf_predict, left_index=True, right_index=True)
# create csv for submission 

# output.to_csv('rf6.csv', index=False, encoding='utf-8')