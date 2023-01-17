# Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import model_selection
# Input the data

train_raw = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Parameters

CROSSV_SIZE_PROP = 0.20



# Model parameters

params = {

    'RandomForestClassifier': {

        'n_estimators': 100

    }

}
# Split the raw training data into train and crossv

train, crossv = model_selection.train_test_split(train_raw, test_size=CROSSV_SIZE_PROP)
# Split the train, crossv datasets into X (samples), y (labels)

def split(dataset):

    return dataset.ix[:, 1:], dataset.label



train_X, train_y = split(train)

crossv_X, crossv_y = split(crossv) 
# Pick an estimator

# Refer to sklearn's page: http://scikit-learn.org/stable/tutorial/machine_learning_map/

# Try a random forest classifier

from sklearn import ensemble



clf = ensemble.RandomForestClassifier(**params['RandomForestClassifier'])

clf = clf.fit(train_X, train_y)
clf.get_params()
# Evaluate the model performance

# NOTE:

# "The evaluation metric for this contest is the categorization accuracy, 

# or the proportion of test images that are correctly classified. 

# For example, a categorization accuracy of 0.97 indicates that you have correctly classified all 

# but 3% of the images."



# Scoring reference:

# http://scikit-learn.org/stable/modules/model_evaluation.html

from sklearn import model_selection



score = model_selection.cross_val_score(clf, crossv_X, crossv_y, scoring='accuracy')

print('Accuracy score (crossv): {}'.format(score.mean()))
# Try it out on the test set

results = clf.predict(test)
# Write results to CSV file

import csv



output_data = [(x + 1, y) for x, y in enumerate(results)]



with open('output.csv', 'wt') as fp:

    writer = csv.writer(fp, delimiter=',')

    writer.writerow(['ImageId','Label'])

    for row in output_data:

        writer.writerow(row)