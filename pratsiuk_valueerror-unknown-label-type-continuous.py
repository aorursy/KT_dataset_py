# Simple example for beginers how to adress ""
import numpy as np
from sklearn                        import metrics, svm
from sklearn.linear_model           import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
training_data_X    = np.array([ [1.2, 6.7, 2.7],  [2.3, 4.6, 2.2],  [0.3, 3.9, 0.8],  [2.1, 1.3, 4.3]  ])
training_scores_Y  = np.array( [1.4, 9.2, 2.5, 2.2] )
prediction_data_test  = np.array([ [1.5, 3.4, 2.2],  [7.6, 7.2, 0.2] ])
clf = LogisticRegression()
clf.fit(training_data_X, training_scores_Y)
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(training_scores_Y)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(training_scores_Y))
print(utils.multiclass.type_of_target(training_scores_Y.astype('int')))
print(utils.multiclass.type_of_target(training_scores_encoded))
clf = LogisticRegression()
clf.fit(training_data_X, training_scores_encoded)
print("LogisticRegression")
print(clf.predict(prediction_data_test))