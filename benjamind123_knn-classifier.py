# K-NN Classifier



from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

import numpy as np



# Read train and test data into Python



train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')



# Classification Labels



Label = train.pop("label")



# Split train data into in train and in test



in_train = train[:38000]

in_test = train[38000:42000]



in_train_label = Label[:38000]

in_test_label = Label[38000:42000]



# K-NN with 1 neighbour



knn_1 = KNeighborsClassifier(n_neighbors = 5)

knn_1.fit(in_train, in_train_label)



# Classifying the test set



Predictions = knn_1.predict(in_test)



# Accuracy



np.mean(Predictions == in_test_label)



# Run K-NN on the whole train and test set

knn_2 = KNeighborsClassifier()

knn_2.fit(train, Label)



# Classify the test set



Predictions_all = knn_2.predict(test)



Predictions_df = pd.DataFrame(Predictions_all)

Index = pd.Series(range(1,28001))

ImageId = pd.DataFrame(Index)

Results = pd.concat([ImageId, Predictions_df], axis = 1)

Results.columns = ["ImageId", "Label"]

Results.to_csv("KNN_Prediction.csv", index = False)