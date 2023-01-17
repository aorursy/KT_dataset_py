import pandas as reader
import csv
import numpy as number
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Reading training data set
train = reader.read_csv("../input/digit-recognizer/train.csv")

# Reading Test Data Set
test = reader.read_csv("../input/digit-recognizer/test.csv")
# Setting inputs
inputData = train.drop(columns=['label'])
# Setting output
result = train['label'].values

# spliting data for training and testing
# size means n% of dataset will be used for training
inputTrain, inputTest, resultTrain, resultTest = train_test_split(inputData, result, test_size=0.2, random_state=1,
                                                                  stratify=result)

# ********* HypterTunning for parameter K ****************

# ********* I have already run it and found value of k which is [1] *********
# ********* putting them in comments as it will take a lot of time  *********

# knn_K = KNeighborsClassifier()
# k_range = number.arange(1, 10)
# knn_hypterTunnig = GridSearchCV(knn_K,{'n_neighbors': k_range}, cv=3)
# knn_hypterTunnig.fit(inputData, result)
# print(knn_hypterTunnig.best_params_)
# print(knn_hypterTunnig.best_score_)

knnClassifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
knnClassifier.fit(inputData, result)
res = [[]]
res.clear()
imageid =1
print("Classification has started")
prediction = knnClassifier.predict(test)
print(prediction)

print("Printing the output")

for i in prediction:
    res.append([imageid,i])
    imageid=imageid+1

with open('KNNoutput.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(res)

