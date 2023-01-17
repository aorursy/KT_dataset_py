import numpy as np

import pandas as pd

import time
test  = pd.read_csv("./test.csv")

train = pd.read_csv("./train.csv")
print("Number of features in Train : ", train.shape[1])

print("Number of records  in Train : ",train.shape[0])

print("Number of features in Test  : ",test.shape[1])

print("Number of records  in Test  : ",test.shape[0])



trainData  = train.drop(['subject','Activity'] , axis=1).values

trainLabel = train.Activity.values



testData  = test.drop(['subject','Activity'] , axis=1).values

testLabel = test.Activity.values



print("Train Data shape  : ",trainData.shape)

print("Train Label shape : ",trainLabel.shape)

print("Test Data  shape  : ",testData.shape)

print("Test Label shape  : ",testLabel.shape)



print("Label examples: ")

print(np.unique(trainLabel))
from sklearn import preprocessing

from sklearn import utils



ltrain = preprocessing.LabelEncoder()

ltest = preprocessing.LabelEncoder()



trainLabel = ltrain.fit_transform(trainLabel)

testLabel  = ltest.fit_transform(testLabel)



print(np.unique(trainLabel))

print(np.unique(testLabel))

print("Train Label shape : ",trainLabel.shape)

print("Test Label shape  : ",testLabel.shape)

print(utils.multiclass.type_of_target(testLabel))
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV

from sklearn.utils import shuffle
t0 = time.clock()

# Create the RFE object and compute a cross-validated score.

svc = SVC(kernel="linear")

# The "accuracy" scoring is proportional to the number of correct

# classifications

rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(6),

              scoring='accuracy')

# Before training the data it is convenient to shuffle the data in training

np.random.seed(1)

print("Labels before Shuffle",testLabel[0:5])

testData,testLabel = shuffle(testData,testLabel)

trainData,trainLabel = shuffle(trainData,trainLabel)

print("Labels after Shuffle",testLabel[0:5])
# train and fit data in the model

rfecv.fit(trainData, trainLabel)



print("Optimal number of features : %d" % rfecv.n_features_)

print("Processing time sec ",time.clock() - t0)

# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
print('Accuracy of the SVM model on test data is ', rfecv.score(testData,testLabel) )

print('Ranking of features starting from the best estimated \n',rfecv.ranking_)

# if we mask the features to get only the best we get this

best_features = []

for ix,val in enumerate(rfecv.support_):

    if val==True:

        best_features.append(testData[:,ix])
from pandas.tools.plotting import scatter_matrix

visualize = pd.DataFrame(np.asarray(best_features).T)

print(visualize.shape)

scatter_matrix(visualize.iloc[:,0:5], alpha=0.2, figsize=(6, 6), diagonal='kde')