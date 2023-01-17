import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import cross_val_predict as cvp
class Utils:
    def getNAStats(self, dataFrame):
        print(dataFrame.isna().sum())

class ScaleFeature():
    def perform(self, dataFrame, training=True):
        if(training):
            self.transformer = StandardScaler()
            self.transformer.fit(dataFrame)   
        npOutput = self.transformer.transform(dataFrame)
        return npOutput

class PreProcess():
    def __init__(self):
        self.scaleFeature = ScaleFeature()

    def perform(self, dataFrame, training=True):
        npOutput = self.scaleFeature.perform(dataFrame, training)
        return npOutput
    
class ValidateModels():
    def __init__(self, split):
        self.split = split
    
    def perform(self, model, attrSet, labelSet, classDist="uniform"):
        if(classDist == 'uniform'):
            scores = cvs(model, attrSet, labelSet, scoring="accuracy", cv=self.split)
            print("Accuracy Estimate: MEAN +/- STD = " + str(np.round(scores.mean(),3)) + " +/- " + str(np.round(scores.std(),3)))
        elif(classDist == 'skewed'):
            preds = cvp(model, attrSet, labelSet, cv=self.split)
            print("Confusion Matrix: ")
            print(metrics.confusion_matrix(labelSet, preds))
            print("Classification report: ")
            print(metrics.classification_report(labelSet, preds))

class Test():
    def perform(self, model, attrSet, labelSet=None):
        predictions = model.predict(attrSet)
        return predictions
### ==== ACTUAL IMPLEMENTATION ==== ###

trainSet = pd.read_csv("../input/digit-recognizer/train.csv", sep=",")
labels = ['label']
trainLabels = trainSet[labels].astype(int).astype('category')
trainAttrs = trainSet.drop(labels, axis=1)

#PreProcessing the Subsets
preProcess = PreProcess()
X_train = preProcess.perform(trainAttrs)
Y_train = trainLabels.values.ravel()
print(X_train.shape)
print(len(Y_train))
validateModels = ValidateModels(3)
et_model = ensemble.ExtraTreesClassifier(random_state=50, n_jobs=-1, n_estimators=500)
validateModels.perform(et_model, X_train, Y_train, 'skewed')
et_model.fit(X_train, Y_train)
testSet = pd.read_csv("../input/digit-recognizer/test.csv", sep=",")
testSet = testSet.reset_index()
testSet = testSet.rename({'index': 'ImageId'}, axis=1)
ids = ['ImageId']
testIds = testSet[ids].astype(int)
testIds.ImageId = testIds.ImageId + 1
testAttrs = testSet.drop(ids, axis=1)
X_test = preProcess.perform(testAttrs, training=False)
print(X_test.shape)

test = Test()
preds = test.perform(et_model, X_test)
testIds['Label'] = pd.Series(preds, dtype=int)
print(testIds.head())
#testIds.to_csv('../output/myPrediction.csv', index=False)