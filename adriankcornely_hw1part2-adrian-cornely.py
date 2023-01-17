# Readme: Run all modules in order
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataframe1 = pd.read_csv("../input/diabetes.csv")
dataframe1
#correlation matrix
correlation = dataframe1.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()
dataframe1 = dataframe1.drop(columns = "SkinThickness")
dataframe1


#what percentage of total data points are trues
trues = dataframe1["Outcome"] == 1
dataframe1Trues = dataframe1[trues]
total = dataframe1.shape[0]
totalTrues = dataframe1Trues.shape[0]
totalTrues/total

#remove data points with garbage data in them
dataframe1 = dataframe1[(dataframe1[['Glucose','BloodPressure','Insulin', 'BMI','DiabetesPedigreeFunction','Age']] != 0).all(axis=1)]
#split data 70/30
data = sklearn.model_selection.train_test_split(dataframe1,test_size = .3)
testData = data[1]
trainingData = data[0]
#ID3 run

outcomes = trainingData.Outcome
trainingOutcomesActual = trainingData.Outcome
trainingData = trainingData.drop(columns = "Outcome")
testOutcomesActual = testData.Outcome

testData = testData.drop(columns = "Outcome")
decisionTree = sklearn.tree.DecisionTreeClassifier(criterion="entropy")
decisionTree = decisionTree.fit(trainingData,outcomes)
dot_data = tree.export_graphviz(decisionTree, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("Diabetes")
graph




#ID3 correct percentage
guessesTest = decisionTree.predict(testData).tolist()
guessesTraining = decisionTree.predict(trainingData).tolist()
answers = testOutcomesActual.tolist()
answersTraining = trainingOutcomesActual.tolist()
testAccuracy=sklearn.metrics.accuracy_score(answers, guessesTest)
trainingAccuracy = sklearn.metrics.accuracy_score(answersTraining, guessesTraining)
testAccuracy,trainingAccuracy
decisionTreeC45 = sklearn.tree.DecisionTreeClassifier()
decisionTreeC45 = decisionTreeC45.fit(trainingData,outcomes)
dot_data = tree.export_graphviz(decisionTreeC45, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("Diabetes")
graph

#C4.5 correct percentage
guessesC45 = decisionTreeC45.predict(testData).tolist()
guessesC45Training = decisionTreeC45.predict(trainingData).tolist()
answersC45 = testOutcomesActual.tolist()
correct=sklearn.metrics.accuracy_score(answersC45,guessesC45)
correctTraining=sklearn.metrics.accuracy_score(answersTraining, guessesC45Training)
correct,correctTraining


confusionMatrixID3Test = sklearn.metrics.confusion_matrix(answers, guessesTest)
confusionMatrixID3Training = sklearn.metrics.confusion_matrix(answersTraining, guessesTraining)
confusionMatrixC45Test = sklearn.metrics.confusion_matrix(answersC45,guessesC45)
confusionMatrixC45Training = sklearn.metrics.confusion_matrix(answersTraining,guessesC45Training)
classificationMatrixID3Test = sklearn.metrics.classification_report(answers, guessesTest)
classificationMatrixID3Training =  sklearn.metrics.classification_report(answersTraining, guessesTraining)
classificationMatrixC45Test = sklearn.metrics.classification_report(answersC45,guessesC45)
classificationMatrixC45Training  = sklearn.metrics.classification_report(answersTraining, guessesC45Training)
print(confusionMatrixID3Test)
print(confusionMatrixID3Training)
print(confusionMatrixC45Test)
print(confusionMatrixC45Training)
print(classificationMatrixID3Test)
print(classificationMatrixID3Training)
print(classificationMatrixC45Test)
print(classificationMatrixC45Training)
