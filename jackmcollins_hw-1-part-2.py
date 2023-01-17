# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load the dataset into the datframe
diabetes = pd.read_csv("../input/diabetes.csv")

raw_data = diabetes
df = pd.DataFrame(raw_data, columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age', 'Outcome'])

df
# load and review the data
pd.set_option("display.max_rows",768)
df
#clean check for null values
df = df[(df[['Glucose','BloodPressure','Insulin', 'BMI','DiabetesPedigreeFunction','Age']] != 0).all(axis=1)]
df
# Inspect: Visualize correlation --> Goal: Red Running diagonally on xy plane
import matplotlib.pyplot as plt
data = pd.read_csv("../input/diabetes.csv")
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

plt.show()
#Clean: Delete the skin column
df = df.drop(['SkinThickness'], axis =1)
pd.set_option("display.max_columns",9)
print(df.describe(include = 'all'))
#printed the describe to make sure that all of columns that should not have 0 as a value removed 
#those rows becasue the min would show min = 0
#Change true to 1 and false to 0
#there is nothing that needs to be done here as all rows in outcome are already 1 or 0
#Check true false ratio

print("Total Number of True Occurences")
print(df.Outcome.sum())
print("Ratio of True Occurences")
print(df.Outcome.sum()/392)

print("===============================")

print("Total Number of False Occurences")
print(768 - df.Outcome.sum())
print("Ratio of False Occurences")
print(262/392)


#split the data into training data
data = sk.model_selection.train_test_split(df, test_size=.3)

training = data[0]
test = data[1]
training


#Delete all rows with 0 

print("I actually did this earlier when checking for null values.  I misread the requirments for that")
#id3

output = training.Outcome
trainActualOutcomes = training.Outcome
training = training.drop(columns = "Outcome")
actualTestOutcomes = test.Outcome

test = test.drop(columns = "Outcome")
dTree = sk.tree.DecisionTreeClassifier(criterion="entropy")
dTree = dTree.fit(training,output)
dot_data = tree.export_graphviz(dTree, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("Diabetes")
graph

#id3 accuracy
guessTest = dTree.predict(test).tolist()
guessTrain = dTree.predict(training).tolist()
outcome = actualTestOutcomes.tolist()
answerTrain = trainActualOutcomes.tolist()
testAcc=sk.metrics.accuracy_score(outcome, guessTest)
trainAcc = sk.metrics.accuracy_score(answerTrain, guessTrain)
testAcc,trainAcc
#c45
dTree45 = sk.tree.DecisionTreeClassifier()
dTree45 = dTree45.fit(training,output)
dot_data = tree.export_graphviz(dTree45, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("Diabetes")
graph

#c45 Accuracy
c45guess = dTree45.predict(test).tolist()
c45TrainGuess = dTree45.predict(training).tolist()
c45answer = actualTestOutcomes.tolist()
solution=sk.metrics.accuracy_score(c45answer,c45guess)
cTrain=sk.metrics.accuracy_score(answerTrain, c45TrainGuess)
solution,cTrain
cMatID3 = sk.metrics.confusion_matrix(outcome, guessTest)
cMatID3Train = sk.metrics.confusion_matrix(answerTrain, guessTrain)
cMat45 = sk.metrics.confusion_matrix(c45answer,c45guess)
cMat45Train = sk.metrics.confusion_matrix(answerTrain,c45TrainGuess)
classMatID3 = sk.metrics.classification_report(outcome, guessTest)
classMatID3Train =  sk.metrics.classification_report(answerTrain, guessTrain)
classMat45 = sk.metrics.classification_report(c45answer,c45guess)
classMat45Train  = sk.metrics.classification_report(answerTrain, c45TrainGuess)
print(cMatID3)
print(cMatID3Train)
print(cMat45)
print(cMat45Train)
print(classMatID3)
print(classMatID3Train)
print(classMat45)
print(classMat45Train)