#read me : run sequentially block by block
# actually learn what these algorithms are doing 

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
#load the data 
df = pd.read_csv("../input/diabetes.csv")
df.describe()

#view the data 
pd.set_option("display.max_rows",768)
df
#vizualize correlation 
import matplotlib.pyplot as plt
import pandas
import numpy

correlations = df.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)


plt.show()
#drop the skin column 
df = df.drop(columns="SkinThickness")
df
trues = df['Outcome'] == 1
dftrues= df[trues]
total = df.shape[0]
totaltrues = dftrues.shape[0]
totaltrues/total
#clean the data of null values
df = df[(df[['Glucose','BloodPressure','Insulin', 'BMI','DiabetesPedigreeFunction','Age']] != 0).all(axis=1)]
df
#split
data =sklearn.model_selection.train_test_split(df,test_size =.3)
testingData = data[1]
learningData = data[0]

#id3 run
results = learningData.Outcome
learningresults = learningData.Outcome
learningData = learningData.drop(columns = 'Outcome')
testingResults = testingData.Outcome

testingData = testingData.drop(columns='Outcome')
decTree = sklearn.tree.DecisionTreeClassifier(criterion= 'entropy')
decTree = decTree.fit(learningData,results)
dotData = tree.export_graphviz(decTree, out_file=None)
graphs=graphviz.Source(dotData)
graphs.render("Diabetes")
graphs

predictionTest = decTree.predict(testingData).tolist()
predictionLearning = decTree.predict(learningData).tolist()
solutions = testingResults.tolist()
solutionsTraining = learningresults.tolist()
accuracy=sklearn.metrics.accuracy_score(solutions, predictionTest)
trainingAccuracy = sklearn.metrics.accuracy_score(solutionsTraining, predictionLearning)
accuracy,trainingAccuracy
#c45
decTreeC45 = sklearn.tree.DecisionTreeClassifier()
decTreeC45 = decTreeC45.fit(learningData,results)
dotData = tree.export_graphviz(decTreeC45, out_file=None)
graphs = graphviz.Source(dotData) 
graphs.render("Diabetes")
graphs
#C4.5 percentage
predictionsC45 = decTreeC45.predict(testingData).tolist()
predictionsC45Training = decTreeC45.predict(learningData).tolist()
solutionsC45 = testingResults.tolist()
answer=sklearn.metrics.accuracy_score(solutionsC45,predictionsC45)
answerTraining=sklearn.metrics.accuracy_score(solutionsTraining, predictionsC45Training)
answer,answerTraining
conMatrixTest = sklearn.metrics.confusion_matrix(solutions, predictionTest)
conMatrixTraining = sklearn.metrics.confusion_matrix(solutionsTraining, predictionLearning)
conMatrixc45 = sklearn.metrics.confusion_matrix(solutionsC45,predictionsC45)
conMatrixc45Training = sklearn.metrics.confusion_matrix(solutionsTraining,predictionsC45Training)
classMatrix = sklearn.metrics.classification_report(solutions, predictionTest)
classMatrixTraining =  sklearn.metrics.classification_report(solutionsTraining, predictionLearning)
classMatrixC45 = sklearn.metrics.classification_report(solutionsC45,predictionsC45)
classMatrixC45Training  = sklearn.metrics.classification_report(solutionsTraining, predictionsC45Training)
print(conMatrixTest)
print(conMatrixTraining)
print(conMatrixc45)
print(conMatrixc45Training)
print(classMatrix)
print(classMatrixTraining)
print(classMatrixC45)
print(classMatrixC45Training)
