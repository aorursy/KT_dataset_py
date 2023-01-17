
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
iris = pd.read_csv("../input/Iris.csv")
iris.shape
iris.head()
iris.corr()
x = iris.drop('Species',axis=1)
y = iris['Species']
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.40, random_state = 42)
logClassifier = linear_model.LogisticRegression(C=150, random_state=11, solver='lbfgs', multi_class='multinomial', max_iter=5000)
logClassifier.fit(xTrain, yTrain)
totalVerdier = len(xTest)
suksessVerdier = len(logClassifier.predict(xTest)[logClassifier.predict(xTest) == yTest])
print("The model predicted correctly in", (suksessVerdier/totalVerdier)*100, "% of the test cases.")