import sklearn.svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from statistics import mean

import warnings
warnings.filterwarnings('ignore')
diabetesDf = pd.read_csv('../input/diabetes.csv')
diabetesDf.head()
diabetesDf.isnull().sum()
sns.countplot(x = 'Outcome', data=diabetesDf)
plt.show()
fig = plt.gcf()
fig.set_size_inches(18.5, 22.5)
i = 1
for colName in diabetesDf.columns:
    if colName == 'Outcome': break
    plotX = np.array(diabetesDf[colName])
    plt.subplot(4,3,i)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    plt.hist(plotX,edgecolor='black')
    plt.title(colName)
    i+=1


plt.show()
posDiabetes = diabetesDf[diabetesDf['Outcome'] == 1]
fig = plt.gcf()
fig.set_size_inches(18.5, 22.5)
i = 1
for colName in posDiabetes.columns:
    if colName == 'Outcome': break
    plotX = np.array(posDiabetes[colName])
    plt.subplot(4,3,i)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    plt.hist(plotX, bins=15,edgecolor='black')
    plt.title(colName)
    i+=1


plt.show()
sns.pairplot(diabetesDf, hue='Outcome', diag_kind='kde')
plt.show()
outcomeY = diabetesDf['Outcome']
featuresX = diabetesDf[diabetesDf.columns[:8]]
trainSet, testSet = train_test_split(diabetesDf, test_size = 0.10, random_state = 0,
                                    stratify = diabetesDf['Outcome'])
trainSetX = trainSet[trainSet.columns[:8]]
trainSetY = trainSet['Outcome']

testSetX = testSet[testSet.columns[:8]]
testSetY = testSet['Outcome']
trainSetX.head()
accuracyDictInitial = {}
logisticReg = LogisticRegression()
logisticReg.fit(trainSetX, trainSetY)
prediction = logisticReg.predict(testSetX)
accuracyVal = metrics.accuracy_score(prediction, testSetY)
print('Accuracy by using Logistic Regression: ', accuracyVal)
accuracyDictInitial['Logistic Regression'] = accuracyVal
decisionTree = DecisionTreeClassifier()
decisionTree.fit(trainSetX, trainSetY)
prediction = decisionTree.predict(testSetX)
accuracyVal = metrics.accuracy_score(prediction, testSetY)
print('Accuracy by using Decision Trees: ', accuracyVal)
accuracyDictInitial['Decision Tree'] = accuracyVal
svmModel = svm.SVC(kernel='rbf')
svmModel.fit(trainSetX, trainSetY)
prediction = svmModel.predict(testSetX)
accuracyVal = metrics.accuracy_score(prediction, testSetY)
print('Accuracy by using rbf kernel SVM: ', accuracyVal)
accuracyDictInitial['Radial SVM'] = accuracyVal
svmModel = svm.SVC(kernel='linear')
svmModel.fit(trainSetX, trainSetY)
prediction = svmModel.predict(testSetX)
accuracyVal = metrics.accuracy_score(prediction, testSetY)
print('Accuracy by using rbf kernel SVM: ', accuracyVal)
accuracyDictInitial['Linear SVM'] = accuracyVal
accuracyVal = []
aVal = list(range(1, 8))
for a in aVal:
    knnModel = KNeighborsClassifier(n_neighbors=a)
    knnModel.fit(trainSetX, trainSetY)
    prediction = knnModel.predict(testSetX)
    accuracyVal.append(metrics.accuracy_score(prediction, testSetY))

plt.plot(aVal, accuracyVal)
print("Values of accuracies are: ", accuracyVal)
accuracyDictInitial['KNN'] = max(accuracyVal)
plt.show()
sns.heatmap(diabetesDf[diabetesDf.columns[:8]].corr(), annot=True, cmap='Blues')
fig = plt.gcf()
fig.set_size_inches(16.5, 15.5)
plt.show()
randomForestModel = RandomForestClassifier(n_estimators=40, random_state=0)
X = diabetesDf[diabetesDf.columns[:8]]
Y = diabetesDf['Outcome']
randomForestModel.fit(X, Y)
pd.Series(randomForestModel.feature_importances_, 
          index = X.columns).sort_values(ascending=False)
outcomeY = diabetesDf['Outcome']
featuresX = diabetesDf[diabetesDf.columns[:8]]
features = featuresX[['Glucose','BMI','Age','DiabetesPedigreeFunction']]
featuresStandardised = StandardScaler().fit_transform(features)
features = pd.DataFrame(featuresStandardised,
                        columns = [['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']])
features['Outcome'] = outcomeY
trainSet1, testSet1 = train_test_split(features, test_size = 0.10, 
                                      random_state = 0, stratify = features['Outcome'])

trainSetX1 = trainSet1[trainSet1.columns[:4]]
trainSetY1 = trainSet1['Outcome']

testSetX1 = testSet1[testSet1.columns[:4]]
testSetY1 = testSet1['Outcome']
accuracyDict = {}
logisticModel = LogisticRegression()
logisticModel.fit(trainSetX1, trainSetY1)
prediction = logisticModel.predict(testSetX1)
accuracyVal = metrics.accuracy_score(prediction, testSetY1)
print(accuracyVal)
accuracyDict['Logistic Regression'] = accuracyVal
svmModel = svm.SVC(kernel='rbf')
svmModel.fit(trainSetX1, trainSetY1)
prediction = svmModel.predict(testSetX1)
accuracyVal = metrics.accuracy_score(prediction, testSetY1)
print(accuracyVal)
accuracyDict['Radial SVM'] = accuracyVal
svmModel = svm.SVC(kernel='linear')
svmModel.fit(trainSetX1, trainSetY1)
prediction = svmModel.predict(testSetX1)
accuracyVal = metrics.accuracy_score(prediction, testSetY1)
print(accuracyVal)
accuracyDict['Linear SVM'] = accuracyVal
decisionTreeModel = DecisionTreeClassifier()
decisionTreeModel.fit(trainSetX1, trainSetY1)
prediction = decisionTreeModel.predict(testSetX1)
accuracyVal = metrics.accuracy_score(prediction, testSetY1)
print(accuracyVal)
accuracyDict['Decision Tree'] = accuracyVal
accuracyValList = []
aVal = list(range(1, 8))
for a in aVal:
    knnModel = KNeighborsClassifier(n_neighbors=a)
    knnModel.fit(trainSetX1, trainSetY1)
    prediction = knnModel.predict(testSetX1)
    accuracyValList.append(metrics.accuracy_score(prediction, testSetY1))
accuracyDict['KNN'] = max(accuracyValList)
plt.plot(aVal, accuracyValList)
print("Values of accuracies are: ", accuracyValList)
plt.show()
accuracyKeys = [keys for keys in accuracyDict]
accuracyValues = [accuracyDict[keys] for keys in accuracyDict]

accuracyKeysInit = [keys for keys in accuracyDictInitial]
accuracyValuesInit = [accuracyDictInitial[keys] for keys in accuracyDictInitial]
plt.subplot(1,2,1)
plt.bar(accuracyKeys, accuracyValues, edgecolor = 'black')

plt.subplot(1,2,2)
plt.bar(accuracyKeysInit, accuracyValuesInit, color = 'red', edgecolor = 'black')

fig = plt.gcf()
fig.set_size_inches(16.5, 5.5)
plt.show()