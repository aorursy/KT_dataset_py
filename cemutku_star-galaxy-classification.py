import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
data.head()
data.info()
data.columns
data.describe()
# We don't need objid, specobjid, rerun for classification
data.drop(["objid", "specobjid", "rerun"], axis = 1, inplace = True)
# QSO data deleting for binary classification. We just need star and galacy classes
data = data[data["class"] != "QSO"]
sns.countplot(x= "class", data = data)
data["class"].value_counts()
sns.pairplot(data.loc[:,["u", "g", "r", "i", "z", "class"]], hue = "class")
plt.show()
# Galaxy = 1 and Star = 0
data['class_binary'] = [1 if i == 'GALAXY' else 0 for i in data.loc[:,'class']]
# Convert STAR and GALAXY classes to int. For binary classification
data["class"] = [1 if each == "GALAXY" else 0 for each in data["class"]] 
# After converting operation. We call Star as 0 and Galaxy as 1
# data after preparation - formatting operations
data.head()
# value selection and normalization
y = data["class"].values
x_data = data.drop(["class"], axis = 1)
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
# after normalization
x.head()
# data separation for train - test operations
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# Dictionary for score results of classification models
algorithmPerfomanceDict = {}
#algorithmPerfomanceDict = {'ClassificationModel': 1, 'Accuracy': 2}
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
logisticRegressionScore = lr.score(x_test, y_test)
print("Score of Logistic Regression : {0}".format(logisticRegressionScore))
algorithmPerfomanceDict['LogisticRegression'] = logisticRegressionScore
# ROC Curve with logistic regression
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report

x,y = data.loc[:,(data.columns != 'class') & (data.columns != 'class_binary')], data.loc[:,'class_binary']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred_prob = logreg.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()
data.drop(["class_binary"], axis = 1, inplace = True)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1) # Number of neighbors to consider.
knn.fit(x_train, y_train)
knnScore = knn.score(x_test, y_test)
print("Score of KNN Regression : {0}".format(knnScore))
algorithmPerfomanceDict['KNeighborsClassifier'] = knnScore
#Lets find best K value
scoreList = []
for each in range(1, 20):
    optimumKnn = KNeighborsClassifier(n_neighbors = each)
    optimumKnn.fit(x_train, y_train)
    scoreList.append(optimumKnn.score(x_test, y_test))
    
plt.plot(range(1, 20), scoreList)
plt.xlabel("K value")
plt.ylabel("Score - Accuracy")
plt.show();
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv = 3)
knn_cv.fit(x, y)
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
k = 5
cv_result = cross_val_score(reg, x, y, cv = k) # uses R^2 as score 
print('CV Scores : ',cv_result)
print('CV Average Score : ',np.sum(cv_result) / k)
# According to plot best value for KNN algorithm is 1. It has highest accuracy percentage.
from sklearn.svm import SVC
svm = SVC(random_state = 42)
svm.fit(x_train, y_train)
svmScore = svm.score(x_test, y_test)
print("Accuracy of Support Vector Machine is : ", svmScore)
algorithmPerfomanceDict['SVM'] = svmScore
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
nb.score(x_test, y_test)
naiveBayesScore = nb.score(x_test, y_test)
print("Accuracy of Naive Bayes Classifier is : ", naiveBayesScore)
algorithmPerfomanceDict['NaiveBayesClassifier'] = naiveBayesScore
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
decisionTreeScore = dt.score(x_test, y_test)
print("Accuracy of Decision Tree Classifier is : ", decisionTreeScore)
algorithmPerfomanceDict['DecisionTreeClassifier'] = decisionTreeScore
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 42) #Number of trees in forest.
rf.fit(x_train, y_train)
randomForestScore = rf.score(x_test, y_test)
print("Accuracy of Random Forest Classifier is : ", randomForestScore)
algorithmPerfomanceDict['RandomForestClassifier'] = randomForestScore
algorithmPerfomanceDict
comparisonData = pd.DataFrame.from_dict(algorithmPerfomanceDict, orient = 'index', columns = ["Accuracy"])
comparisonData.head(10)
plt.figure(figsize = (20, 7))
sns.barplot(x = comparisonData.index, y = comparisonData.Accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Classification Model')
plt.title('Accuracy Values of Classification Models', color = 'blue', fontsize = 15)
plt.show()
from sklearn.metrics import confusion_matrix
y_pred = rf.predict(x_test)
y_actual = y_test
cm = confusion_matrix(y_actual, y_pred)
f, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(cm, annot = True, linewidths = 0.5, linecolor = "red", fmt = ".0f", ax = ax)
plt.xlabel("y_pred -> STAR = 0, GALAXY = 1")
plt.ylabel("y_actual -> STAR = 0, GALAXY = 1")
plt.show()
from sklearn.metrics import classification_report
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test, y_pred))