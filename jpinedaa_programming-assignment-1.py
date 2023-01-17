# TODO: There are two validation methods you have to implement ( resubstitution and k-fold validation ).
#    For each method, you have to calculate:
#        (1) confusion matrix
#        (2) precision and recall for each label
#        (3) total accuracy
# Write a todolist from the lil comments in the code :)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import io
import requests
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.metrics as metrics
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# See if we can use the dataset provided, this code worked for me
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
#pd_train = pd.read_csv(url, names=["sepal length", "sepal width", "petal length", "petal width", "class"])
pd_train = pd.read_csv("../input/iris.data.txt", names=["sepal length", "sepal width", "petal length", "petal width", "class"])
pd_train.head()
pd_train.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pd_train.hist()
scatter_matrix(pd_train)
plt.show()
sepal_length_avg = pd_train["sepal length"].mean()
print("sepal length average:",sepal_length_avg)
sepal_width_avg = pd_train["sepal width"].mean()
print("sepal width average:",sepal_width_avg)
petal_length_avg = pd_train["petal length"].mean()
print("petal length average:",petal_length_avg)
petal_width_avg = pd_train["petal width"].mean()
print("petal width average:",petal_width_avg)

sepal_length_std = pd_train["sepal length"].std()
print("sepal length standard deviation:",sepal_length_std)
sepal_width_std = pd_train["sepal width"].std()
print("sepal width standard deviation:",sepal_width_std)
petal_length_std = pd_train["petal length"].std()
print("petal length standard deviation:",petal_length_std)
petal_width_std = pd_train["petal width"].std()
print("petal width standard deviation:",petal_width_std)

sepal_length_vc = pd_train["sepal length"].value_counts()
sepal_width_vc = pd_train["sepal width"].value_counts()
petal_length_vc = pd_train["petal length"].value_counts()
petal_width_vc = pd_train["petal width"].value_counts()

value_counts = pd.DataFrame(data = {"sepal length": sepal_length_vc, "sepal width": sepal_width_vc, "petal length": petal_length_vc, "petal width": petal_width_vc})
print(value_counts)
pd_features = pd_train.iloc[:,:4]
pd_target = pd_train.iloc[:,4]
print(pd_features.shape)
print(pd_target.shape)

X_train, X_test, y_train, y_test = train_test_split(pd_features, pd_target, test_size = 0.15)
print(pd_target)

model = tree.DecisionTreeClassifier()
print("Training Decision tree model...")
model.fit(X_train,y_train)
score = model.score(X_test,y_test)
print("Training finished model score: ", score)
random_forest = []
random_forest.append(tree.DecisionTreeClassifier())
random_forest[0].fit(X_train[["sepal length","sepal width"]],y_train)
print(random_forest[0].score(X_test[["sepal length","sepal width"]],y_test))
random_forest.append(tree.DecisionTreeClassifier())
random_forest[1].fit(X_train[["petal length","petal width"]],y_train)
print(random_forest[1].score(X_test[["petal length","petal width"]],y_test))
random_forest.append(tree.DecisionTreeClassifier())
random_forest[2].fit(X_train[["sepal length","petal length"]],y_train)
print(random_forest[2].score(X_test[["sepal length","petal length"]],y_test))
random_forest.append(tree.DecisionTreeClassifier())
random_forest[3].fit(X_train[["sepal width","petal width"]],y_train)
print(random_forest[3].score(X_test[["sepal width","petal width"]],y_test))
random_forest.append(tree.DecisionTreeClassifier())
random_forest[4].fit(X_train,y_train)
print(random_forest[4].score(X_test,y_test))
def RandomForest(X):
    predictions = []
    predictions.append(random_forest[0].predict(X[["sepal length","sepal width"]]))
    predictions.append(random_forest[1].predict(X[["petal length","petal width"]]))
    predictions.append(random_forest[2].predict(X[["sepal length","petal length"]]))
    predictions.append(random_forest[3].predict(X[["sepal width","petal width"]]))
    predictions.append(random_forest[4].predict(X))                   
    p= stats.mode(predictions)
    return p[0][0]

predicted = RandomForest(X_test)
print(predicted)
correct = 0
pred_length = predicted.size
#print(y_test.iloc[1])
#print(predicted[0,1])
for i in range(pred_length):
    if (predicted[i]==y_test.iloc[i]):
        correct += 1
accuracy = correct/pred_length
accuracy = accuracy*100

print(accuracy)

categories = pd_train['class'].unique()
print(categories)
conf_matrix=confusion_matrix(y_test,predicted,categories)
print(conf_matrix)
random_forest[0].fit(pd_features[["sepal length","sepal width"]],pd_target)
random_forest[1].fit(pd_features[["petal length","petal width"]],pd_target)
random_forest[2].fit(pd_features[["sepal length","petal length"]],pd_target)
random_forest[3].fit(pd_features[["sepal width","petal width"]],pd_target)
random_forest[4].fit(pd_features,pd_target)
# Maybe we take a look at this :)
predicted = RandomForest(pd_features)
correct = 0
pred_length = predicted.size
#print(y_test.iloc[1])
#print(predicted[0,1])
for i in range(pred_length):
    if (predicted[i]==pd_target.iloc[i]):
        correct = correct + 1
accuracy = correct/pred_length
accuracy = accuracy*100

print(accuracy)
#train = [5,6,2,3]
#pdlol = pd_features.ix[train,["sepal length","sepal width"]]
#print(pdlol)

acc = []

kf = KFold(n_splits = 10)
for train, test in kf.split(pd_features):
    random_forest[0].fit(pd_features[["sepal length","sepal width"]],pd_target)
    random_forest[1].fit(pd_features[["petal length","petal width"]],pd_target)
    random_forest[2].fit(pd_features[["sepal length","petal length"]],pd_target)
    random_forest[3].fit(pd_features[["sepal width","petal width"]],pd_target)
    random_forest[4].fit(pd_features,pd_target)

    predicted = RandomForest(pd_features.iloc[test])
    correct = 0
    pred_length = predicted.size
    y_test = pd_target[test]
    #print(y_test.iloc[1])
    #print(predicted[0,1])
    for i in range(pred_length):
        if (predicted[i]==y_test.iloc[i]):
            correct = correct + 1
    accuracy = correct/pred_length
    accuracy = accuracy*100
    acc.append(accuracy)

    print(accuracy)
    
print("mean: ",np.mean(acc))
# Aleksas code for validation
# Resubstitution and k-fold validation
# For each method, you have to calculate:
#        (1) confusion matrix
#        (2) precision and recall for each label
#        (3) total accuracy
''''''
# Resubstitution
# Resubstitution: In case, the whole data is used for training the model and the error rate is evaluated based on outcome vs actual value from the same training data set, this error is called as the resubstitution error. This technique is called as resubstitution validation technique.
predicted = RandomForest(pd_features)
#print(predicted)
correct = 0
pred_length = predicted.size
for i in range(pred_length):
    if (predicted[i]==pd_target.iloc[i]):
        correct = correct + 1
accuracy = correct/pred_length
accuracy = accuracy*100

print(accuracy)

# matrix
matrix = confusion_matrix(pd_target, predicted, categories)
print(matrix)
# precision and recall for each label
# precision (identified correctly) / (identified)
# recall (identified correctly) / (needed to be indentified)
precisions = metrics.precision_score(pd_target, predicted, average=None)
recalls = metrics.recall_score(pd_target, predicted, average=None)
for i in range(0,3):
#    precisions.append(matrix[i][i] / sum(matrix[:, i]))
#    recalls.append(matrix[i][i] / sum(matrix[i]))
    print(categories[i] + " precision = " + str(precisions[i]))
    print(categories[i] + " recall = " + str(recalls[i]))

# total accuracy
accuracy = metrics.accuracy_score(pd_target, predicted)
print(accuracy)
print("<<<>>>")
# K-fold validation
matrix = []
total_accuracy = []
precisions = []
recalls = []
kf = KFold(n_splits = 10)
predicted2 = []
target2 = []
for train, test in kf.split(pd_features):
    random_forest[0].fit(pd_features.iloc[train][["sepal length","sepal width"]],pd_target.iloc[train])
    random_forest[1].fit(pd_features.iloc[train][["petal length","petal width"]],pd_target.iloc[train])
    random_forest[2].fit(pd_features.iloc[train][["sepal length","petal length"]],pd_target.iloc[train])
    random_forest[3].fit(pd_features.iloc[train][["sepal width","petal width"]],pd_target.iloc[train])
    random_forest[4].fit(pd_features.iloc[train],pd_target.iloc[train])

    predicted = RandomForest(pd_features.iloc[test])
    predicted2.extend(predicted)
    
    matrix.append(confusion_matrix(pd_target.iloc[test], predicted, categories))
    precisions.append(metrics.precision_score(pd_target.iloc[test], predicted, average=None))
    recalls.append(metrics.recall_score(pd_target.iloc[test], predicted, average=None))
    
    correct = 0
    pred_length = predicted.size
    y_test = pd_target[test]
    target2.extend(y_test)
    for i in range(pred_length):
        if (predicted[i]==y_test.iloc[i]):
            correct = correct + 1
    accuracy = correct/pred_length
    accuracy = accuracy*100
    total_accuracy.append(accuracy)

    #print(total_accuracy)
    
print("mean: ",np.mean(total_accuracy))

# matrix
matrix = confusion_matrix(target2, predicted2, categories)
print(matrix)
# precision and recall for each label
# precision (identified correctly) / (identified)
# recall (identified correctly) / (needed to be indentified)
precisions = metrics.precision_score(target2, predicted2, average=None)
recalls = metrics.recall_score(target2, predicted2, average=None)
for i in range(0,3):
#    precisions.append(matrix[i][i] / sum(matrix[:, i]))
#    recalls.append(matrix[i][i] / sum(matrix[i]))
    print(categories[i] + " precision = " + str(precisions[i]))
    print(categories[i] + " recall = " + str(recalls[i]))
