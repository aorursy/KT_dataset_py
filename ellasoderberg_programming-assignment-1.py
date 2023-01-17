# Hello, I'm here :))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
pd_train = pd.read_csv("../input/iris.data.txt", names=["sepal length", "sepal width", "petal length", "petal width", "class"])
pd_train.head()
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

x_train, x_test, y_train, y_test = train_test_split(pd_features,pd_target,test_size = 0.15)


model = tree.DecisionTreeClassifier()
print("Training Decision tree model...")
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print("Training finished model score: ", score)
model1 = tree.DecisionTreeClassifier()
model1.fit(x_train[["sepal length","sepal width"]],y_train)
print(model1.score(x_test[["sepal length","sepal width"]],y_test))
model2 = tree.DecisionTreeClassifier()
model2.fit(x_train[["petal length","petal width"]],y_train)
print(model2.score(x_test[["petal length","petal width"]],y_test))
model3 = tree.DecisionTreeClassifier()
model3.fit(x_train[["sepal length","petal length"]],y_train)
print(model3.score(x_test[["sepal length","petal length"]],y_test))
model4 = tree.DecisionTreeClassifier()
model4.fit(x_train[["sepal width","petal width"]],y_train)
print(model4.score(x_test[["sepal width","petal width"]],y_test))
def RandomForest(X):
    predictions = [model1.predict(X[["sepal length","sepal width"]]),model2.predict(X[["petal length","petal width"]]),model3.predict(X[["sepal length","petal length"]]),model4.predict(X[["sepal width","petal width"]])]
    p= stats.mode(predictions)
    return p[0]

predicted = RandomForest(x_test)
correct = 0
pred_length = predicted.size
#print(y_test.iloc[1])
#print(predicted[0,1])
for i in range(pred_length):
    if (predicted[0,i]==y_test.iloc[i]):
        correct = correct + 1
accuracy = correct/pred_length
accuracy = accuracy*100

print(accuracy)

categories = pd_train['class'].unique()
print(categories)
conf_matrix=confusion_matrix(y_test,predicted[0],categories)
print(conf_matrix)
model1.fit(pd_features[["sepal length","sepal width"]],pd_target)
model2.fit(pd_features[["petal length","petal width"]],pd_target)
model3.fit(pd_features[["sepal length","petal length"]],pd_target)
model4.fit(pd_features[["sepal width","petal width"]],pd_target)

predicted = RandomForest(pd_features)
correct = 0
pred_length = predicted.size
#print(y_test.iloc[1])
#print(predicted[0,1])
for i in range(pred_length):
    if (predicted[0,i]==pd_target.iloc[i]):
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
    model1.fit(pd_features.ix[train,["sepal length","sepal width"]],pd_target.iloc[train])
    model2.fit(pd_features.ix[train,["petal length","petal width"]],pd_target.iloc[train])
    model3.fit(pd_features.ix[train,["sepal length","petal length"]],pd_target.iloc[train])
    model4.fit(pd_features.ix[train,["sepal width","petal width"]],pd_target.iloc[train])

    predicted = RandomForest(pd_features.iloc[test])
    correct = 0
    pred_length = predicted.size
    y_test = pd_target[test]
    #print(y_test.iloc[1])
    #print(predicted[0,1])
    for i in range(pred_length):
        if (predicted[0,i]==y_test.iloc[i]):
            correct = correct + 1
    accuracy = correct/pred_length
    accuracy = accuracy*100
    acc.append(accuracy)

    print(accuracy)
    
print("mean: ",np.mean(acc))