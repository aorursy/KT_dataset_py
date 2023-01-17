# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
print(os.listdir("../input"))



data = pd.read_csv("../input/data.csv")
data

import random
a = 0
for i in range(0, len(data)):
    S = data["symptom"][i].split()
    travel = data["visiting Wuhan"][i]
    if ('0' in S):
        if ('6' in S) or ('8' in S) or ('4' in S):
            if travel == 1:
                data["result"][i] = 0
    elif travel == 0 and ('4' in S):
        data["result"][i] = 1
    elif ('5' in S):
        if ('6' in S):
            data["result"][i] = 2
    else:
        symtoms = [0] * 10 + [2] * 4
        choice = random.choice(symtoms)
        data["result"][i] = choice
sym = []
pre = []
asym = []
for i in data['result']:
    if i == 0:
        pre.append(1)
    if i == 1:
        asym.append(1)
    if i == 2:
        sym.append(1)

print(len(sym), len(pre), len(asym))
            


from sklearn.linear_model import LogisticRegression
X = data[["symptom", "age", "visiting Wuhan"]]
Y = data['result']
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(X["symptom"])
X["symptom"] = le.transform(X["symptom"])
model =LogisticRegression()

model.fit(X, Y)
data
le.transform(['3 6 1'])
predict = model.predict(X)


import seaborn as sns
import matplotlib.pyplot as plt
plt.title("Logistic Regression with Score 74.76%")
sns.lineplot(x='result', y=np.arange(0, len(data)), data=data, label="Real Values")
sns.lineplot(x=predict, y=np.arange(0, len(data)), label="Predicted Values")
plt.savefig("LogisticGraph.pdf")

from sklearn.metrics import accuracy_score
accuracy_score(Y, predict)*100

from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier()
model.fit(X, Y)
predict = model.predict(X)

accuracy_score(Y, predict)*100
import seaborn as sns
plt.title("Random Forest with Score 99.63%")

sns.lineplot(x='result', y=np.arange(0, len(data)), data=data)
sns.lineplot(x=predict, y=np.arange(0, len(data)))
plt.savefig("RANDOMFORESTGRAPH.pdf")
estimator = model.estimators_[1]
estimator

from sklearn.tree import export_graphviz

export_graphviz(estimator, out_file='tree_limited.dot', feature_names = X.columns,
                class_names = ['1', '2' , '3'],
                rounded = True, proportion = False, precision = 2, filled = True)
X.columns
!dot -Tpng tree_limited.dot -o modelInAtree.png

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y, predict)
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (16, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)
plot_confusion_matrix(cm, classes = ['presym', 'asym', 'sym'],
                      title = 'Poverty Confusion Matrix')
plt.savefig("ConfusionMatrixfgraph454.pdf")
sym = []
pre = []
asym = []
for i in data['result']:
    if i == 0:
        pre.append(1)
    if i == 1:
        asym.append(1)
    if i == 2:
        sym.append(1)

print(len(sym), len(pre), len(asym))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
model_fc = RandomForestClassifier()
model_fc.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)*100
