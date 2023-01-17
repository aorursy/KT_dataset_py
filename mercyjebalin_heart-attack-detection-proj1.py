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
d=pd.read_csv("/kaggle/input/heart-attack-prediction/data.csv")
d.columns
d.columns=['age', 'sex', 'cp', 'bp', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal','num']
import matplotlib.pyplot as plt
d.head()

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

d["chol"].replace({'?': 0},inplace=True)
#d["age"].replace({'?': 0},inplace=True)
d["bp"].replace({'?': 0},inplace=True)
#d["sex"].replace({'?': 0},inplace=True)
#d["cp"].replace({'?': 0},inplace=True)
#d["num"].replace({'?': 0},inplace=True)
x=np.array(d.iloc[:,1:5])
y=np.array(d[['num']])
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat
yhat_prob = LR.predict_proba(x)
yhat_prob
accuracy = metrics.accuracy_score(y_test, yhat)
accuracy_percentage = 100 * accuracy
accuracy_percentage
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['num=1','num=0'],normalize= False,  title='Confusion matrix')
print (classification_report(y_test, yhat))

print(len(d['chol']),len(yhat_prob[:,0]))
plt.scatter(d['age'],yhat_prob[:,0], s=10)
plt.show()
