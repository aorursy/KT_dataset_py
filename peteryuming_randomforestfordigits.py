# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")
x = df_train.drop(["label"],axis = 1)
y = df_train["label"]
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=42)
x_train.isnull().any().describe()
x_train = x_train/255
x_test = x_test/255
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = "gini",min_samples_split = 2 )
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm,classes, normalize =False, title = "Confusion_Matrix",cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation = "nearest", cmap = cmap)
    plt.title=title
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation = 45)
    plt.yticks(tick_marks,classes)
    
    if normalize:
        cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]
    
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(i,j,cm[i,j],horizontalalignment = "center", color= "white" if cm[i,j]>thresh else "black")
    
    plt.tight_layout()
    plt.ylabel("true_labal")
    plt.xlabel("pred_label")

confusion_mtx = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(confusion_mtx,classes = range(10))
x_result = pd.read_csv("../input/test.csv")
y_result = classifier.predict(x_result)
result = pd.Series(y_result,name = "Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
submission.to_csv("randomforest_digits.csv",index = False)
