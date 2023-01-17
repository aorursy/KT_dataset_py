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
from sklearn.datasets import fetch_openml

from sklearn import metrics

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
test.head()
from sklearn.model_selection import train_test_split

trains,tests= train_test_split(train, train_size=0.7, random_state=42)
y_train = trains["label"]

X_train = trains.drop(labels = ["label"],axis = 1) 

y_test = tests["label"]

X_test = tests.drop(labels = ["label"],axis = 1) 
X_train.info()
from sklearn.ensemble import GradientBoostingClassifier

model= GradientBoostingClassifier(learning_rate=0.01,random_state=42)
model.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

testaccuracy= accuracy_score(y_test,model.predict(X_test))

testaccuracy

print("Test Data Accuracy    :{} %".format(round((testaccuracy*100),2)))
cm= metrics.confusion_matrix(y_test,model.predict(X_test))

cm
model.score(X_test,y_test)
plt.figure(figsize=(9,9))

plt.imshow(cm,cmap='rainbow_r')

plt.title("Confusion Matrix for MNIST Data")

plt.xticks(np.arange(10))

plt.yticks(np.arange(10))

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')

plt.colorbar()

width,height = cm.shape

for x in range(width):

    for y in range(height):

        plt.annotate(str(cm[x][y]),xy=(y,x),horizontalalignment='center',verticalalignment='center')

plt.show()
y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 
from sklearn.ensemble import GradientBoostingClassifier

model= GradientBoostingClassifier(learning_rate=0.01,random_state=42)
model.fit(X_train,y_train)
# predict result

pred = model.predict(test)

result=pd.DataFrame(pred)

result= result.rename(columns={ 0 : 'Label'})

result.head()
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)



submission.head()