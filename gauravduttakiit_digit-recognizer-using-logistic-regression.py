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

from sklearn.model_selection import train_test_split

from  sklearn.linear_model import LogisticRegression

from sklearn import metrics

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
test.head()
y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



# free some space

del train 



y_train.value_counts()
lr1=LogisticRegression(solver='lbfgs')

lr1.fit(X_train,y_train)

predict= lr1.predict(X_train)

predict
cm= metrics.confusion_matrix(y_train,lr1.predict(X_train))

cm
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
# predict result

result = lr1.predict(test)





result = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)



from sklearn.metrics import accuracy_score

trainaccuracy= accuracy_score(y_train,lr1.predict(X_train))

trainaccuracy

print("Train Data Accuracy    :{} %".format(round((trainaccuracy*100),2)))