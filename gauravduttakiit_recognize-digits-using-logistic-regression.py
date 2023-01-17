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
df=fetch_openml("mnist_784")
df.data[:5]
df.target[:5]
plt.figure(figsize=(10,5))

for index, (image, label) in enumerate(zip(df.data[:5],df.target[:5])):

    plt.subplot(1,5,index+1)

    plt.imshow(np.reshape(image,(28,28)),cmap='rainbow')

    plt.title("Number: %s" % label)
X_train,X_test, y_train,y_test = train_test_split(df.data,df.target,test_size=0.3, random_state=125)
lr1=LogisticRegression(solver='lbfgs')

lr1.fit(X_train,y_train)

predict= lr1.predict(X_test)

score=lr1.score(X_test,y_test)

print(score)
from sklearn.metrics import accuracy_score

trainaccuracy= accuracy_score(y_train,lr1.predict(X_train))

trainaccuracy

testaccuracy= accuracy_score(y_test,lr1.predict(X_test))

testaccuracy





index=1

plt.imshow(np.reshape(X_test[index],(28,28)))

print("Prediction: " + lr1.predict([X_test[index]])[0])
cm= metrics.confusion_matrix(y_test,predict)

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
# Let us compare the values obtained for Train & Test:

print("Train Data Accuracy    :{} %".format(round((trainaccuracy*100),2)))

print("Test Data Accuracy     :{} %".format(round((testaccuracy*100),2)))
