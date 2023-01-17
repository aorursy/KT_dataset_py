# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 

import seaborn as sb

from sklearn.metrics import accuracy_score
X_train=pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
X_test=pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
X_train.shape
X_test.shape
y_train=X_train.iloc[:,0]
y_train.head()
X_train.drop("label",axis=1,inplace=True)
Id=X_test["id"]
Id.head()
X_test.drop("id",axis=1,inplace=True)
X_train.shape
X_test.shape
plt.figure(figsize=(20,20))

for index,(image,label) in enumerate(zip(np.array(X_train.iloc[:25,:]),np.array(y_train.iloc[:25]))):

    plt.subplot(5,5,index+1)

    plt.imshow(np.reshape(image,(28,28)))

    plt.title("training {}".format(label))
from sklearn.linear_model import LogisticRegression
logistic_model=LogisticRegression(solver="lbfgs").fit(X_train,y_train)
logistic_predictions=logistic_model.predict(X_test)
logistic_predictions.shape
submission=pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")
submission.shape
logistic_model.score(X_train,y_train)
submission.drop("label",axis=1,inplace=True)
submission.head()
submission["label"]=logistic_predictions
submission.shape
submission.tail()
submission.to_csv("submission.csv",index=False)