# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv("../input/Admission_Predict.csv")

df.head()
serialNo = df["Serial No."].values

df.drop(["Serial No."],axis=1,inplace = True)



df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})

y = df["Chance of Admit"].values

x = df.drop(["Chance of Admit"],axis=1)



# separating train (80%) and test (%20) sets

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 32)



# normalization

from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler(feature_range=(0, 1))

x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])

x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])



y_train_01 = [1 if each > 0.8 else 0 for each in y_train]

y_test_01  = [1 if each > 0.8 else 0 for each in y_test]



# list to array

y_train_01 = np.array(y_train_01)

y_test_01 = np.array(y_test_01)
from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression()

lrc.fit(x_train,y_train_01)

print("score: ", lrc.score(x_test,y_test_01))

print("real value of y_test_01[1]: " + str(y_test_01[1]) + " -> the predict: " + str(lrc.predict(x_test.iloc[[1],:])))

print("real value of y_test_01[2]: " + str(y_test_01[2]) + " -> the predict: " + str(lrc.predict(x_test.iloc[[2],:])))



# confusion matrix

from sklearn.metrics import confusion_matrix

cm_lrc = confusion_matrix(y_test_01,lrc.predict(x_test))

# print("y_test_01 == 1 :" + str(len(y_test_01[y_test_01==1]))) # 29



# cm visualization

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_lrc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()



from sklearn.metrics import precision_score, recall_score

print("precision_score: ", precision_score(y_test_01,lrc.predict(x_test)))

print("recall_score: ", recall_score(y_test_01,lrc.predict(x_test)))



from sklearn.metrics import f1_score

print("f1_score: ",f1_score(y_test_01,lrc.predict(x_test)))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train_01)

print("score: ", nb.score(x_test,y_test_01))

print("real value of y_test_01[1]: " + str(y_test_01[1]) + " -> the predict: " + str(nb.predict(x_test.iloc[[1],:])))

print("real value of y_test_01[2]: " + str(y_test_01[2]) + " -> the predict: " + str(nb.predict(x_test.iloc[[2],:])))



# confusion matrix

from sklearn.metrics import confusion_matrix

cm_nb = confusion_matrix(y_test_01,nb.predict(x_test))

# print("y_test_01 == 1 :" + str(len(y_test_01[y_test_01==1]))) # 29

# cm visualization

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm_nb,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()



from sklearn.metrics import precision_score, recall_score

print("precision_score: ", precision_score(y_test_01,nb.predict(x_test)))

print("recall_score: ", recall_score(y_test_01,nb.predict(x_test)))



from sklearn.metrics import f1_score

print("f1_score: ",f1_score(y_test_01,nb.predict(x_test)))
y = np.array([lrc.score(x_test,y_test_01),nb.score(x_test,y_test_01)])

x = ["LogisticReg.","GaussianNB"]

plt.bar(x,y)

plt.title("Comparison of Classification Algorithms")

plt.xlabel("Classfication")

plt.ylabel("Score")

plt.show()