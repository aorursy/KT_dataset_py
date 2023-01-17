# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read CSV

df = pd.read_csv("../input/Iris.csv")



y = np.array(df[['Species']])

x = np.array(df.drop(['Id','Species'], axis=1))



print("y")

print(y[0:5])

print("x")

print(x[0:5])
#Split Data

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, stratify=y)

#Linear SVM



#Fit The plot

from sklearn.svm import LinearSVC



model = LinearSVC()

model.fit(x_train, y_train.ravel())





#Calculate Test Prediction

y_pred = model.predict(x_test)

print(model.score(x_test,y_test.ravel()))



#Plot Confusion Matrix

from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)



import matplotlib.pyplot as plt

import seaborn as sn



df_cm = pd.DataFrame(cm, index = [i for i in np.unique(y)],

                  columns = [i for i in np.unique(y)])

plt.figure(figsize = (5,5))

sn.heatmap(df_cm, annot=True)
#Non-linear SVM



#Fit The plot

from sklearn.svm import SVC



model = SVC()

model.fit(x_train, y_train.ravel())





#Calculate Test Prediction

y_pred = model.predict(x_test)

print(model.score(x_test,y_test.ravel()))



#Plot Confusion Matrix

from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)



import matplotlib.pyplot as plt

import seaborn as sn



df_cm = pd.DataFrame(cm, index = [i for i in np.unique(y)],

                  columns = [i for i in np.unique(y)])

plt.figure(figsize = (5,5))

sn.heatmap(df_cm, annot=True)