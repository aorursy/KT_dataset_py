# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import warnings
# filter warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")
data.drop(["id","Unnamed: 32"], axis = 1, inplace = True)
data.tail()
# malignant = M kotu huylu tümor
# benign    = B iyi huylu tümor
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis = 1)
# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=M.concavity_mean,
    y=M.area_mean,
    z=M.texture_mean,
    mode='markers',
    name = "malignant",
    marker=dict(
        size=M.radius_mean,
        color='red',                # set color to an array/list of desired values      
    )
)

trace2 = go.Scatter3d(
    x=B.concavity_mean,
    y=B.area_mean,
    z=B.texture_mean,
    mode='markers',
    name = "benign", 
    marker=dict(
        size=B.radius_mean,
        color='green',                # set color to an array/list of desired values      
    )
)



data1 = [trace1,trace2]
layout = go.Layout(
    scene = dict(
                    xaxis = dict(
                        title='Concavity  Mean'),
                    yaxis = dict(
                        title='Area Mean'),
                    zaxis = dict(
                        title='Texture Mean'),),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data1, layout=layout)
iplot(fig, filename='legend-names')
#Normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
param_grid = {"C": np.logspace(-3,3,7), "penalty" : ["l1","l2"]}

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,param_grid,cv = 10)
logreg_cv.fit(x_train,y_train)  
print("tuned hyperparameters: (best parameters): ", logreg_cv.best_params_)
print("accuracy: ", logreg_cv.best_score_)
logreg2 = LogisticRegression(C = 10, penalty="l2")
logreg2.fit(x_train,y_train)
print("score: ", logreg2.score(x_test,y_test))

from sklearn.neighbors import KNeighborsClassifier
score_list = []
for each in range(1,15):
    knn2 =  KNeighborsClassifier(n_neighbors= each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("{}nn score: {}".format(4,knn.score(x_test,y_test)))

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors" :np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid , cv = 10) # GridSearchCV
knn_cv.fit(x,y)

print("tuned hyperparameter K:  ",knn_cv.best_params_)
print("tuned parametreye gore en iyi accuracy (best score): ",knn_cv.best_score_)


from sklearn.svm import SVC

svm = SVC(random_state = 1)
svm.fit(x_train,y_train)

print("Accuracy of svm algo: ",svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)



print("Accuracy of Naive Bayes Algo: ", nb.score(x_test,y_test))

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("Accuracy of Decision Tree Algo: ", dt.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100, random_state = 1)
rf.fit(x_train,y_train)

print("Accuracy of Random Forest Algo: ", rf.score(x_test,y_test))
y_pred = rf.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm,annot = True, linewidths = 0.5, linecolor = "red", fmt = ".0f",ax = ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show

