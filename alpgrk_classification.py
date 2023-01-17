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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected = True)

import plotly.graph_objs as go

import plotly.figure_factory as ff
#Import Dataset

data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")
# Data Preprocessing and Visualization

data.info()

data.describe()
classes = pd.get_dummies(data["class"])

classes
data.isnull().sum()
f,ax = plt.subplots(figsize=(5,5))

sns.scatterplot(x = data.pelvic_incidence,y = data.degree_spondylolisthesis, hue= data["class"],ax=ax)

plt.show()
trace1 = go.Scatter3d(

    x=data.pelvic_incidence,

    y=data.degree_spondylolisthesis,

    z=data["class"],

    mode='markers',

    marker=dict(

        size=10,

    )

)



df = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=df, layout=layout)

iplot(fig)
data["class"] = [2 if each == "Hernia" else 0 if each == "Normal" else 1 for each in data["class"]]

x_data = data.drop(["class"],axis = 1)





y = data["class"].values

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
#Classification



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



dc = DecisionTreeClassifier()

rf = RandomForestClassifier(n_estimators=301,random_state=42)

svm = SVC(gamma="auto")

knn = KNeighborsClassifier(n_neighbors=13)

nb = GaussianNB()





dc.fit(x_train,y_train)

rf.fit(x_train,y_train)

svm.fit(x_train,y_train)

knn.fit(x_train,y_train)

nb.fit(x_train,y_train)



y_pred_dc = dc.predict(x_test)

y_pred_rf = rf.predict(x_test)        

y_pred_svm = svm.predict(x_test)

y_pred_knn = knn.predict(x_test)

y_pred_nb = nb.predict(x_test)







print("Decision Tree Accuracy :",dc.score(x_test,y_test))

print("Random Forest Accuracy :",rf.score(x_test,y_test))

print("Support vector Accuracy :",svm.score(x_test,y_test))

print("KNN Accuracy :",knn.score(x_test,y_test))

print("Naive Bayes Accuracy :",nb.score(x_test,y_test))



d = { "Classifiar_Algortihm":["Decisin Tree","Random Forest","SVM","KNN","Naive Bayes"],

     "Accuracy":[dc.score(x_test,y_test),

                 rf.score(x_test,y_test),

                 svm.score(x_test,y_test),

                 knn.score(x_test,y_test),

                 nb.score(x_test,y_test)]}



Acc_Score = pd.DataFrame(data=d)

Acc_Score.sort_values(by=["Accuracy"],ascending = False,inplace = True)

print(Acc_Score)



f,ax = plt.subplots(figsize = (10,7))

sns.barplot(x = Acc_Score.Classifiar_Algortihm,y = Acc_Score.Accuracy)

sns.pointplot(x = Acc_Score.Classifiar_Algortihm,y = Acc_Score.Accuracy,color = "green",alpha = 0.8)

plt.xlabel("Classification Algorithm",fontsize=15,color="blue")

plt.ylabel("Accuracy Values",fontsize=15,color="blue")

plt.show()
# Model Evaluation with Confusion Matrix



from sklearn.metrics import confusion_matrix

dc_cm = confusion_matrix(y_pred_dc,y_test)

rf_cm = confusion_matrix(y_pred_rf,y_test)

svm_cm = confusion_matrix(y_pred_svm,y_test)

knn_cm = confusion_matrix(y_pred_knn,y_test)

nb_cm = confusion_matrix(y_pred_nb,y_test)



print("Decision Tree Confusion_matrix:",dc_cm,sep="\n")

print("Random Forest Confusion_matrix:",rf_cm,sep="\n")

print("Support Vector Confusion_matrix:",svm_cm,sep="\n")

print("KNN Confusion_matrix:",knn_cm,sep="\n")

print("Naive Bayes Confusion_matrix:",nb_cm,sep="\n")
#Find optimum k values for KNN

score_list = []



for each in range(1,50):

    knn2 = KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,50),score_list)

plt.xlabel("K Values")

plt.ylabel("KNN Accuracy")

plt.show()
#Find optimum n_estimators for Random Forest

score_list2 = []



for each in range(1,310,10):

    rf2 = RandomForestClassifier(n_estimators=each,random_state=42)

    rf2.fit(x_train,y_train)

    score_list2.append(rf2.score(x_test,y_test))

    

plt.plot(range(1,310,10),score_list2)

plt.xlabel("N_Estimator")

plt.ylabel("Random Forest Accuracy")

plt.show()