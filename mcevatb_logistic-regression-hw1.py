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

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from sklearn.metrics import accuracy_score



# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/diabetes.csv")
data.head()
data.info()
data.shape
data.describe()
# correlation between features

data.Outcome =["D" if each == 1 else "ND" for each in data.Outcome]



sns.pairplot(data=data,palette="Set2",hue="Outcome")

plt.show()
data.Outcome =["1" if each == "D" else "0" for each in data.Outcome]

# we find out number of zeros in each feature

zeros = (data == 0)

zeros.sum(axis=0)
# we replace zeros of each column ex. pregnancies ,age  and outcome with their column's mean 

for each in data.columns[1:6]:

    data[each] = data[each].replace(0, data[each].median())

data.head()

y=data.Outcome.values                                  

x_data=data.drop(['Outcome'],axis=1)

print(y.shape,x_data.shape)

x_data.head()
#normalization: to get a value between 0 and 1 for each feature to prevent  some features from being dominant

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

x.head()
diabet = data["BMI"]

similarity_with_other_col = data.corrwith(diabet) #correlation of BMI with other features

similarity_with_other_col
#Seaborn Heatmap to find out correlation between each feature

f,ax = plt.subplots(figsize=(12,10))

cmap=sns.diverging_palette(150, 275, s=80, l=55,n=9)

sns.heatmap(

data.corr(), 

annot=True, annot_kws={'size':12},

linewidths=.8,linecolor="blue", fmt= '.2f',ax=ax,square=True,cmap=cmap)



plt.show()
from sklearn.model_selection import train_test_split

#we split our data in 80% train and 20% test data

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

#random state is important to get the same values after each forward_backward_propagation



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
#Logistic Regression with sklearn

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(C=100,penalty="l2",solver="saga",class_weight=None)

#C : float, default value: 1.0

#Inverse of regularization strength should be  positive, 

#small values>stronger regularization.

#solver : For small datasets choose ‘liblinear’ ,‘sag’ and ‘saga’ are faster for large datasets.

#For multiclass problems choose only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ for multinomial loss

#‘newton-cg’, ‘lbfgs’ and ‘sag’ with L2 penalty,‘liblinear’ and ‘saga’ with L1 penalty.

logreg.fit(x_train, y_train)

y_pre_lr = logreg.predict(x_test)



test_acc= logreg.score(x_test,y_test) 



print("LR accuracy :  ",test_acc)

lr_acc=logreg.score(x_test,y_test)

from sklearn.model_selection import GridSearchCV

grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  



logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg,grid,cv = 10)

logreg_cv.fit(x_train,y_train)

y_pre_lrcv = logreg_cv.predict(x_test)

print("tuned hyperparameters: ",logreg_cv.best_params_)

print("lr_accuracy: ",logreg_cv.best_score_)



logreg2 = LogisticRegression(C=100.0,penalty="l1")

logreg2.fit(x_train,y_train)

print("lr_score: ", logreg2.score(x_test,y_test))

# knn 

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

           metric_params=None, n_jobs=None, n_neighbors=16, p=2,

           weights='uniform')

#‘distance’ : weight points, closer neighbors  have  greater influence than neighbors far away.

#‘uniform’ : uniform weights, all points are weighted equally

#algorithm :‘auto’  to decide the most appropriate algorithm 

# n_neighbors = k

knn.fit(x_train,y_train)

y_pre_knn = knn.predict(x_test)

print(" With KNN (K= {}) accuracy is: {} ".format(16,knn.score(x_test,y_test)))

knn_acc=knn.score(x_test,y_test)
# find k value

k_list = []

for each in range(1,25):

    knn_2 = KNeighborsClassifier(n_neighbors = each)

    knn_2.fit(x_train,y_train)

    k_list.append(knn_2.score(x_test,y_test))

    



f = plt.subplots(figsize=(18,8))

plt.plot(range(1,25),k_list)

   

plt.xlabel('k values',fontsize = 15,color='black')             

plt.ylabel('accuracy',fontsize = 15,color='black')

plt.title('K values-Accuracy Plot',fontsize = 20,color='black')

plt.xticks(range(1,25))

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(k_list),1+k_list.index(np.max(k_list))))
from sklearn.svm import SVC



svm = SVC(C=100.0, cache_size=200, class_weight="balanced", kernel='rbf',max_iter=-1)

#Penalty parameter C ,default = 1.0

svm.fit(x_train,y_train)

y_pre_svm = svm.predict(x_test)

print("SVM accuracy is: ",accuracy_score(y_test, y_pre_svm))

svm_acc=accuracy_score(y_test, y_pre_svm)
 # %% Naive bayes 

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

y_pre_nb = nb.predict(x_test)



print("NB accuracy is: ",nb.score(x_test,y_test))

nb_acc=nb.score(x_test,y_test)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(class_weight="balanced",max_leaf_nodes=100)

dt.fit(x_train,y_train)



y_pre_dt = dt.predict(x_test)



print("DT accuracy is: ", dt.score(x_test,y_test))

dt_acc= dt.score(x_test,y_test)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 500,max_depth=200)

rf.fit(x_train,y_train)



y_pre_rf = rf.predict(x_test)

print("RF accuracy is: ",rf.score(x_test,y_test))

rf_acc=rf.score(x_test,y_test)
import numpy  as np

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt



cm_lr = confusion_matrix(y_test,y_pre_lr)        #logistic regression

cm_dt = confusion_matrix(y_test,y_pre_dt)        #decision tree

cm_knn = confusion_matrix(y_test,y_pre_knn)      #nearest neighbors

cm_nb = confusion_matrix(y_test,y_pre_nb)        #naine bayes

cm_rf = confusion_matrix(y_test,y_pre_rf)        #random forest

cm_svm = confusion_matrix(y_test,y_pre_svm)      #support vector machine



cm= np.array([cm_lr,cm_dt,cm_knn,cm_nb,cm_rf,cm_svm])

  



plt.figure(figsize=(20,10))

plt.suptitle("Confusion Matrix",fontsize=24,color="b") 

classification = np.array(["Logistic Regression","Decision Tree","Random Forest","K Nearest Neighbors","Naive Bayes","Support Vector Machine"])



i=0

k=1

while i < len(classification):

    plt.subplot(2,3,k)

    plt.title(classification[i],fontsize=14,color="b")

    sns.heatmap(cm[i],cbar=False,annot=True,cmap="PuBuGn",fmt="d",linewidths=.8,linecolor="red")

    i=i+1

    k=k+1



plt.show()

 

        



      
dictionary = {"Class":["Logistic Regression","Decision Tree","Random Forest","K Nearest Neighbors","Naive Bayes","Support Vector Machine"],

              "Accuracy":[lr_acc,dt_acc,rf_acc,knn_acc,nb_acc,svm_acc]} 

dataFrame1 = pd.DataFrame(dictionary)

dataFrame1


fig, ax = plt.subplots(figsize=(15,10))

N = 6  # number of groups

ind = np.arange(N)  # group positions

width = 0.2  # bar width



sns.barplot(x=dataFrame1['Class'], y=dataFrame1["Accuracy"])



ax.set_xticks(ind + width)

ax.set_xticklabels(['LogisticRegression\n',

                    "Decision Tre\n",

                    'RandomForest\n',

                    "K Nearest Neighbors\n",

                    'Naive Bayes\n',

                    'Support Vector Machine\n'],

                   rotation=40,

                   ha='right',fontsize = 13,color='magenta')

plt.xlabel('Class',fontsize = 18,color='blue')

plt.ylabel('Accuracy',fontsize = 18,color='blue')

plt.ylim(0.65,0.85)

plt.title('Class-Accuracy Diagram',fontsize = 20,color='blue')



plt.savefig('graph.png')

plt.grid()  


trace1 = go.Bar(

                x = dataFrame1['Class'],

                y = dataFrame1["Accuracy"],

                name = "Accuracy",

                marker = dict(color = ['rgba(160, 200, 155, 0.7)','rgba(60, 20, 155, 0.7)','rgba(16, 200, 55, 0.7)','rgba(90, 2, 155, 0.7)',

                              'rgba(33, 234, 155, 0.7)','rgba(67, 56, 155, 0.7)'],

                             line=dict(color='rgba(0,0,0)',width=2)))

dt = [trace1]

layout = go.Layout(barmode = "relative",title = 'Class-Accuracy Diagram',hovermode='closest',font=dict(family='Arial', size=14,color="rgba(123,34,121,0.7)"),

         xaxis= dict(title= 'Class',ticklen= None,zeroline= False,gridwidth=2,tickangle=-20), 

         yaxis= dict(title= 'Accuracy',ticklen= None,zeroline= False,gridwidth=2))







fig = go.Figure(data = dt, layout = layout)



iplot(fig)