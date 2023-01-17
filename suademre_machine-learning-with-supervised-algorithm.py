# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# plotly library

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data.info()
data.head()
data.describe()
# I drop the unnecessery columns for my prediction

data = data.drop(['Unnamed: 32', 'id'], axis=1)

data.head()
data['diagnosis'].value_counts()
color_list = ['red' if i == 'M' else 'blue' for i in data.loc[:,'diagnosis']]

pd.plotting.scatter_matrix(data.iloc[:, 7:13],

                                       c=color_list,

                                       figsize= [10,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '.',

                                       edgecolor= "black")

plt.show()
data['diagnosis'] = [1 if x=='M' else 0 for x in  data['diagnosis']]
data.head()
#Choosing x and y values



#x is our features except diagnosis (classification columns)

#y is diagnosis

x_data = data.iloc[:,1:]

y = data['diagnosis']
# Normalization

x = (x_data - np.min(x_data) / (np.max(x_data) - np.min(x_data)))
x.head()
# train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1) # 0.3 means 30% of data is splitted for testing. Remaining 70% is used to train our data

print('x_train shape : ', x_train.shape)

print('y_train shape : ', y_train.shape)

print('x_test shape : ', x_test.shape)

print('y_test shape : ', y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train) # to train our data

predicted_values = knn.predict(x_test)

correct_values = np.array(y_test) # just to make them array

print('KNN (with k=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy
# find best n value for knn

best_neig= range(1,25) 

train_accuracy_list =[]

test_accuracy_list =[]



for each in best_neig:

    knn = KNeighborsClassifier(n_neighbors =each)

    knn.fit(x_train,  y_train)

    train_accuracy_list.append( knn.score(x_train, y_train))    

    test_accuracy_list.append( knn.score(x_test, y_test))    

    

        

print( 'best k for Knn : {} , best accuracy : {}'.format(test_accuracy_list.index(np.max(test_accuracy_list))+1, np.max(test_accuracy_list)))

plt.figure(figsize=[13,8])

plt.plot(best_neig, train_accuracy_list,label = 'Train Accuracy')

plt.plot(best_neig, test_accuracy_list,label = 'Test Accuracy')

plt.title('Neighbors vs accuracy ')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.legend()

plt.grid()

plt.xticks(best_neig)

plt.show()
#SVM

from sklearn.svm import SVC

svm = SVC(random_state = 1,gamma='auto')

svm.fit(x_train,y_train)

print("accuracy of svm: ",svm.score(x_test,y_test))
#Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("accuracy of naive bayes: ",nb.score(x_test,y_test))
#Decision Tree Classification

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("accuracy of Decision Tree Classification: ", dt.score(x_test,y_test))
#Random Forest Classcification

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100 ,random_state=1)

rf.fit(x_train,y_train)

print("accuracy of Random Forest Classicifation: ",rf.score(x_test,y_test))
score_list1=[]

for i in range(100,501,50):

    rf2=RandomForestClassifier(n_estimators=i,random_state=1)

    rf2.fit(x_train,y_train)

    score_list1.append(rf2.score(x_test,y_test))

plt.figure(figsize=(10,10))

plt.plot(range(100,501,50),score_list1)

plt.xlabel("number of estimators")

plt.ylabel("accuracy")

plt.grid()

plt.show()



print("Maximum value of accuracy is {} \nwhen n_estimators= {}.".format(max(score_list1),(1+score_list1.index(max(score_list1)))*100))
#Confusion matrix of RFC

y_pred=rf.predict(x_test)

y_true=y_test



#cm

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



#cm visualization

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("predicted value")

plt.ylabel("real value")

plt.show()
##Confusion matrix of KNN

y_pred1=knn.predict(x_test)

y_true=y_test

#cm

cm1=confusion_matrix(y_true,y_pred1)



#cm visualization

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm1,annot=True,linewidths=0.5,linecolor="blue",fmt=".0f",ax=ax)

plt.xlabel("predicted value")

plt.ylabel("real value")

plt.show()
#Confusion matrix of Decision Tree Classf.

y_pred2=dt.predict(x_test)

y_true=y_test

#cm

cm2=confusion_matrix(y_true,y_pred2)



#cm visualization

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm2,annot=True,linewidths=0.5,linecolor="green",fmt=".0f",ax=ax)

plt.xlabel("predicted value")

plt.ylabel("real value")

plt.show()
y_pred3=svm.predict(x_test)

y_true=y_test

#cm

cm3=confusion_matrix(y_true,y_pred3)



#cm visualization

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm3,annot=True,linewidths=0.5,linecolor="green",fmt=".0f",ax=ax)

plt.xlabel("predicted value")

plt.ylabel("real value")

plt.show()
y_pred4=nb.predict(x_test)

y_true=y_test

#cm

cm4=confusion_matrix(y_true,y_pred4)



#cm visualization

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm4,annot=True,linewidths=0.5,linecolor="green",fmt=".0f",ax=ax)

plt.xlabel("predicted value")

plt.ylabel("real value")

plt.show()
dictionary={"model":["KNN","SVM","NB","DT","RF"],"score":[knn.score(x_test,y_test),svm.score(x_test,y_test),nb.score(x_test,y_test),dt.score(x_test,y_test),rf.score(x_test,y_test)]}

df1=pd.DataFrame(dictionary)
#sort the values of data 

new_index5=df1.score.sort_values(ascending=False).index.values

sorted_data5=df1.reindex(new_index5)



# create trace1 

trace1 = go.Bar(

                x = sorted_data5.model,

                y = sorted_data5.score,

                name = "score",

                marker = dict(color = 'rgba(200, 125, 200, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = sorted_data5.model)

dat = [trace1]

layout = go.Layout(barmode = "group",title= 'Scores of Classifications')

fig = go.Figure(data = dat, layout = layout)

iplot(fig)
from sklearn.metrics import classification_report

print('Classification report: \n',classification_report(y_test,y_pred))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

print('logistic regression score: ', logreg.score(x_test, y_test))