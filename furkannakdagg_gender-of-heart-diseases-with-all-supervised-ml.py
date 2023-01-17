# import necessary libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Firstly, import data

data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
# First look to data

data.head()
data.info()
# Adjust x and y for making normalization

y = data.sex.values # make it np array

x_data = data.drop(["sex"], axis = 1) # everything, exludes sex, is on the x



# Normalization 

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

x_train.shape
x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T
best_accuracy = [] # in order to compare all results, I will create an empty list that will be filled after all tests.



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))



best_accuracy.append(lr.score(x_test.T,y_test.T))
x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T
from sklearn.neighbors import KNeighborsClassifier

score_list = [] # to keep all scores for plot

best_score = 0

best_k = 0

for each in range(1,15): # I will try for 15 values of k

    knn = KNeighborsClassifier(n_neighbors = each)

    knn.fit(x_train,y_train) # train the model

    score_list.append(knn.score(x_test,y_test))

    if (knn.score(x_test,y_test) > best_score): # if you find a value that bigger than before, keep it!

       best_score = knn.score(x_test,y_test)

       best_k = each

    

plt.plot(range(1,15), score_list) # x_axis=range(1,15), y_axis=score_list

plt.xlabel("k values")

plt.ylabel("Accuracy")

plt.show()



print("The best accuracy we got is ", best_score)

print("Best accuracy's k value is ", best_k)



best_accuracy.append(best_score)

from sklearn.svm import SVC

svm = SVC(random_state = 42)

svm.fit(x_train,y_train)

print("Accuracy of SVM: ",svm.score(x_test,y_test))



best_accuracy.append(svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("Accuracy of NB: ", nb.score(x_test,y_test))



best_accuracy.append(nb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("Accuracy of Decision Tree: ", dt.score(x_test,y_test))



best_accuracy.append(dt.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,random_state=1) 

#n_estimator = we determined of how many trees will be on our forest

rf.fit(x_train,y_train)

print("Accuracy of Random Forest: ", rf.score(x_test,y_test)) # It gives us a bit better results than decision tree.



best_accuracy.append(rf.score(x_test,y_test))

best_accuracy
# Bar Plot with Seaborn

sv_ml = ["Logistic Regression", "KNN", "SVM","Naive Bayes", "Decision Tree", "Random Forest"]



plt.figure(figsize=(15,10))

sns.barplot(x = sv_ml, y = best_accuracy)

plt.xticks(rotation= 30)

plt.xlabel('Accuracy')

plt.ylabel('Supervised Learning Types')

plt.title('Supervised Learning Types v Accuracy')

plt.show()
# Pie Chart with Seaborn

colors = ['red','green','blue','cyan','purple','yellow']

labels = sv_ml

explode = [0,0,0,0,0,0]

sizes = best_accuracy



# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, labels=labels, explode=explode, colors=colors, autopct='%1.1f%%')

plt.title('Comparison of Accuracies',color = 'brown',fontsize = 15)

plt.show()
# I think, Bar Plot with Plotly will be much useful than Seaborn's Bar Plot, because values are too close to read with Seaborn.

import plotly.graph_objs as go

import plotly.io as pio

# Create trace

trace1 = go.Bar(

                x = sv_ml,

                y = best_accuracy,

                name = "Accuracy Plot",

                marker = dict(color = 'rgba(10, 100, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5))

)

data = [trace1]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

pio.show(fig)
# go has been defined above

trace1 = go.Scatter(

                    x = sv_ml,

                    y = best_accuracy,

                    mode = "lines",

                    name = "Accuracy",

                    marker = dict(color = 'rgba(255, 0, 0, 0.7)')

)

# trace2 for finding the top points easily with attention getting colur

trace2 =go.Scatter( 

                    x = sv_ml,

                    y = best_accuracy,

                    mode = "markers",

                    name = "Highlight Point",

                    marker = dict(color = 'rgba(0, 255, 155, 1)')

)



data = [trace1,trace2]

layout = dict(title = 'Accuracies of Supervised Learning Types',

              xaxis= dict(title= 'Accuracy',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Types',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

pio.show(fig)
# It's just a bonus, not for an analysis, just because it is one of the my favorite plots :)

# Word Cloud

from wordcloud import WordCloud



sv_ml2 = ["Logistic Regression", "KNN", "SVM","Naive Bayes", "Decision Tree", "Random Forest"]

list_label = ["Type","Accuracy"]

list_col = [sv_ml2,best_accuracy]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped) 

df = pd.DataFrame(data_dict)



cloud = df.Type



plt.subplots(figsize=(10,10))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(cloud))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
