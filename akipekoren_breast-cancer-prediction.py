# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")

data.head(20)


data = data.loc[:,["diagnosis","radius_mean",

            "fractal_dimension_mean",

           "texture_mean","perimeter_mean",

           "area_mean", "smoothness_mean",

           "compactness_mean","concavity_mean",

           "concave points_mean","symmetry_mean",

           "fractal_dimension_mean"]]



data.info()
data.describe()
data.columns
sns.countplot(data.diagnosis)



data["diagnosis"].value_counts()
target_M_data = data[data.diagnosis == "M"] 

target_B_data = data[data.diagnosis == "B"]



trace1 = go.Scatter(

                    x= target_M_data.radius_mean,

                    y= target_M_data.area_mean,

                    mode = "markers",

                    name = "Malignant",

                    marker = dict(color = "rgba(120,15,150,0.8)"),

                    text = target_M_data.diagnosis)



trace2 = go.Scatter(

                    x= target_B_data.radius_mean,

                    y= target_B_data.area_mean,

                    mode = "markers",

                    name = "Beningn",

                    marker = dict(color = "rgba(66,222,222,0.8)"),

                    text = target_B_data.diagnosis)



scatter_data = [trace1, trace2]



layout = dict(title = "M or B according to radius and area",

             xaxis = dict(title = "radius"),

             yaxis = dict(title = "area")

             )



fig = dict(layout = layout, data = scatter_data)



iplot(fig)

trace1 = go.Scatter(

                    x= target_M_data.radius_mean,

                    y= target_M_data.smoothness_mean,

                    mode = "markers",

                    name = "Malignant",

                    marker = dict(color = "rgba(120,15,150,0.8)"),

                    text = target_M_data.diagnosis)



trace2 = go.Scatter(

                    x= target_B_data.radius_mean,

                    y= target_B_data.smoothness_mean,

                    mode = "markers",

                    name = "Beningn",

                    marker = dict(color = "rgba(66,222,222,0.8)"),

                    text = target_B_data.diagnosis)



scatter_data = [trace1, trace2]



layout = dict(title = "M or B according to radius and smoothness",

             xaxis = dict(title = "radius"),

             yaxis = dict(title = "smoothness")

             )



fig = dict(layout = layout, data = scatter_data)



iplot(fig)



plt.bar(data.radius_mean, data.smoothness_mean, color="blue")


trace1 = go.Scatter(

                    x= target_M_data.concavity_mean,

                    y= target_M_data.compactness_mean,

                    mode = "markers",

                    name = "Malignant",

                    marker = dict(color = "rgba(120,15,150,0.8)"),

                    text = target_M_data.diagnosis)



trace2 = go.Scatter(

                    x= target_B_data.concavity_mean,

                    y= target_B_data.compactness_mean,

                    mode = "markers",

                    name = "Beningn",

                    marker = dict(color = "rgba(66,222,222,0.8)"),

                    text = target_B_data.diagnosis)



scatter_data = [trace1, trace2]



layout = dict(title = "M or B according to concavity and compactness",

             xaxis = dict(title = "concavity"),

             yaxis = dict(title = "compactness")

             )



fig = dict(layout = layout, data = scatter_data)



iplot(fig)


trace1 = go.Scatter(

            x = target_M_data.symmetry_mean,

            y = target_M_data.texture_mean,

            mode = "markers",

            name = "Malignant",

            marker = dict(color = "rgba(200,5,5,0.8)"),

            text = target_M_data.diagnosis



)



trace2 = go.Scatter(

            x = target_B_data.symmetry_mean,

            y = target_B_data.texture_mean,

            mode = "markers",

            name = "Beningn",

            marker = dict(color = "rgba(5,5,200,0.8)"),

            text = target_B_data.diagnosis



)





graph_data = [trace1, trace2]



layout = dict(title =  "M or B according to symmetry and texture",

             xaxis = dict(title = "symmetry"),

             yaxis = dict(title = "texture"))





fig = dict(layout = layout,data = graph_data)





iplot(fig)



radius_group = []



for each in data.radius_mean:

    if each <= 13:

        radius_group.append("probably good")

    elif 13 < each < 17:

        radius_group.append("test is needed")

    else:

        radius_group.append("probably bad")

        

        

data["first_look"] = radius_group



sns.countplot(data.first_look)



data["first_look"].value_counts()







pd.crosstab(data.first_look,data.diagnosis).plot(kind="bar",figsize=(10,6))



plt.title("Is first look successfull to diagnose")

plt.xlabel("First look observations")

plt.ylabel("Distribution")

plt.show()
y = data.diagnosis



x = data.drop(["diagnosis","first_look"],axis=1)



x_norm = (x-x.min())/(x.max()-x.min())



new_data = pd.concat([y,x_norm],axis=1)

new_data = pd.melt(new_data,

                  id_vars ="diagnosis",

                  var_name = "features",

                  value_name = "value")



plt.figure(figsize = (20,10))



sns.boxplot(x="features", y="value", hue = "diagnosis", data=new_data)

plt.xticks(rotation = 90)



from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x_norm,y,test_size =0.3,random_state=42)



print("X train :  {}".format(x_train.shape))

print("X test : {}".format(x_test.shape))

print("Y train : {}".format(y_train.shape))

print("Y test : {}".format(y_test.shape))
from sklearn.neighbors import KNeighborsClassifier



train_score = []

test_score =[]



for each in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=each)

    knn.fit(x_train,y_train)

    

    train_score.append(knn.score(x_train,y_train))

    test_score.append(knn.score(x_test,y_test))





plt.figure(figsize = (10,6))

plt.plot(train_score, label = "Train accuracy")

plt.plot(test_score, label = "Test accuracy")

plt.grid()

plt.xlabel("n neighbor")

plt.ylabel("score")





check = 0

count =0

for num in test_score:

    if check < num :

        check = num

        count += 1

    else:

        pass



print("The most accuracy is {} with the neighbor value of {}".format(check,count))



knn2 = KNeighborsClassifier(n_neighbors = 2)

knn2.fit(x_train,y_train)





y_pred = knn2.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_true,y_pred)



sns.heatmap(cm,annot= True,fmt="d")

data1 = data[data.diagnosis =="M"]





x = np.array(data1.loc[:,"radius_mean"]).reshape(-1,1)



y= np.array(data1.loc[:,"concave points_mean"]).reshape(-1,1)



plt.figure(figsize = (10,6))



plt.scatter(x=x,y=y)

plt.title("Graph of radius and concave point")

plt.xlabel("radius")

plt.ylabel("concave point")



from sklearn.linear_model import LinearRegression



reg = LinearRegression()

predict_space = np.linspace(min(x), max(x)).reshape(-1,1)





reg.fit(x,y)

predict = reg.predict(predict_space)



print("Score of regression : {}".format(reg.score(x,y)))



plt.figure(figsize = (10,6))

plt.plot(predict_space, predict, color='black', linewidth=3)

plt.scatter(x=x,y=y)

plt.title("Graph of radius and concave point")

plt.xlabel("radius")

plt.ylabel("concave point")













from sklearn.model_selection import cross_val_score

reg = LinearRegression()

k = 5

cv_result = cross_val_score(reg,x,y,cv=k) 

print('CV Scores: ',cv_result)

print('CV scores average: ',np.sum(cv_result)/k)
# ROC Curve with logistic regression

from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

# abnormal = 1 and normal = 0

data['class_binary'] = [1 if i == "M" else 0 for i in data.loc[:,"diagnosis"]]

x,y = data.loc[:,(data.columns != "diagnosis") & (data.columns != "class_binary") &(data.columns != "first_look")],data.loc[:,'class_binary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

logreg = LogisticRegression()

logreg.fit(x_train,y_train)

y_pred_prob = logreg.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.show()
from sklearn.ensemble import RandomForestClassifier



x_train , x_test, y_train, y_test = train_test_split(x_norm,y,test_size = 0.3,random_state = 42)

rf = RandomForestClassifier()



rf.fit(x_train,y_train)



score = rf.score(x_test,y_test)



print("Accuracy of the Random Forest Algorithm : {}".format(score))



from sklearn.metrics import confusion_matrix



y_pred = rf.predict(x_test)

y_true = y_test



cm = confusion_matrix(y_true,y_pred)

sns.heatmap(cm, annot = True,fmt="d")
from sklearn.metrics import classification_report



report = classification_report(y_true,y_pred)



print("Classification report : {}".format(report))
plt.scatter(data["radius_mean"], data["compactness_mean"])

plt.xlabel("radius")

plt.ylabel("compactness")

plt.title("Distribution")
new_data = data.loc[:,["radius_mean","compactness_mean"]]

from sklearn.cluster import KMeans



kmean = KMeans(n_clusters = 2)



kmean.fit(new_data)



labels = kmean.predict(new_data)



plt.scatter(data["radius_mean"], data["compactness_mean"], c = labels)
df = pd.DataFrame({"label" : labels , "class" : data.diagnosis} )



new_df = pd.crosstab(df["label"],df["class"])

new_df