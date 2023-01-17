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

import graphviz



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/heart.csv")

data.head(20)
data.info()
data.describe()
ax = sns.countplot(data.target)

yes , no = data.target.value_counts()





print("Number of people who does not have heart disease : {}".format(no))

print("Number of people who has heart disease : {}".format(yes))

ax = sns.countplot(data.sex)



men, women = data.sex.value_counts()



print("Number of women in this data {}".format(women))

print("Number of men in this data {}".format(men))
age_group = []

for each in data["age"]:

    if each < 20:

        age_group.append("0-20")

    elif 20 <= each < 30 :

        age_group.append("20-30")

    elif 30 <= each < 40 :

        age_group.append("30-40")

    elif 40 <= each < 50:

        age_group.append("40-50")

    elif 50 <= each < 60 :

        age_group.append("50-60")

    elif 60 <= each < 70 :

        age_group.append("60-70")

    else:

        age_group.append("70+")









data["age_group"] = age_group

ax = sns.countplot( data.age_group)



a = data.age_group.value_counts()

print(a)

        

    
new_da= data[data.age_group =="20-30"]

new_da
pd.crosstab(data.age_group, data.sex).plot(kind="bar", figsize = (10,6))

plt.title("Age group distribution due to gender")

plt.xlabel("age groups")

plt.ylabel("distribution")

plt.show()

pd.crosstab(data.sex, data.target).plot(kind ="bar", figsize = (10,6))

plt.title("Target distribution over sex")

plt.xlabel("Women  and  Men")

plt.ylabel("Distribution")

plt.show()

men_data = data[data.sex == 1]

women_data = data[data.sex == 0]



men_data_target = men_data[men_data.target == 1]

women_data_target= women_data[women_data.target == 1]



men_ratio = (len(men_data_target)* 100 )/ len(men_data)

women_ratio = (len(women_data_target)* 100) / len(women_data)



print("Percantage of men with heart disease {}".format(men_ratio))

print("Percantage of women with heart disease {}".format(women_ratio))
ax = sns.countplot(data.cp)

pd.crosstab(data.cp, data.target).plot(kind="bar", figsize = (10,6))

plt.xlabel("CP numbers")

plt.ylabel("Distribution")

plt.title("Which CP has the most heart disease event")

plt.show()
target_1_data = data[data.target == 1]

target_0_data = data[data.target == 0]



trace1 = go.Scatter(

                    x =target_1_data.age,

                    y =target_1_data.chol,

                    mode = "markers",

                    name = "Target = 1",

                    text = target_1_data.target,

                    marker = dict(color = "rgba(250,120,120,0.8)")

)



trace2 = go.Scatter(

                    x= target_0_data.age,

                    y = target_0_data.chol,

                    mode ="markers",

                    name = "Target = 0",

                    text = target_0_data.target,

                    marker = dict(color = "rgba(15,20,230,0.8)")

)



data_plot = [trace1, trace2]



layout = dict(title = "Scatter plot of target according to age and cholosterol",

        xaxis = dict(title = "Age"),

        yaxis = dict(title = "cholosterol")

             )

        

fig = dict(layout = layout, data = data_plot)

iplot(fig)

          

         

         
trace1 = go.Scatter(

                    x =target_1_data.age,

                    y =target_1_data.chol,

                    mode = "markers",

                    name = "Target = 1",

                    text = target_1_data.target,

                    marker = dict(color = "rgba(250,120,120,0.8)")

)



trace2 = go.Scatter(

                    x= target_0_data.age,

                    y = target_0_data.chol,

                    mode ="markers",

                    name = "Target = 0",

                    text = target_0_data.target,

                    marker = dict(color = "rgba(15,20,230,0.8)")

)



data_plot = [trace1, trace2]



layout = dict(title = "Scatter plot of target according to age and cholosterol",

        xaxis = dict(title = "Age"),

        yaxis = dict(title = "cholosterol")

             )

        

fig = dict(layout = layout, data = data_plot)

iplot(fig)

          
f,ax=plt.subplots(figsize = (18,18))

sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)



plt.show()
plt.hist(target_1_data.chol,bins = 50, fc = (0,1,0,0.5), label ="Heart Disease",

        normed = True, cumulative = True)

sorted_data= np.sort(target_1_data.chol)

y = np.arange(len(sorted_data))/float(len(sorted_data)-1)

plt.plot(sorted_data,y,color='red')

plt.title('CDF of bening heart disease')

plt.show()
y = data.target

x= data.drop(columns =["target","age_group"],axis=1)

norm_x = (x- x.min())/(x.max()-x.min())



data = pd.concat([y,norm_x], axis =1)

data = pd.melt(data,id_vars = "target",

                        var_name = "features",

                        value_name ="value")



plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="target", data=data,split=True, inner="quart")

plt.xticks(rotation=90)
plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value", hue="target", data=data)

plt.xticks(rotation=90)
sns.jointplot(x.loc[:,"trestbps"], x.loc[:,"oldpeak"], kind="reg")
sns.jointplot(x.loc[:,"age"], x.loc[:,"oldpeak"], kind="reg")
sns.jointplot(x.loc[:,"thalach"], x.loc[:,"slope"], kind="reg")
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(norm_x,y,

                                                    random_state = 42, test_size =0.3)

from sklearn.tree import DecisionTreeClassifier





dt = DecisionTreeClassifier(max_leaf_nodes=6)

dt.fit(x_train,y_train)

dt_score = dt.score(x_test,y_test)

print("Score of Decision Tree Classifier : {}".format(dt_score))

y_pred = dt.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix



cm=confusion_matrix(y_true,y_pred)

f,ax=plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot=True)



plt.show()



from sklearn.metrics import classification_report

target_names = ["class 0", "class 1"]

print(classification_report(y_true, y_pred, target_names=target_names))





feat_names = x.columns

targ_names = ["Yes", "No"]

data = export_graphviz(dt,out_file=None,feature_names=feat_names,class_names=targ_names,   

                         filled=True, rounded=True,  

                         special_characters=True)

graph = graphviz.Source(data)

graph
from sklearn.neighbors import KNeighborsClassifier



score_list = []



for each in range(1,15):

    knn = KNeighborsClassifier(n_neighbors = each)

    knn.fit(x_train,y_train)

    score_list.append(knn.score(x_test,y_test))



num_list = [each for each in range(1,15)]



score_data = pd.DataFrame(num_list)

score_data["score"] = score_list

score_data



knn2 = KNeighborsClassifier(n_neighbors = 9)

knn2.fit(x_train,y_train)

score = knn2.score(x_test,y_test)





print("Score of KNN algorithm with k value = 8  : {}".format(score))

y_pred = knn2.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix



cm=confusion_matrix(y_true,y_pred)

f,ax=plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot=True)



plt.show()
x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)



from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)

print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))

print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))







from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(norm_x, y)

score = lr.score(norm_x,y)

print("The score of linear regression : {} ".format(score))
