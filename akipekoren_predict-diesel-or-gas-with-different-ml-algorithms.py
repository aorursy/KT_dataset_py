# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import scipy.stats as stats

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
data = pd.read_csv("../input/auto_clean.csv")

data.head(20)
data.isnull().values.any() # So we have missing value

data.info()
data["stroke"].value_counts(dropna = False) # That shows us we have 4 missing value.
avg = data["stroke"].mean() # we can fill the Nan values with the average

avg
data["stroke"].fillna(avg, inplace = True)

data["stroke"].value_counts(dropna = False)

data["horsepower-binned"].value_counts(dropna = False)



avg_list = []

for each in data["horsepower-binned"]:

    if each == "Low":

        avg_list.append(1)

    elif each == "Medium":

        avg_list.append(2)

    else:

        avg_list.append(3)

sum = 0

count = 0

for num in avg_list:

    sum = sum +num

    count += 1



avg = sum/count

avg
data["horsepower-binned"].fillna("Medium", inplace = True)

data.isnull().values.any() # We get false, so now our data has not any missin value.
data.columns = data.columns.str.replace("-","_")



gas_or_diesel = ["diesel"if each == 0 else "gas" for each in data.gas]

data["gas_or_diesel"] = gas_or_diesel
data["make"].unique()
labels = data["make"].unique()

fig = {

  "data": [

    {

      "values": data["make"].value_counts(),

      "labels": labels,

      "domain": {"x": [0, .5]},

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Distribution of Car makers",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Pie Chart",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
ax = sns.countplot(data.gas_or_diesel, label = "Counts")

Gas, Diesel = data.gas_or_diesel.value_counts()

print("Number of gas user car : {}".format(Gas))

print("Number of diesel user car : {}".format(Diesel))


fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'gas_or_diesel', y = 'horsepower', data = data)

fig = plt.figure (figsize = (10,6))

sns.barplot(x = "gas_or_diesel", y = "city_L/100km" ,data = data )
fig = plt.figure(figsize = (10,6))

sns.barplot(x = "gas_or_diesel" , y = "price", data = data)
fig = plt.figure (figsize = (10,6))

sns.barplot(x = "gas_or_diesel", y = "stroke",data = data)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = "gas_or_diesel", y = "engine_size", data = data)
y = data.gas_or_diesel

x= data[["normalized_losses","wheel_base","length",

        "width","height","curb_weight","engine_size",

        "bore","stroke","horsepower","peak_rpm","city_mpg",

        "highway_mpg","price","city_L/100km"]]



x_norm = (x- x.min()) / (x.max()- x.min())



new_data = pd.concat([y,x_norm], axis = 1)



new_data = pd.melt(new_data,

                  id_vars = "gas_or_diesel",

                  var_name = "features",

                  value_name = "values")

sns.violinplot(x = "features", y = "values", hue = "gas_or_diesel", data= new_data,

              split=True, inner = "quart")

plt.xticks(rotation = 90)
plt.figure(figsize = (10,10))

sns.boxplot(x = "features", y = "values", hue = "gas_or_diesel", data = new_data)

plt.xticks(rotation = 90)
sns.set(style ="darkgrid",color_codes = True)



a = sns.jointplot(x_norm.loc[:,"width"],x_norm.loc[:,"height"],data = x_norm,

                 kind="reg", height=8,color="#ce1414")

a.annotate(stats.pearsonr)

plt.show()
sns.set(style="darkgrid", color_codes = True)

a = sns.jointplot(x_norm.loc[:,"engine_size"],x_norm.loc[:,"horsepower"],

             data = x_norm, kind ="reg", height = 8, color ="#ce1414")

a.annotate(stats.pearsonr)

plt.show()
sns.set(style ="darkgrid", color_codes = True)



a = sns.jointplot(x_norm.loc[:,"city_L/100km"], x_norm.loc[:,"horsepower"],

                  data = x_norm, kind ="reg", height = 8, color = "#ce1414")



a.annotate(stats.pearsonr)

plt.show()
sns.set(style="darkgrid", color_codes = True)

a = sns.jointplot(x_norm.loc[:,"engine_size"], x_norm.loc[:,"peak_rpm"],

                 data = x_norm, kind = "reg", height = 8 , color = "#ce1414")

a.annotate(stats.pearsonr)

plt.show()
sns.set(style="darkgrid", color_codes = True)

a = sns.jointplot(x_norm.loc[:,"stroke"],x_norm.loc[:,"horsepower"],data = x_norm,

                 kind = "reg", height = 8 , color = "#ce1414")

a.annotate(stats.pearsonr)

plt.show()
f,ax = plt.subplots(figsize = (18,18))

sns.heatmap(x_norm.corr(),annot = True)

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size = 0.3,

                                                    random_state = 42)

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 100, random_state = 42)



rf.fit(x_train,y_train)



score = rf.score(x_test,y_test)



print("The score of the RandomForestClassifier is {}".format(score))
y_pred = rf.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_true,y_pred)



f,ax = plt.subplots(figsize = (5,5))



sns.heatmap(cm, annot = True)
from sklearn.metrics import classification_report

target_names = ["class 0","class 1"]



print(classification_report(y_true, y_pred, target_names=target_names))
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

score = dt.score(x_test, y_test)



print("Score of the Decision Three Classifier is {}".format(score))
y_pred = dt.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_true,y_pred)



f,ax = plt.subplots(figsize = (5,5))



sns.heatmap(cm, annot = True)


print(classification_report(y_true, y_pred, target_names=target_names))
from sklearn.svm import SVC

svm = SVC(random_state = 42)

svm.fit(x_train,y_train)

score = svm.score(x_test,y_test)

print("Score of the Support Vector Machine is {}".format(score))
y_pred = svm.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_true,y_pred)



f,ax = plt.subplots(figsize = (5,5))



sns.heatmap(cm, annot = True)


print(classification_report(y_true, y_pred, target_names=target_names))


cluster_data = x_norm[["engine_size", "horsepower"]]





from sklearn.cluster import KMeans

wcss = []



for k in range (1,15):

    kmeans = KMeans(n_clusters = k)

    kmeans.fit(cluster_data)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1,15), wcss)

plt.show()
kmeans2 = KMeans(n_clusters =3 )

clusters = kmeans2.fit_predict(cluster_data)



cluster_data["label"] = clusters



plt.scatter(cluster_data.engine_size[cluster_data.label == 0],

            cluster_data.horsepower[cluster_data.label == 0],

            color="red")

plt.scatter(cluster_data.engine_size[cluster_data.label == 1],

            cluster_data.horsepower[cluster_data.label == 1],

            color="purple")

plt.scatter(cluster_data.engine_size[cluster_data.label == 2],

            cluster_data.horsepower[cluster_data.label == 2],

            color="yellow")



plt.scatter(kmeans2.cluster_centers_[:,0],

            kmeans2.cluster_centers_[:,1])



plt.show()




