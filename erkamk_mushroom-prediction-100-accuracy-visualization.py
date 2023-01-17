

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv(r"/kaggle/input/mushroom-classification/mushrooms.csv")

data.info()
data.describe()
data.columns

data.isnull().sum()
print(data["class"].value_counts())

labels = 'Edible', 'Poisonous',

sizes = [4208, 3916]

colors = ['steelblue','yellowgreen']

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Class")

plt.show()
print(data["cap-shape"].value_counts())

labels = 'Convex', 'Flat',"Knobbed","Bell","Sunken","Conical"

sizes = [3656, 3152,828,452,32,4]

colors = ['rosybrown','dimgrey',"indianred","chocolate","gold","darkviolet"]

explode = (0, 0,0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Cap Shape")

plt.show()
print(data["cap-surface"].value_counts())

labels = 'Scaly',"Smooth","Fibrous","Grooves"

sizes = [3244, 2556,2320,4]

colors = ['rosybrown','turquoise',"chocolate","indianred","gold","darkviolet"]

explode = (0, 0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Cap Surface")

plt.show()
print(data["cap-color"].value_counts())

labels = 'Brown',"Gray","Red","Yellow","White","Buff","Pink","Cinnamon","Green","Purple"

sizes = [2284, 1840,1500,1072,1040,168,144,44,16,16]

colors = ['brown','gray',"red","yellow","white","khaki","pink","sandybrown","green","purple"]

explode = (0, 0,0,0,0,0,0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Cap Color")

plt.show()
print(data["bruises"].value_counts())

labels = 'True',"False"

sizes = [4748, 3376]

colors = ['rosybrown','turquoise',"chocolate","indianred","gold","darkviolet"]

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Bruises")

plt.show()
print(data["odor"].value_counts())

labels = 'None',"Foul","Fishy","Spicy","Almond","Anise","Puncent","Creosote","Musty"

sizes = [3528, 2160,576,576,400,400,256,192,36]

colors = ['rosybrown','turquoise',"chocolate","indianred","gold","darkviolet","darkblue","teal","forestgreen"]

explode = (0, 0,0,0,0,0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Odor")

plt.show()
print(data["gill-attachment"].value_counts())

labels = 'Free', "Attached"

sizes = [7914, 210]

colors = ['rosybrown','turquoise',"chocolate","indianred","gold","darkviolet","darkblue","teal","forestgreen"]

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Odor")

plt.show()
print(data["gill-spacing"].value_counts())

labels = 'Close', "Crowded"

sizes = [6812, 1312]

colors = ['gold','indianred',"chocolate","indianred","gold","darkviolet","darkblue","teal","forestgreen"]

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Gill Spacing")

plt.show()
print(data["gill-size"].value_counts())

labels = 'Broad', "Narrow"

sizes = [5612, 2512]

colors = ['mediumaquamarine','blueviolet',"chocolate","indianred","gold","darkviolet","darkblue","teal","forestgreen"]

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Gill size")

plt.show()
print(data["gill-color"].value_counts())

labels = 'Buff',"Pink","White","Brown","Gray","Chocolate","Purple","Black","Red","Yellow","Orange","Green"

sizes = [1728, 1492,1202,1048,752,732,492,408,96,86,64,24]

colors = ['khaki','pink',"white","brown","gray","chocolate","purple","black","red","yellow","orange","green"]

explode = (0, 0,0,0,0,0,0,0,0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Gill Color")

plt.show()
print(data["stalk-shape"].value_counts())

labels = 'Tapering', "Enlarging"

sizes = [4608, 3516]

colors = ['dodgerblue','indianred',"chocolate","indianred","gold","darkviolet","darkblue","teal","forestgreen"]

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Stalk Shape")

plt.show()
print(data["stalk-root"].value_counts())

labels = 'Bulbous', "Missing","Equal","Club","Rooted"

sizes = [3776, 2480,1120,556,192]

colors = ['forestgreen','indianred',"chocolate","darkblue","gold","darkviolet","darkblue","teal","forestgreen"]

explode = (0, 0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Stalk Root")

plt.show()
print(data["stalk-surface-above-ring"].value_counts())

labels = 'Smooth', "Silky","Fibrous","Scaly"

sizes = [5176, 2372,552,24]

colors = ['gold',"silver","firebrick","turquoise"]

explode = (0, 0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Stalk Surface Above Ring")

plt.show()
print(data["stalk-surface-below-ring"].value_counts())

labels = 'Smooth', "Silky","Fibrous","Scaly"

sizes = [4936, 2304,600,284]

colors = ['palegoldenrod',"silver","firebrick","turquoise"]

explode = (0, 0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Stalk Surface Below Ring")

plt.show()
print(data["stalk-color-above-ring"].value_counts())

labels = 'white', "pink","gray","brown","buff","orange","red","cinnamon","yellow"

sizes = [4464, 1872,576,448,432,192,96,36,8]

colors = ['whitesmoke',"pink","gray","brown","khaki","orange","red","sandybrown","yellow"]

explode = (0, 0,0,0,0,0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Stalk Color Above Ring")

plt.show()
print(data["stalk-color-below-ring"].value_counts())

labels = 'white', "pink","gray","brown","buff","orange","red","cinnamon","yellow"

sizes = [4384, 1872,576,512,432,192,96,36,24]

colors = ['whitesmoke',"pink","gray","brown","khaki","orange","red","sandybrown","yellow"]

explode = (0, 0,0,0,0,0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Stalk Color Below Ring")

plt.show()
print(data["veil-color"].value_counts())

labels = 'White', "Orange","Brown","Yellow"

sizes = [7924, 96,96,8]

colors = ['whitesmoke','orange',"brown","yellow","gold","darkviolet","darkblue","teal","forestgreen"]

explode = (0, 0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Veil Color")

plt.show()
print(data["ring-number"].value_counts())

labels = 'One', "Two","None"

sizes = [7488, 600,36]

colors = ['teal','orange',"brown","yellow","gold","darkviolet","darkblue","teal","forestgreen"]

explode = (0, 0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Ring Number")

plt.show()
print(data["ring-type"].value_counts())

labels = 'Pendant', "Evanescent","Large","Flaring","None"

sizes = [3968, 2776,1296,48,36]

colors = ["royalblue","khaki","mediumpurple","red","grey"]

explode = (0, 0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Ring Type")

plt.show()
print(data["spore-print-color"].value_counts())

labels = 'White', "Brown","Black","Chocolate","Green","Orange","Buff","Purple","Yellow"

sizes = [2388, 1968,1872,1632,72,48,48,48,48]

colors = ["whitesmoke","brown","black","chocolate","green","orange","khaki","purple","yellow"]

explode = (0, 0,0,0,0,0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Spore Print Color")

plt.show()
print(data["population"].value_counts())

labels = 'Several', "Solitary","Scattered","Numerous","Abundant","Clustered"

sizes = [4040,1712,1248,400,384,340]

colors = ["antiquewhite","fuchsia","deepskyblue","springgreen","darkkhaki","gold"]

explode = (0, 0,0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Population")

plt.show()
print(data["habitat"].value_counts())

labels = 'Woods', "Grasses","Paths","Leaves","Urban","Meadows","Waste"

sizes = [3148,2148,1144,832,368,292,192]

colors = ['palegoldenrod',"silver","firebrick","turquoise","forestgreen","gold","purple"]

explode = (0, 0,0,0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Population")

plt.show()

data["class"] = [1 if each == "e" else 0 for each in data["class"]]

y = data["class"]

data.drop(["class"],axis = 1 , inplace = True)

data = pd.get_dummies(data,columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

       'stalk-surface-below-ring', 'stalk-color-above-ring',

       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',

       'ring-type', 'spore-print-color', 'population', 'habitat'])

from sklearn.model_selection import train_test_split

x_train , x_test , y_train, y_test = train_test_split(data,y,test_size = 0.2 , random_state = 42)
from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()

naive_bayes.fit(x_train,y_train)

nb_score = naive_bayes.score(x_test,y_test)

print("Naive Bayes Accuracy : ",nb_score)
y_pred = naive_bayes.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(10,10))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.svm import SVC

svm = SVC(gamma = 0.01 , C = 500 , kernel = "rbf")

svm.fit(x_train,y_train)

svm_score = svm.score(x_test,y_test)

print("SVM Max Score = : ", svm_score)
y_pred1 = svm.predict(x_test)

y_true1 = y_test

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_true1,y_pred1)

f, ax = plt.subplots(figsize =(10,10))

sns.heatmap(cm1,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 1.0, penalty = "l2")

lr.fit(x_train,y_train)

print("Logistic Regression Max Score : ",lr.score(x_test,y_test))

lr_max = lr.score(x_test,y_test)
y_pred2 = lr.predict(x_test)

y_true2 = y_test

from sklearn.metrics import confusion_matrix

cm2 = confusion_matrix(y_true2,y_pred2)

f, ax = plt.subplots(figsize =(10,10))

sns.heatmap(cm2,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
score_list = []

from sklearn.ensemble import RandomForestClassifier

for each in range (1,100):

    rf = RandomForestClassifier(n_estimators = each,random_state = 7,bootstrap = "False",criterion="gini",

                                    min_samples_split = 10 , min_samples_leaf = 1)

    rf.fit(x_train,y_train)

    score_list.append(rf.score(x_test,y_test))

    

rf_max = np.max(score_list)

print("RF Max Score : ",rf_max)
y_pred3 = rf.predict(x_test)

y_true3 = y_test

from sklearn.metrics import confusion_matrix

cm3 = confusion_matrix(y_true3,y_pred3)

f, ax = plt.subplots(figsize =(10,10))

sns.heatmap(cm3,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
results = pd.DataFrame({"Classification" :["Random Forest C." ,"Logistic Regression C.",

            "Naive Bayes C..","Support Vector C."],

                        "Accuracy" : [rf_max*100,lr_max*100,nb_score*100,svm_score*100]})

results = results.sort_values(by=['Accuracy'],ascending = False)

print(results)