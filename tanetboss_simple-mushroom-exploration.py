%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from pylab import rcParams

rcParams['figure.figsize'] = 10, 15

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings('ignore')

mushrooms = pd.read_csv('../input/mushrooms.csv')
mushrooms.head()
mushrooms.info()
mushrooms.describe()
mushrooms.isnull().sum()
Labledmushrooms = pd.DataFrame()



for col in mushrooms.columns:

    Labledmushrooms[col] = LabelEncoder().fit_transform(mushrooms[col])



Labledmushrooms.head()
with plt.xkcd():

#correlation map

    f,ax = plt.subplots(figsize=(18, 18))

    g = sns.heatmap(Labledmushrooms.corr(), annot=True, linewidths=.5, fmt= '.1f',cmap = "coolwarm",ax=ax)
sorted(mushrooms['bruises'].unique())
g = sns.factorplot(x="bruises",y="class",data=Labledmushrooms,kind="bar", size = 6,  palette =["lavender","darkslateblue"], legend = True)

g.despine(left=True)



g.set(xticks=range(0,2), xticklabels=["not-bruise","bruise"])

g = g.set_ylabels("poisonous probability")
sorted(mushrooms['gill-color'].unique())
gillcolor = ["khaki","Red","lightGrey","chocolate","Black","saddleBrown","orange","lightpink","limegreen","orchid","whitesmoke","Yellow"]

gillname =["buff","red","gray","chocolate","black","brown","orange","pink","green","purple","white","yellow"]
counts = mushrooms['gill-color'].value_counts(sort = True)

counts


labels = ["buff","pink","white","brown","gray","chocolate","purple","black","red","yellow","orange","green"]

sizes = counts

colors = ["khaki","pink","whitesmoke","saddlebrown","lightgrey","chocolate","orchid","black","red","yellow","orange","limegreen"]

explode = (0.1,0.05, 0.05, 0,0,0,0,0,0,0,0,0)

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140,)

plt.axis('equal')

plt.figtext(.5,.8,'Mushroom gill-color',fontsize=30,ha='center')

plt.show()
g = sns.factorplot(x="gill-color",y="class",data=Labledmushrooms, kind="bar", size = 10 ,

palette = gillcolor)

g.despine(left=True)

g.set_xticklabels(rotation=30)

g.set( xticks=range(0,12),xticklabels=gillname)

g = g.set_ylabels("poisonous probability")
mushrooms[mushrooms['gill-color'] == 'e'].count().unique()
mushrooms[mushrooms['gill-color'] == 'e'].head()
g = sns.factorplot("gill-color", col="bruises",  data=Labledmushrooms,

                   size=6, kind="count", palette =gillcolor)

g.despine(left=True)



g.set( xticks=range(0,12),xticklabels=gillname)

g.set_xticklabels(rotation=30)

g = g.set_ylabels("Count")
sorted(mushrooms['stalk-root'].unique())
g = sns.factorplot(x="stalk-root",y="class",data=Labledmushrooms,kind="bar", size = 6,  palette ="cubehelix" )

g.despine(left=True)

g.set(xticks=range(0,5), xticklabels=["missing","bulbous","club","equal","rooted"])

g = g.set_ylabels("poisonous probability")
sorted(mushrooms['ring-type'].unique())
g = sns.factorplot(x="ring-type",y="class",data=Labledmushrooms,kind="bar", size = 6,  palette ="BrBG" )

g.despine(left=True)

g.set(xticks=range(0,5), xticklabels=["evanescent","flaring","large","none","pendant"])

g = g.set_ylabels("poisonous probability")
g = sns.factorplot("ring-type", col="bruises",  data=Labledmushrooms,

                   size=6, kind="count", palette="muted")

g.despine(left=True)

g.set(xticks=range(0,5), xticklabels=["evanescent","flaring","large","none","pendant"])

g = g.set_ylabels("Count")
sorted(mushrooms['gill-size'].unique())
g = sns.factorplot("gill-size",  data=Labledmushrooms,

                   size=6, kind="count", palette ="BrBG" )

g.despine(left=True)



g.set( xticks=range(0,2), xticklabels=["broad","narrow"])

g.set_xticklabels(rotation=30)

g = g.set_ylabels("Count")
g = sns.factorplot(x="gill-size",y="class",data=Labledmushrooms,kind="bar", size = 6,  palette =["thistle","darkviolet"] )

g.despine(left=True)

g.set(xticks=range(0,2), xticklabels=["broad","narrow"])

g = g.set_ylabels("poisonous probability")
labels =["edible","poisonous"]

sizes = Labledmushrooms['class'].value_counts(sort = True)

colors = ["peru","mediumorchid"]

explode = (0.1,0)  

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140,)

rcParams['figure.figsize'] = 13,10

plt.axis('equal')

plt.figtext(.5,.9,'Mushrooms',fontsize=30,ha='center')

plt.show()
Labledmushrooms['is_train'] = np.random.uniform(0, 1, len(Labledmushrooms)) <= .75



# View the top 5 rows

Labledmushrooms.head()
# Create two new dataframes, one with the training rows, one with the test rows

train, test = Labledmushrooms[Labledmushrooms['is_train']==True], Labledmushrooms[Labledmushrooms['is_train']==False]



train["class"] = train["class"].astype(int)



Y_train = train["class"]

X_train = train.drop(labels = ["class"],axis = 1)

Y_test = test["class"]

X_test = test.drop(labels = ["class"],axis = 1)
print(len(train))
print(len(test))
from sklearn.neighbors import KNeighborsClassifier



train_acc = []

test_acc = []



# try n_neighbors from 1 to 10

n_range = range(1, 11)



for neighbors in n_range:

    

    clf = KNeighborsClassifier(n_neighbors= neighbors)

    clf.fit(X_train, Y_train)



    train_acc.append(clf.score(X_train, Y_train))

    test_acc.append(clf.score(X_test, Y_test))

    

plt.plot(n_range, train_acc, label="training accuracy")

plt.plot(n_range, test_acc, label="test accuracy")



plt.ylabel("Accuracy")

plt.xlabel("neighbors")

plt.legend()
