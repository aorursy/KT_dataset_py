# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing



import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load Iris dataset

iris=sns.load_dataset('iris')

iris

x_data = range(0, iris.shape[0])

y_data = iris['sepal_length']



fig, ax = plt.subplots()

ax.plot(x_data, y_data)

ax.set_title('Iris Dataset')

ax.grid()





plt.show()
x_data = range(0, iris.shape[0])

y_data = iris['sepal_length']



fig, ax = plt.subplots()

ax.plot(x_data, y_data)

ax.set_title('Iris Dataset')

ax.grid(color='r', linestyle='-', linewidth=1)



plt.show()
x_data = range(0, iris.shape[0])

y1_data = iris['petal_length']

y2_data = iris['petal_width']



fig, ax = plt.subplots()

ax.plot(x_data, y1_data)

ax.plot(x_data, y2_data)

ax.set_title('Iris Dataset')



ax.grid()

ax.legend()

plt.show()
x_data = range(0, iris.shape[0])

y1_data = iris['petal_length']

y2_data = iris['petal_width']



fig, ax = plt.subplots()



ax.set(xlabel='Samples',

       ylabel='Sepal',

       title='Iris Dataset')



ax.plot(x_data, y1_data,

        x_data, y2_data, "r")



ax.legend()

plt.show()

x_data = range(0, iris.shape[0])

y1_data = iris['sepal_length']

y2_data = iris['sepal_width']



plt.figure(1,figsize=(30,8))



plt.subplot(1,2,1) # 1 fila, 2 columnas, indice = 1

plt.plot(x_data, y1_data, '-', lw=2)

plt.title('Iris Dataset')

plt.legend()

plt.xlabel('Sample')

plt.ylabel('Length')

plt.title('Sepal Length')



plt.subplot(122) # equivalente a plt.subplot(1,2,2)

plt.plot(x_data, y2_data, '-', color='r', lw=2)

plt.xlabel('Sample')

plt.ylabel('Width')

plt.title('Sepal Width')



plt.grid(True)

plt.show()
data = iris['species'].value_counts()

points = data.index

frequency = data.values



fig, ax = plt.subplots(figsize=(8,6))



ax.bar(points, frequency)

ax.set_title('Iris Scores')

ax.set_xlabel('Species')

ax.set_ylabel('Frequency')



plt.show()





sns.countplot('species',data=iris)

plt.show()


y_1 = iris[iris['species'] == 'setosa']['sepal_length']

y_2 = iris[iris['species'] == 'versicolor']['sepal_length']

y_3 = iris[iris['species'] == 'virginica']['sepal_length']



y_4 = iris[iris['species'] == 'setosa']['petal_length']

y_5 = iris[iris['species'] == 'versicolor']['petal_length']

y_6 = iris[iris['species'] == 'virginica']['petal_length']



y_7 = iris[iris['species'] == 'setosa']['sepal_width']

y_8 = iris[iris['species'] == 'versicolor']['sepal_width']

y_9 = iris[iris['species'] == 'virginica']['sepal_width']





y_10 = iris[iris['species'] == 'setosa']['petal_width']

y_11 = iris[iris['species'] == 'versicolor']['petal_width']

y_12 = iris[iris['species'] == 'virginica']['petal_width']



#x_labels = iris['species'].unique()

x_labels = ['sepal length', 'sepal width', 'petal length', 'petal width']

x_pos = np.arange(4)



Y_1 = [y_1.mean(), y_4.mean(), y_7.mean(), y_10.mean()]

Y_2 = [y_2.mean(), y_5.mean(), y_8.mean(), y_11.mean()]

Y_3 = [y_3.mean(), y_6.mean(), y_9.mean(), y_12.mean()]





fig_size = [15,5]



plt.figure(figsize=fig_size)



plt.bar(x_pos-0.2, Y_1,color="r",width=0.2)

plt.bar(x_pos+0.2,Y_2,color="g",width=0.2)

plt.bar(x_pos, Y_3,color="b", width=0.2)





plt.title("Iris dataset")

plt.xticks(x_pos,x_labels);

plt.xlabel("Features")

plt.ylabel("Mean")

plt.legend(["setosa","versicolor", "virginica"])

plt.show();
df = sns.load_dataset('iris')

data1 = df.loc[df.species=='setosa', "sepal_length"]

data2 = df.loc[df.species=='virginica', "sepal_length"]

data3 = df.loc[df.species=='versicolor', "sepal_length"]



plt.subplots(figsize=(7,6), dpi=100)



sns.distplot( data1 , color="dodgerblue", label="Setosa")

sns.distplot( data2 , color="orange", label="virginica")

sns.distplot( data3 , color="deeppink", label="versicolor")



plt.title('Iris Histogram')

plt.legend();
data1 = iris['sepal_length']

data2 = iris['sepal_width']



# create a figure and axis

fig, ax = plt.subplots()



# scatter the sepal_length against the sepal_width

ax.scatter(data1, data2)



# set a title and labels

ax.set_title('Iris Dataset')

ax.set_xlabel('sepal_length')

ax.set_ylabel('sepal_width')



plt.show()
data1 = iris['sepal_length']

data2 = iris['sepal_width']

species = iris['species']



colors = {'setosa':'r', 'versicolor':'g', 'virginica':'b'}



fig, ax = plt.subplots()



for i in range(len(iris['sepal_length'])):

    ax.scatter(data1[i], data2[i],color=colors[species[i]])



ax.set_title('Iris Dataset')

ax.set_xlabel('sepal_length')

ax.set_ylabel('sepal_width')

plt.show()
sns.set_style("whitegrid");

sns.pairplot(iris, hue="species", height=3);

plt.show()


fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.boxplot(

    x='species', y='petal_length',

    data=iris, order=['virginica','versicolor','setosa'],

    linewidth=2.5, orient='v', dodge=False

)
data = iris['sepal_length']

            

ax = sns.distplot(data, kde=True)

# matriz de correlación

corr = iris.corr()



fig, ax = plt.subplots()



# creación del mapa de calor

im = ax.imshow(corr.values)



# etiquetas

ax.set_xticks(np.arange(len(corr.columns)))

ax.set_yticks(np.arange(len(corr.columns)))

ax.set_xticklabels(corr.columns)

ax.set_yticklabels(corr.columns)



# rotación de los textos

plt.setp(ax.get_xticklabels(), 

         rotation=45, ha="right",

         rotation_mode="anchor")



plt.show()
ax = sns.heatmap(iris.corr(), annot=True)
from sklearn import tree

from sklearn.datasets import load_iris

import graphviz 



dataset = load_iris()



clf = tree.DecisionTreeClassifier()

#clf = tree.DecisionTreeClassifier(random_state=0, max_depth=2)

clf = clf.fit(dataset.data, dataset.target)





dot_data = tree.export_graphviz(

    clf, out_file=None, 

    feature_names=dataset.feature_names,  

    class_names=dataset.target_names,  

    filled=True, rounded=True,      

    special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 