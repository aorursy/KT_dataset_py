import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as mpatches
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, confusion_matrix, precision_score, recall_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import graphviz 
import warnings
import os

plt.rcParams["axes.labelsize"] = 16.
plt.rcParams["xtick.labelsize"] = 14.
plt.rcParams["ytick.labelsize"] = 14.
plt.rcParams["legend.fontsize"] = 12.
plt.rcParams["figure.figsize"] = [15, 6]

warnings.filterwarnings('ignore')
#auxiliary functions we're going to use later
def print_metrics(df):
    print("total: {}".format(len(df)))
    
    groups = df.groupby('Species').size()
    
    setosa = groups['Iris-setosa'] if 'Iris-setosa' in groups else 0
    versicolor = groups['Iris-versicolor'] if 'Iris-versicolor' in groups else 0
    virginica = groups['Iris-virginica'] if 'Iris-virginica' in groups else 0
    
    print("Iris-setosa: {}".format(setosa))
    print("Iris-versicolor: {}".format(versicolor))
    print("Iris-virginica: {}".format(virginica))
    
def merge_left_only(left, right):
    merged = left.merge(right, how='left', indicator=True)
    result = merged[merged['_merge']=='left_only']
    return result.drop(columns=['_merge'])

def show_scatter(data, graph_type): #graph_type values: 'Petal' or 'Sepal'
    fig = data[data.Species=='Iris-versicolor'].plot(kind='scatter',x=graph_type + 'LengthCm',y=graph_type + 'WidthCm',color='orange', label='versicolor')
    data[data.Species=='Iris-virginica'].plot(kind='scatter',x=graph_type + 'LengthCm',y=graph_type+'WidthCm',color='blue', label='virginica',ax=fig)
    fig.set_xlabel(graph_type + " Length")
    fig.set_ylabel(graph_type + " Width")
    fig.set_title(graph_type + " Length VS Width")
    fig=plt.gcf()
    fig.set_size_inches(5,3)
    return plt
iris = pd.read_csv('../input/Iris.csv')
iris.drop('Id',axis=1,inplace=True)
iris['PetalSurface'] = iris.apply(lambda row: row.PetalLengthCm * row.PetalWidthCm, axis=1)
iris.head(1)
df_train, df_test = train_test_split(iris, test_size=0.30)
df = df_train
df.groupby(['Species']).mean()
fig = df[df.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
df[df.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
df[df.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
fig = df[df.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
df[df.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
df[df.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
h1 = df[(df.PetalLengthCm<2.5)]
print_metrics(h1)
setosa = h1
sg = merge_left_only(df,h1)#from the original dataset we remove all flowers from h1
show_scatter(sg,'Petal').show()
show_scatter(sg,'Sepal').show()
h2 = sg[sg.SepalLengthCm>7]
print_metrics(h2)
virginica = h2
sg = merge_left_only(sg,h2)
show_scatter(sg,'Petal').show()
show_scatter(sg,'Sepal').show()
h3 = sg[sg.SepalWidthCm<2.5]
print_metrics(h3)
versicolor = h3
sg = merge_left_only(sg,h3)
show_scatter(sg,'Petal').show()
show_scatter(sg,'Sepal').show()
h4 = sg[sg.PetalWidthCm < 1.6]
print_metrics(h4)
versicolor = pd.concat([versicolor,h4])
sg = merge_left_only(sg,h4)
show_scatter(sg,'Petal').show()
show_scatter(sg,'Sepal').show()
h5 = sg[(sg.PetalLengthCm > 5.2) & (sg.PetalWidthCm>1.8)]
print_metrics(h5)
virginica = pd.concat([virginica, h5])
sg = merge_left_only(sg,h5)
show_scatter(sg,'Petal').show()
show_scatter(sg,'Sepal').show()
h6 = sg[sg.PetalSurface>8.7]
print_metrics(h6)
virginica = pd.concat([virginica, h6])
sg = merge_left_only(sg,h6)
show_scatter(sg,'Petal').show()
show_scatter(sg,'Sepal').show()
versicolor = pd.concat([versicolor, sg])
# This function uses the same cuts we saw before
def by_hand_prediction(row):
    if row.PetalLengthCm<2.5:
        return 'Iris-setosa'
    elif row.SepalLengthCm>7:
        return 'Iris-virginica'
    elif row.SepalWidthCm<2.5:
        return 'Iris-versicolor'
    elif row.PetalWidthCm < 1.6:
        return 'Iris-versicolor'
    elif row.PetalLengthCm > 5.2 and row.PetalWidthCm>1.8:
        return 'Iris-virginica'
    elif row.PetalSurface>8.7:
        return 'Iris-virginica'
    else:
        return 'Iris-versicolor'
    
# Let's see how we compare with a DecisionTree model
clf = tree.DecisionTreeClassifier()
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
clf = clf.fit(df_train[features], df_train.Species)

dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=features,  
                      class_names=['setosa','virginica','versicolor'],  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data) 
graph
df_test['ModelPredicted'] = clf.predict(df_test[features])
df_test['ByHandPredicted'] = df_test.apply(lambda row: by_hand_prediction(row),axis=1)
df_test[df_test.ModelPredicted!=df_test.Species]
df_test[df_test.ByHandPredicted!=df_test.Species]