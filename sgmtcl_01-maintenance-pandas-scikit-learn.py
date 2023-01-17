# Import Libraries needed

import pandas as pd                 #dataframe manipulation

import numpy as np                  #numerical processing of vectors

import matplotlib.pyplot as plt     #plotting

%matplotlib inline                



#import tensorflow as tf

import sklearn

from sklearn import tree

import graphviz

import dask





print("Pandas:\t\t", pd.__version__)

print("Numpy:\t\t", np.__version__)

#print("Tensorflow:\t", tf.__version__)

print("Dask:\t\t", dask.__version__)

print("Scikit-learn:\t", sklearn.__version__)
df_init = df = pd.read_csv('../input/maintenance_data.csv')
df.columns
df.head()
df.tail()
df.describe()
df.sort_values(by='lifetime', ascending=True).head()
df.sort_values(by='lifetime', ascending=True).tail()
plt.bar(df.sort_values('team').team, df.sort_values('lifetime').lifetime)
df.groupby([df.team, df.broken]).count()
df.groupby(['team','broken']).agg({'broken': 'count'}).apply(lambda x:100 * x / float(x.sum()))
show_perc = df.groupby(['team','broken']).agg({'broken': 'count'})

show_perc.apply(lambda x:100 * x / float(x.sum()))
column = 'provider'

show_perc = df.loc[df['broken'] == 1].groupby([column]).agg({'broken': 'count'})

show_perc.apply(lambda x:round(100 * x / float(x.sum()),2)).rename(columns={"broken": "%"})
tree_data = df_init.drop('broken', axis=1)

tree_target = df_init.broken



#workaround replacement strings to integers - DO NOT DO IT LIKE THIS ;-)

try:

    tree_data.replace('TeamA',1, inplace=True)

    tree_data.replace('TeamB',2, inplace=True)

    tree_data.replace('TeamC',3, inplace=True)

    tree_data.replace('Provider1',1, inplace=True)

    tree_data.replace('Provider2',2, inplace=True)

    tree_data.replace('Provider3',3, inplace=True)

    tree_data.replace('Provider4',4, inplace=True)

except:

    pass  



#convert dataframes to arrays

tree_data = tree_data.values

tree_target = tree_target.values

#column names - labels

tree_feature_names = ['lifetime', 'pressureInd', 'moistureInd', 'temperatureInd', 'team', 'provider']

#target names - class

tree_target_names = ['BROKEN!','Operational']



#Tree Classifiers

tree_clf = tree.DecisionTreeClassifier()



#tree_clf.set_params(max_depth=3)



tree_clf = tree_clf.fit(tree_data, tree_target)



tree_clf.get_params()
#output graph tree

tree_dot_data = tree.export_graphviz(tree_clf, 

                                out_file=None, 

                                feature_names=tree_feature_names,

                                class_names=tree_target_names,

                                filled=True, 

                                rounded=True,

                                special_characters=True) 

graph = graphviz.Source(tree_dot_data) 

graph.render("Maintenance_classification_tree")

#show tree

graph
df_init.drop('broken', axis=1).columns
#PREDICTION WITHOUT REGRESSION - 1-->BROKEN 0-->Operational

for t in range(1,5):

    for p in range(1,5):

        print("team\tprov\tlife\tno\tyes")

        for i in range(10,110,10):

            arr = tree_clf.predict_proba([[float(i), 100., 100., 100., t, p]])

            print(t, "\t", p, "\t", i, "\t", round(arr[0][0], 2), "\t", round(arr[0][1], 2))