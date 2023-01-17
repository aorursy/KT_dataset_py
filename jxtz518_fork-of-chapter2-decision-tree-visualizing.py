#import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import tree
import collections
import graphviz 
#Import the dataset
churn_data = pd.read_csv('../input/churn_data.csv')
churn_data.head() #Printing first 5 rows of the dataset
# Define X, y, which are IV and DV (Independent variable and dependent variable)
X=churn_data[['couponDiscount','purchaseValue','giftwrapping','prodSecondHand']]
y=churn_data['returnCustomer']
# Sklearn is a very popular package in maching learning, we will often use this
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#Tell the model there are two categorical variables: title, paymentMethod
X=pd.get_dummies(X)

#Set 50% of the data as test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

#Decision tree
Tree = DecisionTreeClassifier(random_state=0,class_weight='balanced')
Model = Tree.fit(X_train, y_train)
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
data_feature_names = [ 'couponDiscount','purchaseValue','giftwrapping','prodSecondHand' ]
class_names = ['Churn', 'Not Churn']
dot_data = tree.export_graphviz(Model ,
                                feature_names=data_feature_names,
                                out_file=None,
                                class_names=class_names,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

Image(graph.create_png())
