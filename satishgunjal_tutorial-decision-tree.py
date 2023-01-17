import pandas as pd

import numpy as np

from sklearn import datasets

from sklearn import model_selection

from sklearn import tree

import graphviz
iris = datasets.load_iris()

print('Dataset structure= ', dir(iris))



df = pd.DataFrame(iris.data, columns = iris.feature_names)

df['target'] = iris.target

df['flower_species'] = df.target.apply(lambda x : iris.target_names[x]) # Each value from 'target' is used as index to get corresponding value from 'target_names' 



print('Unique target values=',df['target'].unique())



df.sample(5)
# label = 0 (setosa)

df[df.target == 0].head(3)
# label = 1 (versicolor)

df[df.target == 1].head(3)
# label = 2 (verginica)

df[df.target == 2].head(3)
#Lets create feature matrix X  and y labels

X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

y = df[['target']]



print('X shape=', X.shape)

print('y shape=', y.shape)
X_train,X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state= 1)

print('X_train dimension= ', X_train.shape)

print('X_test dimension= ', X_test.shape)

print('y_train dimension= ', y_train.shape)

print('y_train dimension= ', y_test.shape)
"""

To obtain a deterministic behaviour during fitting always set value for 'random_state' attribute

Also note that default value of criteria to split the data is 'gini'

"""

cls = tree.DecisionTreeClassifier(random_state= 1)

cls.fit(X_train ,y_train)
print('Actual value of species for 10th training example=',iris.target_names[y_test.iloc[10]][0])

print('Predicted value of species for 10th training example=', iris.target_names[cls.predict([X_test.iloc[10]])][0])



print('\nActual value of species for 20th training example=',iris.target_names[y_test.iloc[20]][0])

print('Predicted value of species for 20th training example=', iris.target_names[cls.predict([X_test.iloc[20]])][0])



print('\nActual value of species for 30th training example=',iris.target_names[y_test.iloc[29]][0])

print('Predicted value of species for 30th training example=', iris.target_names[cls.predict([X_test.iloc[29]])][0])
cls.score(X_test, y_test)
tree.plot_tree(cls) 
dot_data = tree.export_graphviz(cls, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("iris_decision_tree") 
dot_data = tree.export_graphviz(cls, out_file=None, 

                      feature_names=iris.feature_names,  

                      class_names=iris.target_names,  

                      filled=True, rounded=True,  

                      special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
boston = datasets.load_boston()

print('Dataset structure= ', dir(boston))



df = pd.DataFrame(boston.data, columns = boston.feature_names)

df['target'] = boston.target



df.sample(5)
#Lets create feature matrix X  and y labels

X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]

y = df[['target']]



print('X shape=', X.shape)

print('y shape=', y.shape)
X_train,X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state= 1)

print('X_train dimension= ', X_train.shape)

print('X_test dimension= ', X_test.shape)

print('y_train dimension= ', y_train.shape)

print('y_train dimension= ', y_test.shape)
"""

To obtain a deterministic behaviour during fitting always set value for 'random_state' attribute

To keep the tree simple I am using max_depth = 3

Also note that default value of criteria to split the data is 'mse' (mean squared error)

mse is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node

"""

dtr = tree.DecisionTreeRegressor(max_depth= 3,random_state= 1)

dtr.fit(X_train ,y_train)
predicted_price= pd.DataFrame(dtr.predict(X_test), columns=['Predicted Price'])

actual_price = pd.DataFrame(y_test, columns=['target'])

actual_price = actual_price.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe

df_actual_vs_predicted = pd.concat([actual_price,predicted_price],axis =1)

df_actual_vs_predicted.T
dtr.score(X_test, y_test)
tree.plot_tree(dtr) 
dot_data = tree.export_graphviz(dtr, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("boston_decision_tree") 
dot_data = tree.export_graphviz(dtr, out_file=None, 

                      feature_names=boston.feature_names,  

                      filled=True, rounded=True,  

                      special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 