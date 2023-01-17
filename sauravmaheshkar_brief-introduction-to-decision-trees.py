## Importing Libraries

from sklearn.datasets import load_iris

import matplotlib.pyplot as plt



## Loading the Dataset

iris = load_iris(return_X_y = True, as_frame= True)

dataset = iris[0]

target = iris[1]



## Plotting Histogram

dataset.hist(edgecolor='black', linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
max_depth = 5

min_samples_split = 2 

criterion = "gini"  

splitter = "best" 



from sklearn.tree import DecisionTreeClassifier



X, y = load_iris(return_X_y=True)

clf = DecisionTreeClassifier(

    max_depth=max_depth,

    min_samples_split=min_samples_split,

    criterion=criterion,

    splitter=splitter

)

clf = clf.fit(X,y)
from sklearn import tree

%matplotlib inline

fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']

cn=['setosa', 'versicolor', 'virginica']

fig = plt.figure(figsize=(20,15))

_ = tree.plot_tree(clf, 

                   feature_names=fn,  

                   class_names=cn,

                   filled=True)
max_depth = 5

min_samples_split = 2  

criterion = "mse"  

splitter = "best" 



from sklearn.tree import DecisionTreeRegressor



X, y = load_iris(return_X_y=True)

reg = DecisionTreeRegressor(

    max_depth=max_depth,

    min_samples_split=min_samples_split,

    criterion=criterion,

    splitter=splitter

)

reg = reg.fit(X,y)
%matplotlib inline

tree.plot_tree(reg, filled = True)

plt.show()