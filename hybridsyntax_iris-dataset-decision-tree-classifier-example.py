import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

%matplotlib inline
iris_ds = load_iris()

iris_df = pd.DataFrame(iris_ds.data, columns=iris_ds.feature_names)
iris_df.shape
iris_df.head()
iris_df.info()
iris_df['petal width (cm)'].hist()
iris_df.describe()
plt.title("Proportion of each species")

plt.pie(np.bincount(iris_ds.target), 

        labels=iris_ds.target_names, 

        autopct='%1.1f%%',

        shadow=True, startangle=90);
fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4) 

fig.set_size_inches(18, 5)



ax1.boxplot(iris_df['sepal length (cm)']);

ax1.set_title('sepal length (cm)');



ax2.boxplot(iris_df['sepal width (cm)']);

ax2.set_title('sepal width (cm)');



ax3.boxplot(iris_df['petal length (cm)']);

ax3.set_title('petal length (cm)');



ax4.boxplot(iris_df['petal width (cm)']);

ax4.set_title('petal width (cm)');
fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4) 

fig.set_size_inches(18, 5)



ax1.hist(iris_df['sepal length (cm)']);

ax1.set_title('sepal length (cm)');



ax2.hist(iris_df['sepal width (cm)']);

ax2.set_title('sepal width (cm)');



ax3.hist(iris_df['petal length (cm)']);

ax3.set_title('petal length (cm)');



ax4.hist(iris_df['petal width (cm)']);

ax4.set_title('petal width (cm)');
import matplotlib.pyplot as plt

%matplotlib inline

# Visualize the data sets

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)

for target, target_name in enumerate(iris_ds.target_names):

    X_plot = iris_ds.data[iris_ds.target == target]

    plt.plot(X_plot[:, 0], X_plot[:, 1], linestyle='none', marker='o', label=target_name)

plt.xlabel(iris_ds.feature_names[0])

plt.ylabel(iris_ds.feature_names[1])

plt.axis('equal')

plt.legend();



plt.subplot(1, 2, 2)

for target, target_name in enumerate(iris_ds.target_names):

    X_plot = iris_ds.data[iris_ds.target == target]

    plt.plot(X_plot[:, 2], X_plot[:, 3], linestyle='none', marker='o', label=target_name)

plt.xlabel(iris_ds.feature_names[2])

plt.ylabel(iris_ds.feature_names[3])

plt.axis('equal')

plt.legend();
#Parameters



train_size = 0.75                  #float or int, default=None

test_size= 1- train_size           #float or int, default=None

random_state=None                  #int or RandomState instance

shuffle=True                       #bool

stratify=None                      #array-like



#Splitting test and train data

X_train, X_test, y_train, y_test = train_test_split(iris_ds.data,

                                                    iris_ds.target,

                                                   train_size=train_size,

                                                   test_size=test_size,

                                                   random_state=random_state,

                                                   shuffle=shuffle,

                                                   stratify=stratify)
#Parameters 



criterion='gini'                            # {“gini”, “entropy”},

splitter='best'                             # {“best”, “random”}

max_depth=None                              # int

min_samples_split =2                        # int or float

min_samples_leaf=1                          # int or float

min_weight_fraction_leaf=0.0                # float

max_features=None                           # int, float or {“auto”, “sqrt”, “log2”}

random_state=None                           # int, RandomState instance

max_leaf_nodes=None                         # int

min_impurity_decrease = 0.0                 # float

class_weight=None                           # dict, list of dict or “balanced”

ccp_alpha=0.0                               # non-negative float, default=0.0



#Creating model

classifier =  DecisionTreeClassifier(criterion=criterion,

                                     splitter=splitter,

                                    max_depth=max_depth,

                                     min_samples_split=min_samples_split,

                                    min_samples_leaf=min_samples_leaf,

                                    max_features = max_features,

                                    random_state = random_state,

                                    max_leaf_nodes = max_leaf_nodes,

                                    min_impurity_decrease  = min_impurity_decrease ,

                                    class_weight = class_weight,

                                    ccp_alpha = ccp_alpha)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_test,y_test)
#Parameters

labels=[0,1,2]                      #array-like of shape (n_classes)

sample_weight=None                  #array-like of shape (n_samples,)

normalize=None                      #{‘true’, ‘pred’, ‘all’}



confusion_matrix(y_pred,y_test,

                labels=labels,

                sample_weight=sample_weight,

                normalize=normalize)
#Parameters

labels=[0,1,2]                        #[n_labels]

target_names= iris_ds.target_names    #list of strings

sample_weight=None                    #array-like of shape (n_samples,)

digits=2                              #int

output_dict= False                    #bool

zero_division="warn"                  #“warn”, 0 or 1



print(classification_report(y_pred,

                            y_test,

                            labels=labels,

                            target_names=target_names,

                            sample_weight=sample_weight,

                            digits=digits,

                            output_dict=output_dict,

                            zero_division=zero_division))
fig, ax = plt.subplots(figsize=(15, 15)) 



#Paramters

decision_tree=classifier             #decision tree regressor or classifier

filled = True                        #bool, optional (default=False)

feature_names=iris_ds.feature_names  #list of strings, optional (default=None)

class_names=iris_ds.target_names     #list of strings, bool or None, optional (default=None)

max_depth=None                       #int, optional (default=None)

fontsize=10                          #int, optional (default=None)

ax=ax                                #matplotlib axis, optional (default=None)

label='all'                          #{‘all’, ‘root’, ‘none’}, optional

impurity=True                        #bool, optional 

node_ids=False                       #bool, optional 

proportion=False                     #bool, optional

rounded=True                         #bool, optional (default=False)

precision=3                          #int, optional



plot_tree(decision_tree=decision_tree,

          filled=filled,

          feature_names=feature_names, 

          class_names=class_names,

          fontsize=fontsize,

          ax=ax,

          max_depth=max_depth,

          label=label,

          impurity=impurity,

          node_ids=node_ids,

          proportion=proportion,

          rounded=rounded,

          precision=precision);