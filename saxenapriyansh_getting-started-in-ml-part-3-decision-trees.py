# Supervised learning

# Classification

# 1. Naive Bias            - Simple

# 2. Logistic regression   - Simple

# 3. Decision tree         - Complex

# 4. Random Forest         - Large amount of data
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import accuracy_score

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
iris = pd.read_csv("../input/Iris.csv")

print("Type: "+str(type(iris)))

print("Names of column fields: \n"+str(iris.columns))

print("\n Shape of dataset: "+str(iris.shape))

print("Head of dataset: ")

print(iris.head())
print("\n Brief description of data: ")

print(iris.describe())
# Shuffling data

from sklearn.utils import shuffle

iris = shuffle(iris)
# Seperating target and features

X = iris.iloc[:,1:5].values

y = iris.iloc[:,5].values
# Scaling and normalization

X_scale = (X-X.mean(0))/X.std(0)
# Encoding categorical data

labelencoder_y = LabelEncoder()

y_encode = labelencoder_y.fit_transform(y)



print("Name of classes: "+str(labelencoder_y.classes_))

print("Classes: "+str(np.unique(y_encode)))
print(y.reshape(150,1).shape)

print(X.shape)
# Using pairplot available in the seaborn package to visualize data

data = np.hstack((X,y.reshape(150,1)))

columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']

sns.pairplot(pd.DataFrame(data= data, columns= columns),hue="Species")
X_train, X_test, y_train, y_test = train_test_split(X_scale, y_encode,test_size=0.3)
###### use gridsearch #######

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='entropy',random_state=14,

                            splitter='best', max_depth=4, max_features=4, max_leaf_nodes=4)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(score)
from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_test, y_pred)



from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, y_pred),

                                colorbar=True,

                                show_absolute=False,

                                show_normed=True)

plt.xticks(np.arange(0,3,1),labelencoder_y.classes_,rotation=90,size=15)

plt.yticks(np.arange(0,3,1),labelencoder_y.classes_,size=15)

plt.xlabel('Predicted label',color='red',fontsize=20)

plt.ylabel('True label',color='red',fontsize=20)

plt.show()
print(list(zip(iris.columns[1:5].values, clf.feature_importances_)))

plt.barh(iris.columns[1:5].values, clf.feature_importances_)
# To graphicallyt show the decision trees

import graphviz

from sklearn.tree import export_graphviz

dot_data = export_graphviz(clf, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("iris")