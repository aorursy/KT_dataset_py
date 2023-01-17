import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

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
# from above plot we can easily arrive to following conclusions

# 1. from the plot of SepalWidthCm vs PetalWidthCm we can see a clear seperation between Setosa and other #species based on PetalWidth

# 2. from the plot of SepalWidthCm vs SepalLengthCm there is ALMOST  boundary between versicolor and verginica and a clear boundary between setosa and other two species based on the first obeservation we can with 100 certainity and accuracy predict whether specie is setosa or not, based on second obervation we can with high accuracy predict the specie.
X_train, X_test, y_train, y_test = train_test_split(X_scale, y_encode,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=40, criterion='entropy')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
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
temp = []

for i in range(3,100,2):

    clf = RandomForestClassifier(n_estimators = i, criterion='entropy', max_depth=150, random_state=15)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    temp.append(accuracy_score(y_test, y_pred))

plt.plot(np.arange(3,100,2), temp, color='red')