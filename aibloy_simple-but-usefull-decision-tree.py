#Library

import matplotlib.pyplot as plt

import pandas as pd 

import numpy as np 
#Data

df = pd.read_csv("../input/iris/Iris.csv")

df = df.drop("Id", axis=1)

df.head(5)
# Preprocessed data

X = df.drop("Species",axis=1)

Y = df["Species"]

X_columns = X.columns



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
#MODEL

from sklearn import tree



decision_tree = tree.DecisionTreeClassifier()

decision_tree = decision_tree.fit(X_train,Y_train)

Y_pred = decision_tree.predict(X_test)



decision_tree.score(X_test,Y_test)

#EVALUATE THE MODEL

from sklearn.model_selection import cross_val_score

scores = cross_val_score(decision_tree, X_train, Y_train,cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)



cm = pd.DataFrame(cm)

cm.columns=Y.unique()

cm.index = Y.unique()

cm
# GRAPH MODEL

plt.figure(figsize=(15,10))

tree.plot_tree(decision_tree, filled=True, class_names=Y.unique(), feature_names=X_columns, impurity=False)

plt.show()
# PetalWidthCm < 0.75

df[df["PetalWidthCm"]<=0.75]["Species"].unique()
#Importance of each variable

feat_importances = pd.Series(decision_tree.feature_importances_, index=X_columns)

feat_importances.nlargest(5).plot(kind='barh', color="slategrey")

plt.xlabel('Importancia de las características')

plt.ylabel('Características')

plt.show()