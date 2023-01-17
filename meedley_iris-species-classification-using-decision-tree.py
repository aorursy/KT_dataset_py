import matplotlib.pyplot as plt
from pandas import read_csv
import seaborn as sns
import pandas as pd
import numpy as np

idf = read_csv('/kaggle/input/iris/Iris.csv')
print(idf.shape)
idf.info()
print(idf.Species.unique())
sns.countplot(x=idf.Species);
#drop Species and Id from the dataframe
ylabel = idf['Species']
print(ylabel.unique())
x = idf.drop(['Species', 'Id'],axis=1)
ylabel = ylabel.astype('category')
y = ylabel.cat.codes #save label code as y variable
#Center and scale the data
feature_mean = x.mean()
feature_std = x.std()
x = (x - feature_mean)/feature_std
print(np.sign(x.groupby(y).mean()-x.groupby(y).median()))
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for col, axs in zip(x.columns, axes.flatten()):
    sns.boxplot(ylabel, x[col],ax=axs)
    axs.set_title(col)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for col, axs in zip(x.columns, axes.flatten()):
    sns.violinplot(ylabel, x[col],ax=axs, inner="point")
    axs.set_title(col)
from numpy.linalg import matrix_rank
print(matrix_rank(x))
sns.heatmap( x.corr().abs(),annot=True);
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn import tree
#I experiemented with split criteria and method, and ended up using the default values.
clf = tree.DecisionTreeClassifier(criterion='gini',splitter='best')
clf = clf.fit(x_train,y_train)
scores = cross_val_score(clf,x,y)
print("Cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

#Predict species
y_pred = clf.predict(x_test)
print("Accuracy: %0.2f" % accuracy_score(y_test, y_pred))
print('Precision: %0.2f' % precision_score(y_test,y_pred, average='micro'))
print('Recall: %0.2f' % recall_score(y_test,y_pred, average='micro'))

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap="Reds", xticklabels=ylabel.unique(),yticklabels=ylabel.unique());