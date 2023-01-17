import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.listdir('../input/')
data_path = "../input/Tomato_Pixel_DataSet.csv"
data_raw = pd.read_csv(data_path)
print("Number of rows in data =",data_raw.shape[0])
print("Number of columns in data =",data_raw.shape[1])
print("\n")
print("**Sample data:**")
data_raw.head()
df = data_raw.drop('Unnamed: 0',axis =1)
from sklearn.utils import shuffle
df = shuffle(df)
df.head()
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(style="darkgrid")
sns.set(font_scale = 2)
sns.countplot(x="Category", data=df)
y = df['Category']
X = df.drop('Category',axis =1)
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 0) 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
accuracy = gnb.score(X_test, y_test) 
print('Accuracy   : --  ' ,accuracy )
cm = confusion_matrix(y_test, gnb_predictions) 
ax = sns.heatmap(cm, annot=True, fmt="d")
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB().fit(X_train, y_train) 
accuracy = clf.score(X_test, y_test) 
print('Accuracy   : --  ' ,accuracy )
clf_predictions = clf.predict(X_test)  
cm = confusion_matrix(y_test, clf_predictions) 
ax = sns.heatmap(cm, annot=True, fmt="d")
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
cm = confusion_matrix(y_test, dtree_predictions) 
accuracy = dtree_model.score(X_test, y_test) 
print('Accuracy   : --  ' ,accuracy )
ax = sns.heatmap(cm, annot=True, fmt="d")
from lightgbm import LGBMClassifier
clf = LGBMClassifier( random_state=5)
clf.fit(X_train, y_train) 
clf_predictions = clf.predict(X_test) 
cm = confusion_matrix(y_test, clf_predictions) 
accuracy = clf.score(X_test, y_test) 
print('Accuracy   : --  ' ,accuracy )
ax = sns.heatmap(cm, annot=True, fmt="d")
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
accuracy = knn.score(X_test, y_test) 
print('Accuracy   : --  ' ,accuracy )
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(y_test, knn_predictions) 
ax = sns.heatmap(cm, annot=True, fmt="d")