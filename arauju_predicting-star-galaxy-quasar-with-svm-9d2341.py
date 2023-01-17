import numpy as np
import pandas as pd
from pandas import Timestamp
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split
from IPython.display import HTML, display
from tabulate import tabulate
import ipywidgets as widgets
#Importing the data
df_sky = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv", dtype={'class':'category'})
df_sky.drop(['objid','specobjid'],axis =1,inplace=True)
df_sky.head()
counts = df_sky.groupby(['class'])['class'].count().plot(kind='bar')
for column in df_sky.columns:
    if column not in ['rerun','class']:
        df_sky.groupby('class')[column].plot.kde()
        plt.title(column)
        plt.legend()
        plt.show()
X = df_sky.drop('class',axis=1)
y = df_sky['class']
model = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model.fit(X_train,y_train)
predicted = model.predict(X_test)
#print(confusion_matrix(y_test, predicted))  
predict_table = widgets.HTML(value = "")
predict_table.value = "<pre>" + classification_report(y_test, predicted) + "</pre>"
display(predict_table)
print(classification_report(y_test, predicted))
svclassifier = SVC(kernel='linear')
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = svclassifier.fit(X_train,y_train)
y_pred = svclassifier.predict(X_test)
#print(confusion_matrix(y_test, predicted))  
# predict_table = widgets.HTML(value = "")
# predict_table.value = "<pre>" + classification_report(y_test, y_pred) + "</pre>"
print(classification_report(y_test, y_pred))
svclassifier = SVC(kernel='poly')
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = svclassifier.fit(X_train,y_train)
y_pred = svclassifier.predict(X_test)
#print(confusion_matrix(y_test, predicted))  
# predict_table = widgets.HTML(value = "")
# predict_table.value = "<pre>" + classification_report(y_test, predicted) + "</pre>"
# display(predict_table)
print(classification_report(y_test, predicted))
svclassifier = SVC(kernel='rbf')
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = svclassifier.fit(X_train,y_train)
y_pred = svclassifier.predict(X_test)
#print(confusion_matrix(y_test, predicted))  
# predict_table = widgets.HTML(value = "")
# predict_table.value = "<pre>" +  + "</pre>"
print(classification_report(y_test, y_pred))