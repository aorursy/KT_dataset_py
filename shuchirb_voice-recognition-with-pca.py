# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score  

from sklearn.decomposition import PCA

from pandas import read_csv

from subprocess import check_output



voiceData = read_csv('../input/voice.csv')



df_x=voiceData.iloc[:,1:19]

df_y=voiceData.iloc[:,20]



x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=.2)



print("Dimension reduction by PCA::")



pca=PCA(n_components=9,whiten=True)

x=pca.fit(df_x).transform(df_x)

x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=.2)



print("Decision Tree Classification::")



model=DecisionTreeClassifier()

fittedModel=model.fit(x_train, y_train)

predictions=fittedModel.predict(x_test)

predictions

print("Confussion Matrix ::\n",confusion_matrix(y_test,predictions))

print("Accuracy::",accuracy_score(y_test,predictions) ) #0.968454258675