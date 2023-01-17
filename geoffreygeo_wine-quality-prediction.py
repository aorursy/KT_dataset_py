# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#scikit learn library 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/winequality-red.csv')
df.describe()
#checking for null values
df.isnull().sum()
df.quality.unique()
df_label = df['quality'].values
df_data=df.drop('quality',axis=1)
#splitting the test data and the train data
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(df_data,df_label,test_size=1/3, random_state=0)

clf = GaussianNB()
clf1= MultinomialNB()
clf2=BernoulliNB()
clf.fit(xtrain,ytrain)
clf1.fit(xtrain,ytrain)
clf2.fit(xtrain,ytrain)
accuracy = dict()
predicted_values_GB=clf.predict(xtest)
predicted_values_NB=clf1.predict(xtest)
predicted_values_BB=clf2.predict(xtest)
accuracy['Gaussian'] = accuracy_score(predicted_values_GB,ytest)*100
accuracy['MultinomialNB'] = accuracy_score(predicted_values_NB,ytest)*100
accuracy['BernoulliNB'] = accuracy_score(predicted_values_BB,ytest)*100
accuracy['Max_accuracy'] = 100
accuracy=pd.DataFrame(list(accuracy.items()),columns=['Algorithm','Accuracy'])
display(accuracy)
sns.lineplot(x='Algorithm',y='Accuracy',data=accuracy)
display(df['quality'].unique())
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=(11,11))
# we will create a new feature called label. This column will contain the values bad,average,excellant. 
#This will be split in the following way. 
#1,2,3 --> Bad
#4,5,6,7 --> Average
#8,9,10 --> Excellent
def label_quality(quality):
    if(quality >=1 and quality<=3):
        return 'Bad'
    elif(quality>=4 and quality<=7):
        return 'Average'
    elif(quality>=8 and quality<=10):
        return 'Excellent'
df['label']=df['quality'].apply(label_quality)
sns.countplot(x='quality',data=df,ax=ax1)
sns.countplot(x='label',data=df,ax=ax2)

df_label = df['label'].values
df_data=df.drop(['quality','label'],axis=1)
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(df_data,df_label,test_size=1/3, random_state=0)
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(xtrain,ytrain)
predicted_result_dt= DT.predict(xtest)
print("The accuracy score of DT is {}".format(accuracy_score(predicted_result_dt,ytest)*100))

import graphviz
from sklearn import tree

data = export_graphviz(DT,out_file=None,   
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(data)
display(graph)
dot_data = StringIO()
export_graphviz(DT, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())