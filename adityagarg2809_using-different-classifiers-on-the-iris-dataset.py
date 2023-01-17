

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris.head()
iris = iris.drop('Id',axis=1)
iris['Species'].unique()
def species_to_num(x):

    if x == 'Iris-setosa':

        return 0

    elif x == 'Iris-versicolor':

        return 1

    else:

        return 2
iris['Species'] = iris['Species'].apply(species_to_num)
iris.head()
iris[iris.columns[0:-1]].describe()
X = iris[iris.columns[0:-1]]

y = iris[iris.columns[-1]]
plt.figure(figsize=(12,8))

sns.scatterplot(iris['SepalLengthCm'],iris['SepalWidthCm'],hue=iris['Species'])
sns.heatmap(iris.corr())
from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=101)
scaler = StandardScaler()



scaled_train = scaler.fit_transform(X_train)



scaled_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(scaled_train,y_train)



preds = lr_model.predict(scaled_test)
from sklearn.metrics import confusion_matrix, classification_report 
print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
preds = mnb.predict(X_test)
print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(scaled_train,y_train)
preds = gnb.predict(scaled_test)
print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_train = pca.fit_transform(scaled_train)

reduced_test =pca.transform(scaled_test)
reduced_train.shape
from sklearn.svm import SVC
clf = SVC()
clf.fit(reduced_train,y_train)
preds = clf.predict(reduced_test)
print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))