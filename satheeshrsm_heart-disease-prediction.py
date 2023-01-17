# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
dataset = pd.read_csv('../input/heart.csv')
dataset.head()
dataset.describe()
dataset.info()
sns.heatmap(dataset.isnull())
dataset['sex'] = dataset['sex'].apply(lambda x:'Male' if x == 0 else 'Female')
sns.countplot(x = 'target',data = dataset,hue = 'sex')
plt.figure(figsize = (18,15))

sns.countplot(x = 'age',hue = 'target',data = dataset,palette = ['green','red'])

plt.legend(["Not diseased","Diseased"])
plt.figure(figsize = (10,7))

sns.scatterplot(x = 'age',y = 'thalach',hue = 'target',data = dataset,palette = ['Green','Red'])

plt.xlabel('Age')

plt.ylabel('Maximum Heart Rate')
round(dataset[dataset['target'] == 1]['age'].mean())
dataset['sex'] = dataset['sex'].apply(lambda x:0 if x == 'Male' else 1)
dataset.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
scaler = StandardScaler()
x = dataset.drop('target',axis = 1)

y = dataset['target']

x = scaler.fit_transform(x)
class classifier:

    def __init__(self,model,x,y):

        self.model = model

        self.x = x

        self.y = y

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size = 0.3,random_state = 17)

        self.model.fit(self.x_train,self.y_train)

        self.y_pred = self.model.predict(self.x_test)

    def confusionmatrix(self):

        cm = confusion_matrix(self.y_test,self.y_pred)

        plt.figure(figsize=(7,7))

        sns.heatmap(cm,square = True,annot = True,cbar = False,

                   xticklabels = ['Not Diseased','Diseased'],

                   yticklabels = ['Not Diseased','Diseased'])

        plt.title('Confusion Matrix')

        plt.xlabel('Prediction')

        plt.ylabel('True Values')

    def classificationreport(self):

        print('Classification Report')

        print(classification_report(self.y_test,self.y_pred,target_names=['Not diseased','Diseased']))

    def accuracy(self):

        self.y_train_pred = self.model.predict(self.x_train)

        print('Accuracy Score')

        print('Training Accuracy --->',accuracy_score(self.y_train,self.y_train_pred))

        print('Testing Accuracy  --->',accuracy_score(self.y_test,self.y_pred))    

    def test_accuracy(self):

        return accuracy_score(self.y_test,self.y_pred)
from sklearn.svm import SVC
svc = classifier(model = SVC(gamma = 'scale'),x = x,y = y)
svc.confusionmatrix()
svc.classificationreport()
svc.accuracy()
from sklearn.ensemble import RandomForestClassifier
rfc = classifier(model = RandomForestClassifier(n_estimators=700),x = x,y = y)
rfc.confusionmatrix()
rfc.confusionmatrix()
rfc.accuracy()
from sklearn.linear_model import LogisticRegression
lr = classifier(model = LogisticRegression(solver = 'lbfgs'),x = x,y = y)
lr.confusionmatrix()
lr.classificationreport()

lr.accuracy()
from sklearn.neighbors import KNeighborsClassifier
knn = classifier(model = KNeighborsClassifier(n_neighbors=105),x = x,y = y)
knn.confusionmatrix()
knn.classificationreport()
knn.accuracy()
models = [svc,rfc,lr,knn]

names = ['Support Vector Classifier','Random Forest Classifier','Logistic Regression','KNearestNeighbour']

acc = []

for model in models:

    acc.append(model.test_accuracy())
plt.figure(figsize=(17,10))

sns.barplot(x = names,y = acc)

plt.ylabel("Accuracy")

plt.xlabel("Classifiers")