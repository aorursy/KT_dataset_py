# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df
df.isnull().sum()
df = df.drop(columns=['Name','Cabin','Ticket','Embarked'])

df.head()
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go
iplot([go.Histogram2dContour(x=df.head(500)['Pclass'], 

                             y=df.head(500)['Survived'], 

                             contours=go.Contours(coloring='heatmap')),

       go.Scatter(x=df.head(1000)['Survived'], y=df.head(1000)['Pclass'], mode='markers')])
df1 = df.assign(n=0).groupby(['PassengerId', 'Survived'])['n'].count().reset_index()

df1 = df1[df["Survived"] < 100]

v = df1.pivot(index='Survived', columns='PassengerId', values='n').fillna(0).values.tolist()

iplot([go.Surface(z=v)])
df['Survived'].value_counts().head(10).plot.pie()



# Unsquish the pie.

import matplotlib.pyplot as plt

plt.gca().set_aspect('equal')
plt.scatter(df.index,df['Survived'])

plt.show()
plt.boxplot(df['PassengerId'])
x = df.drop(columns=['Survived'])

print(x)
y = df['Survived']

print(y)
from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()  

x= x.apply(label_encoder.fit_transform)

print(x)
y= label_encoder.fit_transform(y)

print(y)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# random forest model creation

rfc = RandomForestClassifier()

rfc.fit(x_train,y_train)

# predictions

rfc_predict = rfc.predict(x_test)



print("Accuracy:",accuracy_score(y_test, rfc_predict))
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer

clf = clf.fit(x_train,y_train)

#Predict the response for test dataset

y_pred = clf.predict(x_test)





print("Accuracy:",accuracy_score(y_test, y_pred))
nb = GaussianNB()

nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

print(y_pred)



print(accuracy_score(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier  

classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  

classifier.fit(x_train, y_train)  
y_pred= classifier.predict(x_test)  

#Creating the Confusion matrix  

from sklearn.metrics import confusion_matrix  

cm= confusion_matrix(y_test, y_pred)  

print(accuracy_score(y_test, y_pred))
from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)
# Use score method to get accuracy of model

score = logisticRegr.score(x_test, y_test)

print(score)