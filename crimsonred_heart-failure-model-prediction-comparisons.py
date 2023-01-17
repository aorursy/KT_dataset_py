import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import scipy



#Suppressing all warnings

warnings.filterwarnings("ignore")



%matplotlib inline
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.isna().sum()
import plotly.express as px

fig = px.pie(df, names='DEATH_EVENT', title='Distribution of Death Events in Patients',width=600, height=400)

fig.show()
corr = df.corr()

ax, fig = plt.subplots(figsize=(15,15))

sns.heatmap(corr, vmin=-1, cmap='coolwarm', annot=True)

plt.show()
corr[abs(corr['DEATH_EVENT']) > 0.1]['DEATH_EVENT']
x = df[['age','ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]

y = df['DEATH_EVENT']



#Spliting data into training and testing data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

p1=lr.predict(x_test)

s1=accuracy_score(y_test,p1)

print("Linear Regression Success Rate :", "{:.2f}%".format(100*s1))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

p3=rfc.predict(x_test)

s3=accuracy_score(y_test,p3)

print("Random Forest Classifier Success Rate :", "{:.2f}%".format(100*s3))
from sklearn.neighbors import KNeighborsClassifier

scorelist=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    p5=knn.predict(x_test)

    s5=accuracy_score(y_test,p5)

    scorelist.append(round(100*s5, 2))

print("K Nearest Neighbors Top 5 Success Rates:")

print(sorted(scorelist,reverse=True)[:5])
from sklearn.tree import DecisionTreeClassifier

list1 = []

for leaves in range(2,10):

    classifier = DecisionTreeClassifier(max_leaf_nodes = leaves, random_state=0, criterion='entropy')

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    list1.append(accuracy_score(y_test,y_pred)*100)

print("Decision Tree Classifier Top 5 Success Rates:")

print([round(i, 2) for i in sorted(list1, reverse=True)[:5]])