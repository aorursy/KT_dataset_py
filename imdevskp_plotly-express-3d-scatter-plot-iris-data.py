import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px



# plt.style.use('default')

color_pallete = ['#fc5185', '#3fc1c9', '#364f6b']

sns.set_palette(color_pallete)

sns.set_style("white")



from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score
df = pd.read_csv('../input/Iris.csv')

df.head()
df.info()
df.describe()
df['Species'].value_counts().plot(kind='bar')
df.drop(['Id'], inplace=True, axis=1)
plt.figure(figsize=(8, 8))

ax = sns.pairplot(df, hue='Species')

plt.show()
px.scatter_3d(df, x="PetalLengthCm", y="PetalWidthCm", z="SepalLengthCm", size="SepalWidthCm", 

              color="Species", color_discrete_map = {"Joly": "blue", "Bergeron": "violet", "Coderre":"pink"})
px.scatter_3d(df, x="PetalLengthCm", y="PetalWidthCm", z="SepalWidthCm", size="SepalLengthCm", 

              color="Species", color_discrete_map = {"Joly": "blue", "Bergeron": "violet", "Coderre":"pink"})
plt.figure() 

sns.heatmap(df.corr(),annot=True)

plt.show()
X = df.drop(['Species'], axis=1)

y = df['Species']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn import svm



svc = svm.SVC()

svc.fit(X_train,y_train)



pred = svc.predict(X_test) 

accuracy_score(pred, y_test)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)



pred = knn.predict(X_test) 

print(accuracy_score(pred, y_test))
from sklearn.naive_bayes import GaussianNB



nbc = GaussianNB()

nbc.fit(X_train,y_train)



pred = nbc.predict(X_test) 

print(accuracy_score(pred, y_test))
from sklearn.linear_model import LogisticRegression



lrc = LogisticRegression()

lrc.fit(X_train,y_train)



pred = lrc.predict(X_test) 

print(accuracy_score(pred, y_test))