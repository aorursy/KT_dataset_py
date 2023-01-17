import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data=pd.read_csv("../input/iris.csv")
data.head()
data.shape
data.info
data.columns
data.isnull().sum()
data1=data.dropna(how="any",axis=0)
data1.head()
data1.shape
data1.describe
data1.head()
data1["Species"].value_counts()
data1.groupby(data["Sepal.Length"])["Species"].value_counts()
sns.barplot(x="Sepal.Length",y="Species",data=data)
plt.show()
labels='setosa','versicolor','virginica'
colors = ['red','green','blue']
g=data1.Species.value_counts()
plt.pie(g,labels=labels,colors=colors,autopct='%1.1f%%', shadow=False)
plt.axis('equal')
plt.xticks(rotation=0)
plt.show()
X = data[['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']]
y=data[['Species']]

X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size = 0.3, random_state=101)
sv=LogisticRegression()
sv.fit(X_train,Y_train)
sv.predict(X_test)
sv.score(X_test,Y_test)
test_score=sv.score(X_test,Y_test)
test_score

