import pandas as pd

df = pd.read_csv('../input/schoool.csv')

df.tail()


df.isnull().sum()
df.corr()  ##correlation
import matplotlib.pyplot as plt

import seaborn as sns   #statistical visualization





fig= sns.distplot

fig = sns.distplot(df['Research'], kde=False)

plt.title("Distribution of resrach ")

plt.show()



fig = sns.distplot(df['CGPA'], kde=False)

plt.title("Distribution of CGPA")

plt.show()

fig = sns.regplot(x="GRE Score", y="TOEFL Score", data=df)

plt.title("GRE Score vs TOEFL Score")

plt.show()
fig = sns.regplot(x="Research", y="CGPA", data=df)

plt.title("Research vs CGPA")

plt.show()
from sklearn.model_selection import train_test_split



X = df.drop(['Chance of Admit '], axis=1)

y = df['Chance of Admit ']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False)
import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso,Ridge,LinearRegression,LogisticRegression

from sklearn.metrics import mean_squared_error



models = [['DecisionTree :',DecisionTreeRegressor()],

           ['Linear Regression :', LinearRegression()],

           ['RandomForest :',RandomForestRegressor()],

           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],

           ['GradientBoostingClassifier: ', GradientBoostingRegressor()],

           ['Lasso: ', Lasso()],

           ['Ridge: ', Ridge()]]     #['Logisticregression',LogisticRegression()]

print("Results...")





for name,model in models:

    model = model

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))
df.head()
from sklearn.model_selection import train_test_split



X = df[['CGPA','Research']]

y = df['University Number']



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
prediction= knn.predict([['8.21','0']])

prediction[0]
df.tail()