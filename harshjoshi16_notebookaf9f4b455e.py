#Iris Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
#loading the data
df = load_iris()
df
#feature names /independent variable
df.feature_names
#target names /dependent variable
df.target_names
#loading the dependent variable into dataframe
iris_df = pd.DataFrame(df.data,columns=df.feature_names)
iris_df.head()
#adding the target variable in our dataframe
iris_df['target'] = df.target
iris_df
#understanding the data
iris_df.info()
iris_df['flower_name'] =iris_df.target.apply(lambda x: df.target_names[x])
iris_df.tail()
#setosa flower dataset
iris_df[iris_df.target==0].head()
#versicolor flower dataset
iris_df[iris_df.target==1].head()
#virginica flower dataset
iris_df[iris_df.target==2].head()
iris_df[55:145]
df0 = iris_df[:50] #setosa dataset
df1 = iris_df[50:100] # versicolor dataset
df2 = iris_df[100:] # virginica dataset
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')
#plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'],color="red",marker='*')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
#plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'],color="red",marker='*')
from sklearn.model_selection import train_test_split
X = iris_df.drop(['target','flower_name'], axis='columns')
y = iris_df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 10)
# support vector machine
from sklearn.svm import SVC
model = SVC(gamma='auto')
model.fit(X_train, y_train)
model.score(X_test,y_test)
#Random forest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc.score(X_test,y_test)
#decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='gini',max_depth = 1000)
dtc.fit(X_train, y_train)
dtc.score(X_test,y_test)
#Xgboost 
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
xgb.score(X_test,y_test)
