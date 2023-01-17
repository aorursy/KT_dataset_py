#Importing the necessary libraries

from sklearn import datasets

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

%matplotlib inline

sns.set_style("darkgrid")

warnings.filterwarnings("ignore")

import os

df = pd.read_csv('../input/Iris.csv')
#Checking the head 

df.head()
#Checking the info of our dataset.

df.info()
plt.figure(figsize = (5,5))

df['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%.1f%%',shadow=True,cmap = 'Pastel1')

plt.title('Number of Instances Per Species')

plt.ylabel(''); #To prevent label to overlap xtick
f, axes = plt.subplots(2, 2, figsize=(10, 10))

sns.violinplot(x='Species', y='SepalLengthCm', data= df, scale='count', ax= axes[0,0])

sns.violinplot(x='Species', y='PetalLengthCm', data= df, scale='count', ax= axes[1,0])

sns.violinplot(x='Species', y='SepalWidthCm', data= df, scale='count', ax= axes[0,1])

sns.violinplot(x='Species', y='PetalWidthCm', data= df, scale='count', ax= axes[1,1]);
fig = plt.figure(figsize=(12,12))



ax1 = fig.add_subplot(2,2,1)

sns.distplot(df['SepalLengthCm'], kde=False, color=sns.xkcd_rgb["jungle green"],bins = 20)

y_axis = ax1.axes.get_yaxis()

ax1.set_title('Sepal Length Distribution', fontsize=10)

y_axis.set_visible(False)



ax2 = fig.add_subplot(2,2,2)

sns.distplot(df['SepalWidthCm'], kde=False, color=sns.xkcd_rgb["dark blue green"],bins=20)

y_axis = ax2.axes.get_yaxis()

ax2.set_title('Sepal Width Distribution', fontsize=10)

y_axis.set_visible(False)



ax3 = fig.add_subplot(2,2,3)

sns.distplot(df['PetalLengthCm'], kde=False, color=sns.xkcd_rgb["purple blue"], bins=20)

y_axis = ax3.axes.get_yaxis()

ax3.set_title('Petal Length Distribution', fontsize=10)

y_axis.set_visible(False)



ax4 = fig.add_subplot(2,2,4)

sns.distplot(df['PetalWidthCm'], kde=False, color=sns.xkcd_rgb["fuchsia"], bins=20)

y_axis = ax4.axes.get_yaxis()

ax4.set_title('Petal Width Distribution', fontsize=10)

y_axis.set_visible(False)
f, axes = plt.subplots(1, 2, figsize=(14, 7))

sns.scatterplot(x = 'SepalLengthCm',y = 'SepalWidthCm',hue = 'Species',data = df, ax=axes[0]);

sns.scatterplot(x = 'PetalLengthCm',y = 'PetalWidthCm',hue = 'Species',data = df, ax=axes[1]);
#**Modelling with scikitlearn:**

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
x = df.drop(['Species','Id'],axis=1).values 

y = df['Species'].values

x_train, x_test, y_train, y_test = train_test_split (x,y,random_state = 1)



logreg =  LogisticRegression()

fitted= logreg.fit(x_train, y_train)

y_pred= logreg.predict(x_test)

print('The accuracy of our Logistics Regression model is:',round(accuracy_score(y_test,y_pred)*100,2),'%')
print('Confusion Matrix: ')

print(confusion_matrix(y_test,y_pred))

print('\n')

print('Classification report: ')

print(classification_report(y_test,y_pred))
#kfold cross validation to increase accuracy further

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5)



scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')



print(scores.mean())
#Using gridsearchCV to get optimal n_neighbors

from sklearn.model_selection import GridSearchCV

k_range = list(range(1, 21))

params = dict(n_neighbors=k_range)

grid = GridSearchCV(knn, params, cv=10, scoring='accuracy')



grid.fit(x, y)

print(grid.best_params_)

print('The accuracy of our K-Nearest Neighbor model is:',round(grid.best_score_*100,2),'%')