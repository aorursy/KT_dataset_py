# Importing the libraries and Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris            # importing iris dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
%matplotlib inline
dataset = pd.read_csv("../input/iris/Iris.csv")
dataset.head()
dataset['Species'].value_counts()
dataset.info()
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# We can see there is no null values
dataset['Species'] = dataset['Species'].astype('category')

dataset['Species'] = dataset['Species'].cat.codes
dataset.head()
dataset['Species'].value_counts()
dataset.corr()
sns.heatmap(dataset.corr())
dataset.corr()[['Species']].sort_values(by='Species',ascending=False)
sns.pairplot(dataset)
sns.pairplot(dataset.drop("Id", axis=1), hue="Species", size=3)
x_train=dataset.drop('Species',axis=1)
y_train=dataset[['Species']]
x_train.head()
y_train.head()
x_train,x_test,y_train,y_test=train_test_split(x_train, y_train, test_size=0.2, random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
model_logistic=LogisticRegression()
model_logistic.fit(x_train,y_train)
y_predict=model_logistic.predict(x_test)
print("Cofusion matrix is: \n {} \n".format(confusion_matrix(y_test,y_predict)))
print("Accuracy of the model is: {} \n".format(accuracy_score(y_test,y_predict)*100))
print("Classification Report : \n",classification_report(y_predict,y_test))
sns.heatmap(confusion_matrix(y_test,y_predict),annot = True,cmap = 'BuGn')