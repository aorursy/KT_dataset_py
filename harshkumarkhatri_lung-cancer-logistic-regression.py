# This dataset does not belong to me

from sklearn import linear_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
# from sklearn.model_selection import train_test_split
dataset=pd.read_csv("/kaggle/input/lung-cancer/datasets_85411_197066_lung_cancer_examples.csv")
dataset.head()
dataset.info()
y=dataset["Result"]
y.head()
dataset=dataset.drop("Name",axis=1)
dataset=dataset.drop("Surname",axis=1)
dataset.head()

dataset=dataset.drop("Result",axis=1)
x=dataset
x.head()
x_train,x_test,y_train,y_test=train_test_split(x,y)
x_test.head()
reg=linear_model.LogisticRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
print(y_predict)
accuracy_score(y_predict,y_test)
confusion_matrix(y_predict,y_test)

# Plotting the data
sns.pairplot(dataset,hue="Result")