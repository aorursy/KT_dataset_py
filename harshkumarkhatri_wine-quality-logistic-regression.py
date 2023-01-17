from sklearn import linear_model

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
dataset=pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

dataset.head(3)
dataset.isnull().sum()
dataset.dtypes.sample(12)
dataset["quality"].unique()

y=dataset["quality"]

x=dataset.drop("quality",axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
reg=linear_model.LogisticRegression()

reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)

print(y_predict)

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score(y_predict,y_test)
# Now i have seen few kernals where they have used Stochastic gradient descent classifier

# to increase the accuracy for the model



reg2=linear_model.SGDClassifier(penalty=None)

reg2.fit(x_train,y_train)

y_pred=reg2.predict(x_test)

accuracy_score(y_pred,y_test)
bins=(2,5,8)

group_name=["bad","good"]

dataset["quality"]=pd.cut(dataset["quality"],bins=bins,labels=group_name)
dataset["quality"].head()

from sklearn.preprocessing import LabelEncoder

label_quality=LabelEncoder()

dataset["quality"]=label_quality.fit_transform(dataset["quality"])

dataset["quality"].head()



# Splitting the dataset again

y=dataset["quality"]

x=dataset.drop("quality",axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)

print(y_predict)

accuracy_score(y_predict,y_test)
confusion_matrix(y_predict,y_test)



sns.pairplot(dataset)
sns.pairplot(dataset,hue="quality")
# Trying to see it accuracy greater than the previous one can be achieved with SGDC

reg2.fit(x_train,y_train)

y_pred=reg2.predict(x_test)

print(y_pred)

accuracy_score(y_pred,y_test)
confusion_matrix(y_test,y_pred)


