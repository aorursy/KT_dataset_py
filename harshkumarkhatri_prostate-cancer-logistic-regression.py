from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
dataset=pd.read_csv("../input/prostate-cancer/datasets_66762_131607_Prostate_Cancer.csv")
dataset.head()
dataset=dataset.drop("id",axis=1)
dataset.info()
dataset["diagnosis_result"].unique()
y=dataset["diagnosis_result"]
x=dataset.drop("diagnosis_result",axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
y_test.head()
reg=linear_model.LogisticRegression(max_iter=1000)
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
print(y_predict)
accuracy_score(y_predict,y_test)


confusion_matrix(y_predict,y_test)

dataset["diagnosis_result"].head()
dataset["diagnosis_result"]=pd.get_dummies(dataset["diagnosis_result"])
dataset["diagnosis_result"].head()