import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix

import matplotlib.pyplot as plt

from matplotlib import style

style.use('fivethirtyeight')
from sklearn import datasets

iris=datasets.load_iris()

print(iris['target_names'])
x=iris['data'][:,3:] #iam taking only petal width

y=(iris["target"]==2).astype(np.int) #if it is virinica, it is 1 or if not it is 0

x_new=np.linspace(0,3,100).reshape(-1,1)
def log_reg():

    logistic=LogisticRegression()

    logistic.fit(x,y)

    y_pred_prob=logistic.predict_proba(x_new)

    y_pred=logistic.predict(x_new)

    x_1=[x_new[i] for i in range(len(x_new)) if y_pred[i]==1]

    y_1=[1 for i in range(len(x_new)) if y_pred[i]==1]

    x_0=[x_new[i] for i in range(len(x_new)) if y_pred[i]==0]

    y_0=[0 for i in range(len(x_new)) if y_pred[i]==0]

    plt.scatter(x_0,y_0,marker='D',color=['black'],s=80)

    plt.scatter(x_1,y_1,marker='*',color=['red'],s=100)

    plt.plot(x_new,y_pred_prob[:,1],"b-",label="iris_virginica")

    plt.plot(x_new,y_pred_prob[:,0],"g--",label="not iris_virginica")

    plt.legend()
log_reg()