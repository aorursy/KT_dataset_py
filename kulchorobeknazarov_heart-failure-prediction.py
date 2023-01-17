import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix,roc_curve,classification_report
from sklearn.metrics import plot_roc_curve
df=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.describe()
df.isnull().sum()
df.info()
y=df['DEATH_EVENT']
x=df.drop(['DEATH_EVENT'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
models={'LogisticRegression':LogisticRegression(),'KNeighbors':KNeighborsClassifier(),'RandomForestClassifier':RandomForestClassifier()}
def fit_and_score(models,x_train,x_test,y_train,y_test):
    model_scores={}
    for name ,model in models.items():
        model.fit(x_train,y_train)
        model_scores[name]=model.score(x_test,y_test)
    return model_scores
models_score=fit_and_score(models=models,x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test)
models_score
train_scores=[]
test_scores=[]

knn=KNeighborsClassifier()

neighbors=range(1,35)

for i in neighbors:
    knn.set_params(n_neighbors=i)
    knn.fit(x_train,y_train)
    train_scores.append(knn.score(x_train,y_train))
    test_scores.append(knn.score(x_test,y_test))
