import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("../input/Iris.csv")
data.head()
data['Species'].value_counts()
data.loc[data['PetalWidthCm'].idxmax()]
species={"Iris-versicolor":0,"Iris-virginica":1,"Iris-setosa":2}
data['Species']=data['Species'].map(species)
res = data['Species']

data_new = data.drop(['Id', 'Species'], axis=1)

x_train,x_test,y_train,y_test=train_test_split(data_new,res,test_size=0.2,random_state=0)
lm = LogisticRegression(solver='lbfgs',multi_class='ovr').fit(x_train,y_train)
predictions = lm.predict(x_test)
ac = accuracy_score(y_test,predictions)
print("Logistic Regression Accuracy:",ac)
