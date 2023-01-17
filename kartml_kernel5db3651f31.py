import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("../input/heart-disease-uci/heart.csv")
df.dtypes
df.isnull().sum()
df.dtypes
df.head(3)
df.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1), df['target'], test_size = .2, random_state=25) 
X_train.shape
y_test.shape
X_train.head(3)
y_train.shape
y_train.head()
from sklearn.ensemble import RandomForestClassifier 
classifier=RandomForestClassifier(n_estimators=20,random_state=30,max_features=13,max_depth=5)
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)
y_pred_quant = classifier.predict_proba(X_test)[:, 1]
y_predict
y_pred_quant
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predict)
cm
print('Accuracy:{}'.format((21+28)/61))