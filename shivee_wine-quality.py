import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/winequality-red.csv")

data.head(10)
data.info()
data.describe()
data['quality'].value_counts()
data['quality'].value_counts().plot.bar()
data.corr()
data['quality'] = data['quality'].astype(int)



data['quality'].value_counts()
y = data['quality']

X = data.drop('quality',axis=1)



from sklearn.model_selection import train_test_split , cross_val_score, GridSearchCV





from sklearn.ensemble import RandomForestClassifier



train_x,test_x,train_y,test_y = train_test_split(X,y)
forest = RandomForestClassifier(n_estimators=400,random_state = 42)

forest.fit(train_x,train_y)

predicts = forest.predict(test_x)
confusionMatrix = confusion_matrix(test_y,predicts)

confusionMatrix