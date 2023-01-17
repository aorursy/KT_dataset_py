import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt #matplotlib is used for plot the graphs,
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("../input/breast-cancer.csv")
data.head()
data.shape
data.info()
data.isnull().sum()
data=data.drop(["Unnamed: 32","id"],axis=1)
data.head()
data.columns
data["diagnosis"]=data["diagnosis"].map({'B':0,'M':1}).astype(int)
data.head()
corr=data.corr()
corr.nlargest(30,'diagnosis')['diagnosis']
x=data[['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean','radius_se','perimeter_se', 'area_se','compactness_se', 'concave points_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','texture_worst','area_worst']]
y=data[['diagnosis']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.metrics import accuracy_score
model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy_score(predict,y_test)
print(accuracy_score)

accuracy=model.score(x_train,y_train)
print(accuracy)
model.score(x_train,y_train)
predict.max()
from sklearn.model_selection import cross_val_score
score=cross_val_score(model,x,y,cv=2)
score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(max_depth=6,random_state=5)
model.fit(x_train,y_train)
predict=model.predict(x_test)
acc = model.score(x_test,y_test)
acc

