import numpy as np

import pandas as pd
dataset = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
dataset.head(5)
dataset.isnull().sum()
dataset.columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in dataset.columns:

    dataset[col] = le.fit_transform(dataset[col])
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



ct = ColumnTransformer(

    [('one_hot_encoder', OneHotEncoder(), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])],

remainder='passthrough'                         

)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



X_train = sc.fit_transform(dataset)

X_test = sc.transform(dataset)
Corr= dataset.corr()['class'].sort_values()

Corr
X= dataset.iloc[:,1:22].values

y=dataset.iloc[:,0].values
np.delete(X,16,axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)
print(len(X_train), len(X_test), len(y_train), len(y_test))
from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()

log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_train)
log_reg.score(X_train, y_train)
log_reg.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train, y_pred)

cm
from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_train, y_pred)

recall_score(y_train, y_pred)

f1_score(y_train, y_pred)
from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors = 15)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
knn.score(X_train,y_train)
knn.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_train,y_pred)

cm
from sklearn.metrics import precision_score,recall_score,f1_score

precision_score(y_train,y_pred)
recall_score(y_train,y_pred)
f1_score(y_train,y_pred)