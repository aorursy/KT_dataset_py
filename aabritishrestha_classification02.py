import numpy as np
import pandas as pd
df = pd.read_csv(
    filepath_or_buffer='../input/rose species.csv',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) 

df.tail()
X = df.iloc[:,0:4].values
Y = df.iloc[:,4].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.1)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)