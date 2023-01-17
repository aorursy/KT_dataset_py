import numpy as np

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



data=pd.read_csv('/Users/saiteja/Desktop/mushrooms.csv')







le=LabelEncoder()

data['class'] = le.fit_transform(data['class'])





data_en=pd.get_dummies(data)

print(data_en)





data_dataframe = pd.DataFrame(data_en)



X=data_dataframe.drop('class',axis=1)

y=data_dataframe['class']



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state=42)

print("X_train shape: {}".format(X_train.shape))

print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))

print("y_test shape: {}".format(y_test.shape))









kn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

kn.fit(X_train,y_train)





y_pred = kn.predict(X_test)

print("Predicted testiing codes: {}".format(y_pred))



f = open("pred.csv", "a")

np.savetxt('pred.csv',y_pred, delimiter=",")

f.close()

f = open("test.csv", "a")

np.savetxt('test.csv',y_test, delimiter=",")

f.close()

print(f1)

print(f2)



f1 = np.bincount(y_pred)

f2 = np.bincount(y_test)

print("Test set score: {:.2f}%".format(np.mean(y_pred == y_test)*100))






