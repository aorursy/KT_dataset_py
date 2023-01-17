import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/train.csv")
data['Age'] = data['Age'].replace(np.nan, 0)
data['Embarked'] = data['Embarked'].replace(np.nan, '')
data.head()
data_test = pd.read_csv("../input/test.csv")
data_test['Age'] = data_test['Age'].replace(np.nan, 0)
data_test['Embarked'] = data_test['Embarked'].replace(np.nan, '')
Y_test = pd.read_csv("../input/gender_submission.csv").iloc[:,1].values
data_test.head()
Y = data.iloc[:,1].values
X = data.iloc[:, [2,4,5,6,7,9,11]].values
X[:, 2] = X[:, 2].astype(int)
print( X[1])
X_test = data_test.iloc[:, [1,3,4,5,6,8,10]].values
X_test[:, 2] = X_test[:, 2].astype(int)
print(X_test[1])
age_imputer = Imputer(missing_values=0,strategy="mean",axis=0)
age_imputer = age_imputer.fit(X[:,2:3])
X[:,3:3] = age_imputer.transform(X[:,2:3]).astype(int)
print(X[1])
age_imputer_test = Imputer(missing_values=0,strategy="mean",axis=0)
age_imputer_test = age_imputer_test.fit(X_test[:,2:3])
X_test[:,3:3] = age_imputer_test.transform(X_test[:,2:3]).astype(int)
print(X_test[1])
X_test[:,3:3]
X_test[:,-2:-1]
fare_imputer_test = Imputer(missing_values='NaN',strategy="mean",axis=0)
fare_imputer_test = fare_imputer_test.fit(X_test[:,-2:-1])
X_test[:,-2:-1] = fare_imputer_test.transform(X_test[:,-2:-1]).astype(float)
sex_le = LabelEncoder()
sex_le.fit(X[:,1])
list(sex_le.classes_)
X[:,1] = sex_le.transform(X[:,1]) 
sex_le_test = LabelEncoder()
sex_le_test.fit(X_test[:,1])
list(sex_le_test.classes_)
X_test[:,1] = sex_le_test.transform(X_test[:,1]) 
emb_le = LabelEncoder()
emb_le.fit(X[:, -1])
X[:,-1] = emb_le.transform(X[:,-1])
print(X[1])
emb_le_test = LabelEncoder()
emb_le_test.fit(X_test[:, -1])
X_test[:, -1]
X_test[:,-1] = emb_le_test.transform(X_test[:,-1])
print(X_test[1])
onehot = OneHotEncoder(categorical_features=[1])
onehot = onehot.fit(X)
X = onehot.transform(X).toarray()
onehot_test = OneHotEncoder(categorical_features=[1])
onehot_test = onehot_test.fit(X_test)
X_test = onehot_test.transform(X_test).toarray()
X_scaled = scale(X)
X_scaled_test = scale(X_test)
model = LogisticRegression(C=1e5)
model.fit(X_scaled, Y)
predict = model.predict(X_scaled_test)
# Plot outputs
plt.figure(figsize=(30,1))
plt.scatter(data_test.iloc[:,0:1], Y_test, color='red')
plt.scatter(data_test.iloc[:,0:1], predict.reshape(418,1), color='green')
#(Y_test, predict, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
from sklearn.metrics import f1_score
f1_score(predict, Y_test)
