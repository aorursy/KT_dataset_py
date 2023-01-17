import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")
test.head()
train.shape
train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
df1 = train.replace(np.nan, 'nan', regex=True)
train['Embarked'] = df1['Embarked']
train.head()
train.Embarked.unique()
y = train[['Survived']]
train.drop(['Survived'],axis=1,inplace=True)
y = y.values
y = np.reshape(y,(np.shape(y)[0]))
x = train.values
print(x)
from sklearn import preprocessing
gender = preprocessing.LabelEncoder()
gender.fit(["male","female"])
x[:,1] = gender.transform(x[:,1])
embarked = preprocessing.LabelEncoder()
embarked.fit(['C','Q','S','nan'])
x[:,6] = embarked.transform(x[:,6])
keys = embarked.classes_
values = embarked.transform(embarked.classes_)
dictionary = dict(zip(keys, values))
print(dictionary)
x[x[:,6]>=3,6] = np.nan
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
x = imputer.fit_transform(x)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
loss = []
for i in range(1,100):
    #tree = DecisionTreeClassifier(criterion="entropy",max_depth = i)
    clf = RandomForestClassifier(max_depth=i, random_state=0)
    clf.fit(x_train,y_train)
    yhat = clf.predict(x_test)
    loss.append(metrics.f1_score(yhat,y_test))
i = np.argmax(loss)
plt.plot(np.arange(1,100),loss)
plt.show()
print(i+1)
clf = RandomForestClassifier(max_depth=i, random_state=0)
clf.fit(x,y)
test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
df1 = test.replace(np.nan, 'nan', regex=True)
test['Embarked'] = df1['Embarked']
test.head()
pid = test[['PassengerId']]
test.drop(['PassengerId'],axis=1,inplace=True)
x = test.values
print(x)
x[:,1] = gender.transform(x[:,1])
x[:,6] = embarked.transform(x[:,6])
x = imputer.fit_transform(x)
scaler.fit(x)
x = scaler.transform(x)
print(x)
y = clf.predict(x)
y = np.reshape(y,(418))
np.shape(y)
p_id = pid.values
p_id = np.reshape(p_id,(418))
submission_dict = {'PassengerId':p_id,'Survived':y}
df = pd.DataFrame(data=submission_dict)
df.head()
df.to_csv('Submission.csv',index=False)
