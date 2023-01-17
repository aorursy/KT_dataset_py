!unzip /content/titanic.zip
import pandas as pd
dataset = pd.read_csv('/content/train.csv')
trainset= pd.read_csv('/content/test.csv')
dataset
dataset.info()
trainset.info()
len(set(dataset.Embarked))
data = pd.get_dummies(dataset,columns=['Sex','Embarked'])
df   = pd.get_dummies(trainset,columns=['Sex','Embarked'])
data
df
import numpy as np
from sklearn.impute import SimpleImputer
x1 = data['Age'].values.reshape(-1,1)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(x1)

x2 = df['Age'].values.reshape(-1,1)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(x2)

x3 = df['Fare'].values.reshape(-1,1)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(x3)
imp_mean.transform(x1)
imp_mean.transform(x2)
print( )
X = data[['Age','Pclass','SibSp','Parch','Fare','Sex_male','Embarked_C','Embarked_Q','Embarked_S']]
X2  = df[['Age','Pclass','SibSp','Parch','Fare','Sex_male','Embarked_C','Embarked_Q','Embarked_S']]
X['Age'] = imp_mean.transform(x1)
X2['Age']= imp_mean.transform(x2)
X2['Fare']=imp_mean.transform(x3)
X['Age']
y = data['Survived']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X2= scaler.transform(X2)
X2[:20,:]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(random_state=0)
clf.fit(X, y)
clf.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

clf = RandomForestClassifier(max_depth=5, random_state=0,max_features=4)
clf.fit(X,y)
clf.score(X_test,y_test)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X,y)
clf.score(X_test,y_test)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(max_iter=1500)
clf.fit(X, y)
clf.score(X_test,y_test)
clf = AdaBoostClassifier()
clf.fit(X, y)
clf.score(X_test,y_test)
from sklearn.svm import SVC

clf = SVC(kernel="linear", C=0.025)
clf.fit(X, y)
clf.score(X_test,y_test)
from sklearn.svm import SVC

clf = SVC(kernel="rbf", gamma=9,C=1,degree=value)
clf.fit(X, y)
clf.score(X_train,y_train)
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=4)
clf.fit(X, y)
clf.score(X_train,y_train)
y_pred = clf.predict(X2)
X2[:5]
results = ids.assign(Survived = y_pred[-418:])
results.to_csv("/content/gender_submission.csv", index=False)