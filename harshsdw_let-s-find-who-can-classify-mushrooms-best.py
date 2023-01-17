import pandas as pd

df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

df.head()
df.info()
y = df['class']

X = df.drop(columns=['class'],axis=1)

X.head()
X = pd.get_dummies(X)

X.head()
y = y.replace({'e':0,'p':1})
y.value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
X_train.shape , X_test.shape
from sklearn.linear_model import LogisticRegression

log = LogisticRegression().fit(X_train,y_train)

log.score(X_train,y_train)
y_pred = log.predict(X_test)

log.score(X_test,y_test)
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test, y_pred)) 

print(classification_report(y_test, y_pred))

from sklearn.svm import SVC



svc = SVC(C=0.005,kernel='linear' ).fit(X_train , y_train)

svc.score(X_train,y_train)
y_pred = svc.predict(X_test)

svc.score(X_test,y_test)
print(confusion_matrix(y_test, y_pred)) 

print(classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

dec = DecisionTreeClassifier(max_depth=4 , min_samples_leaf=3).fit(X_train,y_train)

dec.score(X_train,y_train)
y_pred = dec.predict(X_test)

dec.score(X_test , y_test)
print(confusion_matrix(y_test, y_pred)) 

print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier



rand = RandomForestClassifier(max_depth=8 , n_estimators=100 , min_samples_leaf=3).fit(X_train,y_train)

rand.score(X_train,y_train)
y_pred = rand.predict(X_test)

rand.score(X_test,y_test)
print(confusion_matrix(y_test, y_pred)) 

print(classification_report(y_test, y_pred))
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB().fit(X_train,y_train)

nb.score(X_train,y_train)
y_pred = nb.predict(X_test)

nb.score(X_test,y_test)
print(confusion_matrix(y_test, y_pred)) 

print(classification_report(y_test, y_pred))
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(max_iter=100,shuffle=True,random_state=69).fit(X_train, y_train)

sgd.score(X_train, y_train)
y_pred = sgd.predict(X_test) 

sgd.score(X_test, y_test)
print(confusion_matrix(y_test, y_pred)) 

print(classification_report(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=2,leaf_size=20, algorithm='kd_tree',p=1).fit(X_train,y_train)

knn.score(X_train, y_train)   
y_pred = knn.predict(X_test)

knn.score(X_test, y_test)
print(confusion_matrix(y_test, y_pred)) 

print(classification_report(y_test, y_pred))