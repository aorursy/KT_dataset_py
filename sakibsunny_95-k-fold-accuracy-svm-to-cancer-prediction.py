import pandas as pd
df = pd.read_csv("../input/data.csv",header = 0)

df.head()
df = df.drop("id",1)
df = df.drop("Unnamed: 32",1)
df.diagnosis.unique()
d = {'M' : 0, 'B' : 1}

df['diagnosis'] = df['diagnosis'].map(d)
df.head()
from sklearn import tree
features = list(df.columns[1:31])

features
x = df[features]

y = df["diagnosis"]
Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(x,y)

output = Tree.predict([12.42,28,71.58,346.1,0.125,0.839,0.414,0.052,0.297,0.9744,0.956,2.156,3.5,31.23,

                       0.00211,0.01458,0.01661,0.01867,0.05963,0.005208,11.91,21.5,92.87,467.7,0.1098,0.9663,0.2869,0.275,0.7738,0.273])

print(output)
from sklearn.ensemble import RandomForestClassifier
Forest = RandomForestClassifier(n_estimators = 10)

Forest = Forest.fit(x,y)

output = Forest.predict([12.42,28,71.58,346.1,0.125,0.839,0.414,0.052,0.297,0.9744,0.956,2.156,3.5,31.23,

                       0.00211,0.01458,0.01661,0.01867,0.05963,0.005208,11.91,21.5,92.87,467.7,0.1098,0.9663,0.2869,0.275,0.7738,0.273])

print(output)
from sklearn import svm
svc = svm.SVC(kernel='linear',C=1).fit(x,y)
output = svc.predict([12.42,28,71.58,346.1,0.125,0.839,0.414,0.052,0.297,0.9744,0.956,2.156,3.5,31.23,

                       0.00211,0.01458,0.01661,0.01867,0.05963,0.005208,11.91,21.5,92.87,467.7,0.1098,0.9663,0.2869,0.275,0.7738,0.273])

print(output)
from sklearn import cross_validation

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .4,random_state=0)
Kfold = KFold(len(df),n_folds = 10,shuffle = False)

print("KfoldCrossVal score using Decision Tree is %s" %cross_val_score(Tree,x,y,cv=10).mean())
Kfold = KFold(len(df),n_folds=10,shuffle=False)

print("KfoldCrossVal score using Random Forest is %s" %cross_val_score(Forest,x,y,cv=10).mean())
Kfold = KFold(len(df),n_folds=10,shuffle=False)

print("KfoldCrossVal score using SVM is %s" %cross_val_score(svc,x,y,cv=10).mean())
from sklearn import metrics
dt = Tree.fit(X_train,y_train)

y_pred = dt.predict(X_test)

metrics.accuracy_score(y_test,y_pred)
rf = Forest.fit(X_train,y_train)

y_pred = rf.predict(X_test)

metrics.accuracy_score(y_test,y_pred)
sm = svc.fit(X_train,y_train)

y_pred = sm.predict(X_test)

metrics.accuracy_score(y_test,y_pred)