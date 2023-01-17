import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
df = pd.read_csv('../input/winedataset.csv')
y = df.iloc[:,0]
x = df.iloc[:,1:]
df.info()
sns.pairplot(df)
le = LabelEncoder()
y=le.fit_transform(y)
y
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.22,random_state=42)
mnb = MultinomialNB()
svc = SVC()
dt = DecisionTreeClassifier()
mnb.fit(x_train,y_train)
mnb.score(x_train,y_train)
predmnb = mnb.predict(x_test)
print("\nAccuracy Score\n")
print(accuracy_score(y_test,predmnb))
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,predmnb))
print("\nClassification Report\n")
print(classification_report(y_test,predmnb))
svc.fit(x_train,y_train)
svc.score(x_train,y_train)
predsvc = svc.predict(x_test)
print("\nAccuracy Score\n")
print(accuracy_score(y_test,predsvc))
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,predsvc))
print("\nClassification Report\n")
print(classification_report(y_test,predsvc))
dt.fit(x_train,y_train)
dt.score(x_train,y_train)
preddt = dt.predict(x_test)
print("\nAccuracy Score\n")
print(accuracy_score(y_test,preddt))
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,preddt))
print("\nClassification Report\n")
print(classification_report(y_test,preddt))

