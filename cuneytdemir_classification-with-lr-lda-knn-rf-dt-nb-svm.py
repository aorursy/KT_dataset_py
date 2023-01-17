import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np # linear algebra
import pandas as pd #
from sklearn.preprocessing import LabelEncoder

datapath ="../input/mushrooms.csv"
data = pd.read_csv(datapath)  

#Onehot  appyl (0,1 convert)
for col in data.columns :
   data[col] = pd.get_dummies(data[col])

X = data.iloc[:,data.columns !='class']
y = data.iloc[:,data.columns == 'class']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)

models = [
    ('LR', LogisticRegression()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('RF',RandomForestClassifier()),
    ('DT', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC())
]

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predictionsRF = rf.predict(X_test)

print('RF accuracy degeri :', accuracy_score(y_test, predictionsRF))
print(confusion_matrix(y_test, predictionsRF))
print(classification_report(y_test, predictionsRF))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictionsKNN = knn.predict(X_test)

print('KNN accuracy degeri :', accuracy_score(y_test, predictionsKNN))
print(confusion_matrix(y_test, predictionsKNN))
print(classification_report(y_test, predictionsKNN))