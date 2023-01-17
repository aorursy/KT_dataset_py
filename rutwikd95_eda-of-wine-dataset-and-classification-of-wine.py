from IPython.display import Image
import pandas as pd
from pandas import Series,DataFrame

wineDF=pd.read_csv('../input/winequality-red.csv')
wineDF.head()
wineDF.info()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
wineDF.shape
sns.factorplot('quality',data=wineDF,kind='count')
reviews=[]
for i in wineDF["quality"]:
    if i <= 6:
        reviews.append(0)
    else:
        reviews.append(1)
        
wineDF["Reviews"] = reviews
        
sns.countplot(wineDF['Reviews'])
wineDF.head()
wineDF.hist(figsize=(20,20), color='red')
plt.show()
mycor= wineDF.corr()
plt.subplots(figsize=(12,12)) #INCREASE HEATMAP SIZE
sns.heatmap(mycor,annot=True)
sns.jointplot(x='quality',y='alcohol',data=wineDF,kind='scatter')
sns.factorplot('quality','alcohol',data=wineDF)
sns.jointplot(x='quality',y='sulphates',data=wineDF,kind='scatter')
sns.jointplot(x='quality',y='citric acid',data=wineDF,kind='scatter')
sns.jointplot(x='quality',y='fixed acidity',data=wineDF,kind='scatter')
# Image("images/random forest.png")
# Image("images/svm.jpg")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
X = wineDF.drop(["quality","Reviews"],axis = 1 )
y = wineDF["quality"]
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.head()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
pred_rfc
print(classification_report(y_test, pred_rfc))
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
rfc_eval.mean()
y = wineDF["Reviews"]
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.head()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
X_train
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)
pred_rfc
print(classification_report(y_test, pred_rfc))
svc = SVC()
svc.fit(X_train,y_train)
pred_svc = svc.predict(X_test)
print(classification_report(y_test, pred_svc))

param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)

grid_svc.best_params_


svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# Build a classification task using 3 informative features


# X, y = make_classification(n_samples=1000,
#                            n_features=10,
#                            n_informative=3,
#                            n_redundant=0,
#                            n_repeated=0,
#                            n_classes=2,
#                            random_state=0,
#                            shuffle=False)


rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

param_grid = { 
    'n_estimators': [200, 500,700],
    'max_features': ['auto', 'sqrt', 'log2']
    

}

grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
grid_rfc.fit(X, y)


grid_rfc.best_params_
X = wineDF.drop(["quality","Reviews"],axis = 1 )
y = wineDF['Reviews']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)


rfc = RandomForestClassifier(n_estimators=700,max_features='auto')
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
pred_rfc
print(classification_report(y_test, pred_rfc))
# Image("images/Table.png")
