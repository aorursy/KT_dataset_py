import numpy as np 
import pandas as pd 
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


db = pd.read_csv('/kaggle/input/glass/glass.csv')
db.head()
db['Type'].value_counts()
db.info()
elements = []

for col in db.columns[0:8]:
    
    elements.append(col)
    
    
x  = db['Type']
for elem in elements:
    
    y = db[elem]
    print(plt.bar(x, y, align='center', alpha=0.5))
    plt.title(elem)

    plt.show()
import matplotlib.pyplot as plt
%matplotlib inline
db.hist(bins=50,figsize=(20,15))
plt.show()

from sklearn.model_selection import train_test_split as tts
x_train = db.iloc[:,0:8]
y_train = db['Type']
x_train,x_test, y_train, y_test = tts(x_train,y_train, test_size = 0.2, random_state = 18,stratify=db["Type"])

from sklearn.linear_model import LogisticRegression
y_train = np.array(y_train)

log_reg = LogisticRegression(C = 1, max_iter = 50)
log_reg.fit(x_train,y_train.ravel())

y_predict = log_reg.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict)

from sklearn.metrics import classification_report
classification_report(y_test, y_predict)
target_names = ['1', '2','3','5','6','7']
print(classification_report(y_test, y_predict, target_names=target_names))
sc = preprocessing.StandardScaler()
dtreeClf = tree.DecisionTreeClassifier()


pipe = Pipeline(steps=[('sc', sc),('dtreeClf', dtreeClf)])

criterion = ['gini', 'entropy']
max_depth = [4,5,6,7,8,10]


parameters = dict(dtreeClf__criterion=criterion, dtreeClf__max_depth=max_depth)
clf = GridSearchCV(pipe, parameters)
best = clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
from sklearn.metrics import classification_report #67 accuracy w. grid search
classification_report(y_test, y_predict)
target_names = ['1', '2','3','5','6','7']
print(classification_report(y_test, y_predict, target_names=target_names)) 
print(clf.best_params_)
sc = preprocessing.StandardScaler()

randomforestClf = RandomForestClassifier()


pipe = Pipeline(steps=[('sc', sc),('randomforestClf', randomforestClf)])

n_estimators = [500, 600, 550, 300, 200,100]
criterion = ['gini', 'entropy']
max_depth = [5, None]
min_samples_split = [0.005, 0.01]
max_features = [0.05 , 0.1]

#parameters = dict(pca__n_components=n_components,dtreeClf__criterion=criterion, dtreeClf__max_depth=max_depth)
parameters = dict(randomforestClf__criterion=criterion, randomforestClf__max_depth=max_depth,
                 randomforestClf__n_estimators = n_estimators,randomforestClf__min_samples_split =min_samples_split,
                 randomforestClf__max_features = max_features)
clf = GridSearchCV(pipe, parameters)
best = clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
from sklearn.metrics import classification_report 
classification_report(y_test, y_predict)
target_names = ['1', '2','3','5','6','7']
print(classification_report(y_test, y_predict, target_names=target_names)) 
print(clf.best_params_)