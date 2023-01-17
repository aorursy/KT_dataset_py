import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
dataset.head()
dataset.tail()
dataset.info()
import seaborn as sns
from pandas.plotting import scatter_matrix
%matplotlib inline
sns.set()
dataset.hist(figsize=(50,50))
plt.show()
plt.figure(figsize=(10,10))
plt.title('Feature Correlation')
sns.heatmap(dataset.astype(float).corr(), linewidths=0.5, vmax=3.0, linecolor='gray', square=True)
y = dataset['quality']
x = dataset.drop('quality',axis = 1)
bins = (2, 6.5, 8)
groups = ['bad','good']
y = pd.cut(y, bins=bins, labels=groups)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
seed=0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
models = []
models.append(('RF',RandomForestClassifier()))
models.append(('SGDC',SGDClassifier()))
models.append(('SVM',SVC()))
# Evaluating each models in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)
rf = RandomForestClassifier(random_state=seed)
rf.fit(x_train,y_train)
predicted_values = rf.predict(x_test)
print(accuracy_score(y_test,predicted_values))
print(confusion_matrix(y_test,predicted_values))
print(classification_report(y_test,predicted_values))