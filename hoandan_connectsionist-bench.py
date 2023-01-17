import warnings

warnings.filterwarnings('ignore')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv('/kaggle/input/sonar-data-set/sonar.all-data.csv')

data.head()
data.info()
data.shape
X = data.iloc[:,:-1]

X.head()
y = data.iloc[:,-1]

y.head()
def make_dummies(value):

    labels = 'M'

    if value == 'M':

        labels=1

    else:

        labels = 2

    return labels
y = [make_dummies(x) for x in y]

y = pd.DataFrame(y)

y.head()
X.describe()
X.plot.box(figsize=(20,10),xticks=[])

plt.title('Boxplots of all frequency bins')

plt.xlabel('Frequency bin')

plt.ylabel('Power spectral density (normalized)')
data.iloc[:,-1].value_counts()
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score,KFold,train_test_split

from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report

from sklearn.pipeline import Pipeline
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
list_k = []

list_acc = []

for k_value in range(1,int(y_train.shape[0]**0.5)):

    list_k.append(k_value)

    model = KNeighborsClassifier(n_neighbors=k_value)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)*100

    list_acc.append(acc)

    print('accuracy is: ',acc,'for k_value:',k_value)

vitri = list_acc.index(max(list_acc))

k = list_k[vitri]

print('')

plt.plot(list_k,list_acc)

plt.xlabel('number of neighbor k')

plt.ylabel('test accuracy')
model_knn = KNeighborsClassifier(n_neighbors=3)

model_knn.fit(X_train,y_train)

y_pred_knn = model_knn.predict(X_test)

print('accuracy:',accuracy_score(y_test,y_pred_knn))

print("training score:",model_knn.score(X_train,y_train))

print("test score:",model_knn.score(X_test,y_test))
X_new =[[0.0123,0.0309,0.0169,0.0313,0.0358,0.0102,0.0182,0.0579,0.1122,0.0835,0.0548,0.0847,0.2026,0.2557,0.1870,0.2032,0.1463,0.2849,0.5824,0.7728,0.7852,0.8515,0.5312,0.3653,0.5973,0.8275,1.0000,0.8673,0.6301,0.4591,0.3940,0.2576,0.2817,0.2641,0.2757,0.2698,0.3994,0.4576,0.3940,0.2522,0.1782,0.1354,0.0516,0.0337,0.0894,0.0861,0.0872,0.0445,0.0134,0.0217,0.0188,0.0133,0.0265,0.0224,0.0074,0.0118,0.0026,0.0092,0.0009,0.0044],

        [0.0203,0.0121,0.0380,0.0128,0.0537,0.0874,0.1021,0.0852,0.1136,0.1747,0.2198,0.2721,0.2105,0.1727,0.2040,0.1786,0.1318,0.2260,0.2358,0.3107,0.3906,0.3631,0.4809,0.6531,0.7812,0.8395,0.9180,0.9769,0.8937,0.7022,0.6500,0.5069,0.3903,0.3009,0.1565,0.0985,0.2200,0.2243,0.2736,0.2152,0.2438,0.3154,0.2112,0.0991,0.0594,0.1940,0.1937,0.1082,0.0336,0.0177,0.0209,0.0134,0.0094,0.0047,0.0045,0.0042,0.0028,0.0036,0.0013,0.0016]]
y_pred_knn = model_knn.predict(X_new)

y_pred_knn
models = [

    LogisticRegression(),

    GaussianNB(),

    KNeighborsClassifier(n_neighbors=3),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    SVC(),

]
CV = 10

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

i=0

for model in models:

    model_name = model.__class__.__name__

    accuracies = cross_val_score(model, X_train, y_train, cv=CV) 

    entries.append([model_name, accuracies.mean()])

    i += 1

cv_df = pd.DataFrame(entries, columns=['model_name', 'accuracy'])
cv_df
from sklearn.model_selection import GridSearchCV
param_grid = { 

    'n_estimators': [30, 50, 100, 150, 200, 250, 300],

    'max_features': ['auto', 'sqrt', 'log2'],

    'bootstrap': [True, False],

    'criterion': ["gini", "entropy"]    

}

CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv= 5)

CV_rfc.fit(X_train, y_train)

print(CV_rfc.best_params_)
clf=RandomForestClassifier(n_estimators=50, max_features= 'log2',criterion='gini',bootstrap= False)

clf.fit(X_train,y_train)

y_pred_1=clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred_1))

print("training score:",clf.score(X_train,y_train))
print("test score:",clf.score(X_test,y_test))
imp_features = clf.feature_importances_

feature_imp = pd.Series(clf.feature_importances_, index = np.array(X.columns)).sort_values(ascending=False)
plt.figure(figsize=(20,10))

plt.bar(feature_imp.index,feature_imp)
X_reduced = X[feature_imp.iloc[:17].index]
X_train, X_test, y_train, y_test = train_test_split(X_reduced,y,test_size=0.3,random_state=7)
clf=RandomForestClassifier(n_estimators=100, max_features= 'auto',criterion='gini',bootstrap= True)

clf.fit(X_train,y_train)

y_pred_1=clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred_1))

print("training score:",clf.score(X_train,y_train))
print("test score:",clf.score(X_test,y_test))
X_reduced.plot.box(figsize=(20,10),xticks=[])

plt.title('Boxplots of all frequency bins')

plt.xlabel('Frequency bin')

plt.ylabel('Power spectral density (normalized)')
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=7)
scaler = StandardScaler()

scaler.fit(X_train)



X_train_scale = scaler.transform(X_train)

X_test_scale = scaler.transform(X_test)
CV = 15

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

i=0

for model in models:

    model_name = model.__class__.__name__

    accuracies = cross_val_score(model, X_train_scale, y_train, cv=CV) 

    entries.append([model_name, accuracies.mean()])

    i += 1

cv_df = pd.DataFrame(entries, columns=['model_name', 'accuracy'])

cv_df
params_svm = {'C' : [1, 5, 7, 10, 30, 50,75,90,100,110],

                                     'kernel' : ['rbf', 'linear'],

                                     'shrinking' : [False, True],

                                     'tol' : [0.001, 0.0001, 0.00001]}
CV_rfc = GridSearchCV(estimator=SVC(), param_grid=params_svm, cv= 15,scoring='accuracy')

CV_rfc.fit(X_train_scale, y_train)

print(CV_rfc.best_params_)
model_svm = SVC(C=5,kernel='rbf',shrinking=False,tol=0.001)

model_svm.fit(X_train_scale,y_train)

y_pred_svm=model_svm.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred_svm))

from sklearn.decomposition import PCA
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
pca = PCA(.95)

pca.fit(X_train)

pca.n_components_
X_train = pca.transform(X_train)

X_test = pca.transform(X_test)
CV = 15

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

i=0

for model in models:

    model_name = model.__class__.__name__

    accuracies = cross_val_score(model, X_train, y_train, cv=CV) 

    entries.append([model_name, accuracies.mean()])

    i += 1

cv_df = pd.DataFrame(entries, columns=['model_name', 'accuracy'])

cv_df
params = {'C' : [1, 5, 7, 10, 30, 50,75,90,100,110],

                                     'kernel' : ['rbf', 'linear'],

                                     'shrinking' : [False, True],

                                     'tol' : [0.001, 0.0001, 0.00001]}
model_svm = SVC(C=110,kernel='rbf',shrinking=False,tol=0.001)

model_svm.fit(X_train,y_train)

y_pred_svm=model_svm.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred_svm))

model_svm.score(X_train,y_train)
model_svm.score(X_test,y_test)
X_new =[[0.0123,0.0309,0.0169,0.0313,0.0358,0.0102,0.0182,0.0579,0.1122,0.0835,0.0548,0.0847,0.2026,0.2557,0.1870,0.2032,0.1463,0.2849,0.5824,0.7728,0.7852,0.8515,0.5312,0.3653,0.5973,0.8275,1.0000,0.8673,0.6301,0.4591,0.3940,0.2576,0.2817,0.2641,0.2757,0.2698,0.3994,0.4576,0.3940,0.2522,0.1782,0.1354,0.0516,0.0337,0.0894,0.0861,0.0872,0.0445,0.0134,0.0217,0.0188,0.0133,0.0265,0.0224,0.0074,0.0118,0.0026,0.0092,0.0009,0.0044],

        [0.0203,0.0121,0.0380,0.0128,0.0537,0.0874,0.1021,0.0852,0.1136,0.1747,0.2198,0.2721,0.2105,0.1727,0.2040,0.1786,0.1318,0.2260,0.2358,0.3107,0.3906,0.3631,0.4809,0.6531,0.7812,0.8395,0.9180,0.9769,0.8937,0.7022,0.6500,0.5069,0.3903,0.3009,0.1565,0.0985,0.2200,0.2243,0.2736,0.2152,0.2438,0.3154,0.2112,0.0991,0.0594,0.1940,0.1937,0.1082,0.0336,0.0177,0.0209,0.0134,0.0094,0.0047,0.0045,0.0042,0.0028,0.0036,0.0013,0.0016]]
y_pred_knn = model_knn.predict(X_new)

y_pred_knn