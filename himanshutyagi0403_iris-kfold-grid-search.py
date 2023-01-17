import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv('../input/IRIS',header=None, names=[1,2,3,4,5])
df.head()
df.info()  ## checking nulls
x = df.iloc[:,0:4].values
y = df.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
y = encode.fit_transform(y)
sns.heatmap(pd.DataFrame(x).corr(), annot=True)
## baseline classifier
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,train_size=0.75, random_state=1)
from sklearn.linear_model import LogisticRegression
classifier_log  = LogisticRegression()
classifier_log.fit(x_train,y_train)

y_pred_log  = classifier_log.predict(x_test)

##metrics

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_log))
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=10,criterion='entropy')
classifier_rf.fit(x_train,y_train)

y_pred_rf = classifier_rf.predict(x_test)

##metrics

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_rf))
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=10,n_jobs=-1)
classifier_knn.fit(x_train,y_train)

y_pred_knn = classifier_knn.predict(x_test)
##metrics

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_knn))
x = x[:,[0,1,3]]
sns.heatmap(pd.DataFrame(x).corr(), annot=True)
from sklearn.linear_model import LogisticRegression
classifier_log  = LogisticRegression()
classifier_log.fit(x_train,y_train)

y_pred_log  = classifier_log.predict(x_test)

##metrics

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_log))
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=1)
classifier_rf.fit(x_train,y_train)

y_pred_rf = classifier_rf.predict(x_test)

##metrics

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_rf))
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=10,n_jobs=-1)
classifier_knn.fit(x_train,y_train)

y_pred_knn = classifier_knn.predict(x_test)
##metrics

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_knn))
from sklearn.svm import LinearSVC
classifier_svm = LinearSVC(random_state=1)
classifier_svm.fit(x_train,y_train)

y_pred_SVM = classifier_svm.predict(x_test)
##metrics

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_SVM))
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier_knn,X=x_train,y=y_train, cv=10)
accuracies
plt.plot(accuracies)
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors': range(1,50),
              'metric' : ['minkowski','manhattan', 'euclidean'],
             }
grid_search = GridSearchCV(classifier_knn, parameters,scoring='accuracy', cv=10,n_jobs=-1,)
grid_search.fit(x_train,y_train)

grid_search.best_params_
## final model
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=2,n_jobs=-1)
classifier_knn.fit(x_train,y_train)

y_pred_knn = classifier_knn.predict(x_test)
##metrics

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_knn))



