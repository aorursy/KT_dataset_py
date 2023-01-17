from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'



import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report 
iris = pd.read_csv('../input/Iris.csv', index_col='Id')
iris.head()

iris.tail()
iris.info()
iris.describe()
sns.pairplot(iris, hue='Species')
X = iris.drop('Species', axis=1)

Y = iris['Species']

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=0.3, random_state=9)
X_train.head()

X_test.head()

Y_train.head()

Y_test.head()
#Modeling SVC with default parameter. Setting only random_state = 9 to get same result any time.

svc_default_model = SVC(random_state=9)

svc_default_model.fit(X_train, Y_train)
predic_svc_default = svc_default_model.predict(X_test)
print(confusion_matrix(Y_test, predic_svc_default))

print(classification_report(Y_test, predic_svc_default))
#Modeling SVC with linear's kernel and using the same random_state. 

svc_linear_model = SVC(kernel='linear', random_state=9)

svc_linear_model.fit(X_train, Y_train)
predic_svc_linear = svc_linear_model.predict(X_test)
print(confusion_matrix(Y_test, predic_svc_linear))

print(classification_report(Y_test, predic_svc_linear))
#Modeling KNeighborsClassifier with K = 9

knn_model = KNeighborsClassifier(n_neighbors=9)

knn_model.fit(X_train, Y_train)
predic_knn = knn_model.predict(X_test)
print(confusion_matrix(Y_test, predic_knn))

print(classification_report(Y_test, predic_knn))