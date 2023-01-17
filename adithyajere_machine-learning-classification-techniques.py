import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
# Y is not scaled, because it contians categorical values.
from sklearn.linear_model import LogisticRegression #Logistic regression  is still a linear model

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
y_pred, Y_test
#Making the Confusion Matrix --> to assess accuracy

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
cm
#65 + 24 = 89 correct results, 3+8 = 11 incorrect results.

#Visualizing the Training Set results
plt.figure(1)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted')
plt.legend()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #n_neighbors parameter can be tuned, but will learn later.
#minkowsky - 2 --> Euclidean distance
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
#visualizing the result

plt.figure (2)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted (KNN)')
plt.legend()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.svm import SVC #Support vecture classifier
classifier = SVC(kernel = 'linear', random_state = 0) #other options for kernel: 'poly', 'rbf' (gaussean), 'sigmoid', 'precomputed'
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
cm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)

plt.figure (2)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted (rbf SVM Kernel)')
plt.legend()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

plt.figure (2)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted (Naive Bayes)')
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

plt.figure (2)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted (DecisionTree Classification)')
plt.legend()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pd.read_csv('../input/Social_Network_Ads.csv')
x = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

plt.figure (2)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
plt.title('Prediction boundary and training examples plotted (Random Forest Classification)')
plt.legend()
plt.show()
cm
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
knn = KNeighborsClassifier(n_neighbors=15)
clf = knn.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
acc_knb_model=roc_auc_score(Y_test, y_pred)*100
acc_knb_model
lr = LogisticRegression(C = 0.2)
clf1 = lr.fit(X_train, Y_train)
y_pred1 = clf1.predict(X_test)
acc_log_reg=roc_auc_score(Y_test, y_pred1)*100
acc_log_reg
clf2 = GaussianNB().fit(X_train, Y_train)
y_pred2 = clf2.predict(X_test)
acc_nb=roc_auc_score(Y_test, y_pred2)*100
acc_nb
clf3 = tree.DecisionTreeClassifier().fit(X_train, Y_train)
y_pred3 = clf3.predict(X_test)
acc_dt=roc_auc_score(Y_test, y_pred3)*100
acc_dt
clf4 = RandomForestClassifier(max_depth=5, random_state=0).fit(X_train, Y_train)
y_pred4 = clf4.predict(X_test)
acc_rmf_model=roc_auc_score(Y_test, y_pred4)*100
acc_rmf_model
clf5 = SVC(gamma='auto').fit(X_train, Y_train)
y_pred5 = clf5.predict(X_test)
acc_svm_model=roc_auc_score(Y_test, y_pred5)*100
acc_svm_model
sgd_model=SGDClassifier()
sgd_model.fit(X_train,Y_train)
sgd_pred=sgd_model.predict(X_test)
acc_sgd=round(sgd_model.score(X_train,Y_train)*100,10)
acc_sgd
xgb_model=XGBClassifier()
xgb_model.fit(X_train,Y_train)
xgb_pred=xgb_model.predict(X_test)
acc_xgb=round(xgb_model.score(X_train,Y_train)*100,10)
acc_xgb
lgbm = LGBMClassifier()
lgbm.fit(X_train,Y_train)
lgbm_pred=lgbm.predict(X_test)
acc_lgbm=round(lgbm.score(X_train,Y_train)*100,10)
acc_lgbm
regr = linear_model.LinearRegression()
regr.fit(X_train,Y_train)
regr_pred=regr.predict(X_test)
acc_regr=round(regr.score(X_train,Y_train)*100,10)
acc_regr
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest','Stochastic Gradient Decent','Linear Regression','Naive Bayes','XGBoost','LightGBM','Decision Tree'],
    'Score': [acc_svm_model, acc_knb_model, acc_log_reg, 
              acc_rmf_model,acc_sgd,acc_regr,acc_nb,acc_xgb,acc_lgbm,acc_dt]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df