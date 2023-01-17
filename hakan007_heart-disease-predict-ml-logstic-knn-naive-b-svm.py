import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# Disabling warnings
import warnings
warnings.simplefilter("ignore")

# ML 
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,mean_squared_error,roc_curve,roc_auc_score,classification_report,r2_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,SVR
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
hd = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df = hd.copy()
df.tail()
df.info()
df.isnull().any()
df.target.value_counts()
df.target.value_counts().plot.barh();
sns.countplot(x="target",hue = "sex", data=df, palette="bwr");
df.describe().T
a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")
frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
df.head()
df = df.drop(columns = ['cp', 'thal', 'slope'])
df.head()
y = df.target.values
x_data = df.drop(['target'], axis = 1)
# Normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
lr = LogisticRegression(solver = 'liblinear')
lr_model = lr.fit(x_train,y_train)
lr_model
y_pred = lr_model.predict(x_test)
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
lr_model.predict(x_test)[-10:]
lr_model.predict_proba(x_test)[-10:]
y_probs = lr_model.predict_proba(x_test)[:,1]
y_pred = [1 if i>0.52 else 0 for i in y_probs]
y_pred[-10:]
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
logit_roc_auc = roc_auc_score(y_test,lr_model.predict(x_test))

fpr, tpr, tresholds = roc_curve(y_test,lr_model.predict_proba(x_test)[:,1])
plt.figure(figsize=(8,8))
plt.plot(fpr,tpr,label = "AUC (area = %0.2f)"%logit_roc_auc)
plt.plot([0,1],[0,1],"r--")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.title('ROC Curve');
print('Accuracy Rate:',accuracy_score(y_test, y_pred))
print("Logistic TRAIN score with ",format(lr_model.score(x_train, y_train)))
print("Logistic TEST score with ",format(lr_model.score(x_test, y_test)))
cm=confusion_matrix(y_test,y_pred)
print(cm)
y = df.target
x = df.drop('target',axis = 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                test_size = 0.20,
                                                random_state = 42)
nb = GaussianNB()
nb_model = nb.fit(x_train,y_train)
nb_model
y_pred = nb_model.predict(x_test)
y_pred[-10:]
accuracy_score(y_test,y_pred)
y_probs = nb_model.predict_proba(x_test)[:,1]
y_pred = [1 if i>0.4 else 0 for i in y_probs]
y_pred[-10:]
accuracy_score(y_test,y_pred)
y = df.target
x = df.drop('target',axis = 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                test_size = 0.20,
                                                random_state = 42)
knn =KNeighborsClassifier()
knn_model = knn.fit(x_train,y_train)
knn_model
y_pred = knn_model.predict(x_test)
accuracy_score(y_test,y_pred)
knn_params = {"n_neighbors":np.arange(1,50)}
knn =KNeighborsClassifier()
knn_cv = GridSearchCV(knn,knn_params,cv = 10)
knn_cv = knn_cv.fit(x_train,y_train)
print("Best Score:"+str(knn_cv.best_score_))
print("Best Parameters:"+str(knn_cv.best_params_))
knn_final =KNeighborsClassifier(n_neighbors = 21)
knn_final = knn_final.fit(x_train,y_train)
y_pred = knn_final.predict(x_test)
accuracy_score(y_test,y_pred)
y = df.target
x = df.drop('target',axis = 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                test_size = 0.20,
                                                random_state = 42)
svm_model = SVC(kernel='linear').fit(x_train,y_train)
svm_model
y_pred = svm_model.predict(x_test)
accuracy_score(y_test,y_pred)
svc_params = {"C":np.arange(1,50)}
svc = SVC(kernel = 'linear')
svc_cv_model = GridSearchCV(svc,svc_params,
                           cv = 10,
                           n_jobs = -1,
                           verbose = 2)
svc_cv_model.fit(x_train,y_train)

print("Best Parameters:"+str(svc_cv_model.best_params_))
accuracy_score(y_test,y_pred)
svc_tuned = SVC(kernel = "linear",C=2).fit(x_train,y_train)
y_pred = svc_tuned.predict(x_test)
accuracy_score(y_test,y_pred)
svc_model = SVC(kernel = "rbf").fit(x_train,y_train)
svc_model
y_pred = svc_model.predict(x_test)
accuracy_score(y_test,y_pred)
svc_params ={"C":[0.00001,0.001,0.01,5,10,50,100],
            "gamma":[0.0001,0.001,0.01,1,5,10,50,100]}
svc =SVC()
svc_cv_model = GridSearchCV(svc,svc_params,
                           cv = 10,
                           n_jobs = -1,
                           verbose = 2)
svc_cv_model.fit(x_train,y_train)
print("Best Parameters:"+str(svc_cv_model.best_params_))
svc_tuned = SVC(C=100,gamma = 0.0001).fit(x_train,y_train)

y_pred = svc_tuned.predict(x_test)
accuracy_score(y_test,y_pred)
