import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns; sns.set()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        df=pd.read_csv(os.path.join(dirname, filename))


df.shape
df.head()
df.columns
df.info()
df.isnull().sum()
df.describe().T
columns=df.columns
for col in columns:
    print(df[col].value_counts())
f,ax = plt.subplots(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, linewidths=0.5, linecolor="red", fmt= '.2f',ax=ax)
plt.show()
a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
df.head()
df = df.drop(columns = ['cp', 'thal', 'slope'])
df.head()
y = df.target.values
X_data = df.drop(['target'], axis = 1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(X_data)
X = scaler.transform(X_data)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)
list_model=["Linear Regression","Naive Bayes", "KNN","SVC_linear","SVC_rbf","Decision Tree","Random Forest"]

#all models
lr = LogisticRegression(solver = 'liblinear')
nb = GaussianNB()
knn = KNeighborsClassifier()
svc_linear = SVC(kernel='linear')
svc_rbf = SVC(kernel='rbf')
cart = DecisionTreeClassifier()
rf = RandomForestClassifier()

list_abr=[lr,nb,knn,svc_linear,svc_rbf,cart,rf]
accuracies={}
con_mat={}

for i in range(0,7):
    model=list_abr[i].fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    accuracies[list_model[i]] = acc
    con_mat[list_model[i]] = cm
accuracies
colors = ["purple", "green", "orange", "magenta","yellow","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()
plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

for i in range(1,8):

    plt.subplot(3,3,i)
    plt.title(list_model[i-1] + " Confusion Matrix")
    sns.heatmap(con_mat[list_model[i-1]],annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
    
plt.show()