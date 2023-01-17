# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df.corr()
df.isnull().sum()
df['Pregnancies'].value_counts()
df['Outcome'].value_counts()
plt.figure(figsize=(10,6))
sns.countplot(df["Age"],palette="muted")
df.nunique()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
one=StandardScaler()
X=one.fit_transform(X)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(x_train,y_train)
pred_svc =svc.predict(x_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, pred_svc))
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,pred_svc))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(x_train,y_train)
pred_knn=knn.predict(x_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, pred_knn))
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,pred_knn))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred_knn)
print("Precision:",metrics.precision_score(y_test, pred_knn))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, pred_knn))
from sklearn import naive_bayes
NB = naive_bayes.GaussianNB()
NB.fit(x_train,y_train)
pred_nb=NB.predict(x_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, pred_nb))
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,pred_nb))
print("Precision:",metrics.precision_score(y_test, pred_nb))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, pred_nb))
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
pred_dt=dt.predict(x_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, pred_dt))
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,pred_dt))
print("Precision:",metrics.precision_score(y_test, pred_dt))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, pred_dt))
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state = 42)
clf_gini = DecisionTreeClassifier(criterion="gini", random_state = 42)
clf_entropy.fit(x_train,y_train)
pred_clf_entropy=clf_entropy.predict(x_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, pred_clf_entropy))
clf_gini.fit(x_train,y_train)
pred_clf_gini=clf_gini.predict(x_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, pred_clf_gini))

X1=df.iloc[:,:-1].values
Y1=df.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
sk=StandardScaler()
X1=sk.fit_transform(X1)
Y1=Y1.reshape(-1,1)
Y1
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, test_size = 0.1,random_state=42)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val  = train_test_split(x_train, y_train, test_size = 0.1,random_state=42)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu,softmax
from keras.regularizers import l2
model = Sequential()
model.add(Dense(16, input_dim=8,kernel_regularizer=l2(0.01), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(48, kernel_regularizer=l2(0.01),activation='relu'))
model.add(Dense(64, kernel_regularizer=l2(0.01),activation='relu',))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint
checkpointer=ModelCheckpoint(filepath='Convolutional.hdf5',verbose=1,save_best_only=True)
history = model.fit(x_train, y_train, epochs=100, batch_size=16,validation_data=(x_val,y_val))
score=model.evaluate(x_test,y_test,verbose=1)               #evaluates the model
accuracy=100*score[1]                                       
print('Test accuracy is %.4f%%' % accuracy)
score=model.evaluate(x_train,y_train,verbose=1)               #evaluates the model
accuracy=100*score[1]                                       
print('Test accuracy is %.4f%%' % accuracy)
