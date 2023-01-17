import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# Load dataset
df = pd.read_csv(r"../input/car_evaluation.csv", names = ["buying","maint", "doors", "persons", "lug_boot","safety","class"])
df.head()
cleanup_nums = {"class":     {"unacc": 4, "acc": 3,'good': 2,'vgood':1}
                }
df.replace(cleanup_nums,inplace = True)
target = df['class']
df.drop( ['class'],axis = 1,inplace = True)
df = pd.get_dummies(df)
df.head()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df,target,random_state = 0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
from sklearn.linear_model import Perceptron
ppn=Perceptron(n_iter=3501, eta0=0.1, random_state=0)
ppn.fit(X_train_std,Y_train)
from sklearn import svm
from sklearn.svm import SVC
svc = svm.SVC(kernel='linear', C=1).fit(X_train_std,Y_train)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
model.score(X_train_std,Y_train)
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
from sklearn import tree
model1 = tree.DecisionTreeClassifier(criterion='gini')
model1.fit(X_train,Y_train)
model.score(X_train,Y_train)
from sklearn.neighbors import KNeighborsClassifier
model2=KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train,Y_train)
y_pred=ppn.predict(X_test_std)
print('Misclassified samples using Perceptron are: %d' %(Y_test!=y_pred).sum())
from sklearn.metrics import accuracy_score
print('Classification Accuracy of Perceptron is %.2f ' %accuracy_score(Y_test,y_pred))
s_pred=svc.predict(X_test_std)
print('Misclassified samples using SVM are: %d' %(Y_test!=s_pred).sum())
print('Classification Accuracy of SVM is %.2f ' %accuracy_score(Y_test,s_pred))
lr_pred= model.predict(X_test)
print('Misclassified samples using Logistic Regression are: %d' %(Y_test!=lr_pred).sum())
print('Classification Accuracy of Logistic Regression is %.2f ' %accuracy_score(Y_test,lr_pred))
t_pred= model1.predict(X_test)
print('Misclassified samples using Trees are: %d' %(Y_test!=t_pred).sum())
print('Classification Accuracy of Decision trees is %.2f ' %accuracy_score(Y_test,t_pred))
k_pred= model2.predict(X_test)
print('Misclassified samples using KNN are: %d' %(Y_test!=k_pred).sum())
print('Classification Accuracy of KNN is %.2f ' %accuracy_score(Y_test,k_pred))
