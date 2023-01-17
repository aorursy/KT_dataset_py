import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.model_selection import  cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



from sklearn.metrics import accuracy_score

%matplotlib inline
data=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

data.head()
data.isnull().sum()
data.describe()
data.columns
sns.countplot(data=data ,x="Outcome",hue="Outcome")

plt.title("Womans have diabetes")
fig, ax=plt.subplots(figsize=(5,5))

sns.boxplot(y="Age",x='Outcome',hue='Outcome',data=data)

plt.title(" Age ")
for i in data.columns:

    plt.figsize=(12,10)

    plt.hist(data[i])

    plt.title(i)

    plt.show()
sns.lineplot(data=data ,x='Age',hue='Age',y="Glucose")
sns.pairplot(x_vars=["Glucose","Pregnancies","BMI"],y_vars="Age",hue="Outcome",data=data)
sns.pairplot(data=data,hue="Outcome")
#Use correlations

corr_matrix=data.corr()

corr_matrix
import seaborn as sns

#get correlations of each features in dataset

corr_matrix = data.corr()

top_corr_features = corr_matrix.index

plt.figure(figsize=(8,6))

#plot heat map

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
x=data[["Pregnancies","BloodPressure","Glucose","Insulin","BMI","DiabetesPedigreeFunction","Age"]]

col=["Pregnancies","Glucose","BloodPressure","Insulin","BMI","DiabetesPedigreeFunction","Age"]

from sklearn.preprocessing import StandardScaler

X_scaler=StandardScaler()

X=X_scaler.fit_transform(x)

scaled_features_df = pd.DataFrame(X,columns=col)
y=data.Outcome

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(scaled_features_df,y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

log_clf = LogisticRegression()

rnd_clf = RandomForestClassifier()

svm_clf = SVC()

tree_clf = DecisionTreeClassifier( max_depth=4, criterion='gini')

voting_clf = VotingClassifier(

estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf), ('tree', tree_clf)],

voting='hard')

voting_clf.fit(x_train, y_train)



from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf,tree_clf, voting_clf):

    clf.fit(x_train,y_train)

    y_pred = clf.predict(x_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
ax = sns.regplot(x="Glucose", y="Outcome", data=data,

                 logistic=True, n_boot=500, y_jitter=.03)
from sklearn.linear_model import LogisticRegression

lr_clf= LogisticRegression()

lr_clf.fit(x_train,y_train)

pred=lr_clf.predict(x_test)

print("Accuracy for Logistic Regression on  data: ",accuracy_score(y_test,pred))

#  0.8246753246753247
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score,recall_score,f1_score

print ("confusion_matrix:",confusion_matrix(y_test,pred))

print("precision:",precision_score(y_test,pred))

print("recall:",recall_score(y_test,pred))

print("f1_score :",f1_score(y_test,pred))
from sklearn.metrics import roc_curve, auc



logistic_fpr, logistic_tpr, threshold = roc_curve(y_test,pred)

auc_logistic = auc(logistic_fpr, logistic_tpr)







plt.figure(figsize=(5, 5), dpi=100)

plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)



plt.xlabel('False Positive Rate -->')

plt.ylabel('True Positive Rate -->')



plt.legend()



plt.show()