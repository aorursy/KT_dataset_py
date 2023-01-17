#loading dataset

import pandas as pd

import numpy as np

#visualisation

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

#EDA

from collections import Counter

import pandas_profiling as pp

# data preprocessing

from sklearn.preprocessing import StandardScaler

# data splitting

from sklearn.model_selection import train_test_split

# data modeling

from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

#ensembling

from mlxtend.classifier import StackingCVClassifier
data = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')

data.head()
data.info()
pp.ProfileReport(data)
y = data["target"]

X = data.drop('target',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
print(y_test.unique())

Counter(y_train)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
m1 = 'Logistic Regression'

lr = LogisticRegression()

model = lr.fit(X_train, y_train)

lr_predict = lr.predict(X_test)

lr_conf_matrix = confusion_matrix(y_test, lr_predict)

lr_acc_score = accuracy_score(y_test, lr_predict)

print("confussion matrix")

print(lr_conf_matrix)

print("\n")

print("Accuracy of Logistic Regression:",lr_acc_score*100,'\n')

print(classification_report(y_test,lr_predict))
m2 = 'Naive Bayes'

nb = GaussianNB()

nb.fit(X_train,y_train)

nbpred = nb.predict(X_test)

nb_conf_matrix = confusion_matrix(y_test, nbpred)

nb_acc_score = accuracy_score(y_test, nbpred)

print("confussion matrix")

print(nb_conf_matrix)

print("\n")

print("Accuracy of Naive Bayes model:",nb_acc_score*100,'\n')

print(classification_report(y_test,nbpred))
m3 = 'Random Forest Classfier'

rf = RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5)

rf.fit(X_train,y_train)

rf_predicted = rf.predict(X_test)

rf_conf_matrix = confusion_matrix(y_test, rf_predicted)

rf_acc_score = accuracy_score(y_test, rf_predicted)

print("confussion matrix")

print(rf_conf_matrix)

print("\n")

print("Accuracy of Random Forest:",rf_acc_score*100,'\n')

print(classification_report(y_test,rf_predicted))
m4 = 'Extreme Gradient Boost'

xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 

                    reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)

xgb.fit(X_train, y_train)

xgb_predicted = xgb.predict(X_test)

xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)

xgb_acc_score = accuracy_score(y_test, xgb_predicted)

print("confussion matrix")

print(xgb_conf_matrix)

print("\n")

print("Accuracy of Extreme Gradient Boost:",xgb_acc_score*100,'\n')

print(classification_report(y_test,xgb_predicted))
m5 = 'K-NeighborsClassifier'

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)

knn_predicted = knn.predict(X_test)

knn_conf_matrix = confusion_matrix(y_test, knn_predicted)

knn_acc_score = accuracy_score(y_test, knn_predicted)

print("confussion matrix")

print(knn_conf_matrix)

print("\n")

print("Accuracy of K-NeighborsClassifier:",knn_acc_score*100,'\n')

print(classification_report(y_test,knn_predicted))
m6 = 'DecisionTreeClassifier'

dt = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)

dt.fit(X_train, y_train)

dt_predicted = dt.predict(X_test)

dt_conf_matrix = confusion_matrix(y_test, dt_predicted)

dt_acc_score = accuracy_score(y_test, dt_predicted)

print("confussion matrix")

print(dt_conf_matrix)

print("\n")

print("Accuracy of DecisionTreeClassifier:",dt_acc_score*100,'\n')

print(classification_report(y_test,dt_predicted))
m7 = 'Support Vector Classifier'

svc =  SVC(kernel='rbf', C=2)

svc.fit(X_train, y_train)

svc_predicted = svc.predict(X_test)

svc_conf_matrix = confusion_matrix(y_test, svc_predicted)

svc_acc_score = accuracy_score(y_test, svc_predicted)

print("confussion matrix")

print(svc_conf_matrix)

print("\n")

print("Accuracy of Support Vector Classifier:",svc_acc_score*100,'\n')

print(classification_report(y_test,svc_predicted))
imp_feature = pd.DataFrame({'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal'], 'Importance': xgb.feature_importances_})

plt.figure(figsize=(10,4))

plt.title("barplot Represent feature importance ")

plt.xlabel("importance ")

plt.ylabel("features")

plt.barh(imp_feature['Feature'],imp_feature['Importance'],color = 'rgbkymc')

plt.show()
lr_false_positive_rate,lr_true_positive_rate,lr_threshold = roc_curve(y_test,lr_predict)

nb_false_positive_rate,nb_true_positive_rate,nb_threshold = roc_curve(y_test,nbpred)

rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,rf_predicted)                                                             

xgb_false_positive_rate,xgb_true_positive_rate,xgb_threshold = roc_curve(y_test,xgb_predicted)

knn_false_positive_rate,knn_true_positive_rate,knn_threshold = roc_curve(y_test,knn_predicted)

dt_false_positive_rate,dt_true_positive_rate,dt_threshold = roc_curve(y_test,dt_predicted)

svc_false_positive_rate,svc_true_positive_rate,svc_threshold = roc_curve(y_test,svc_predicted)





sns.set_style('whitegrid')

plt.figure(figsize=(10,5))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(lr_false_positive_rate,lr_true_positive_rate,label='Logistic Regression')

plt.plot(nb_false_positive_rate,nb_true_positive_rate,label='Naive Bayes')

plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Random Forest')

plt.plot(xgb_false_positive_rate,xgb_true_positive_rate,label='Extreme Gradient Boost')

plt.plot(knn_false_positive_rate,knn_true_positive_rate,label='K-Nearest Neighbor')

plt.plot(dt_false_positive_rate,dt_true_positive_rate,label='Desion Tree')

plt.plot(svc_false_positive_rate,svc_true_positive_rate,label='Support Vector Classifier')

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.legend()

plt.show()
model_ev = pd.DataFrame({'Model': ['Logistic Regression','Naive Bayes','Random Forest','Extreme Gradient Boost',

                    'K-Nearest Neighbour','Decision Tree','Support Vector Machine'], 'Accuracy': [lr_acc_score*100,

                    nb_acc_score*100,rf_acc_score*100,xgb_acc_score*100,knn_acc_score*100,dt_acc_score*100,svc_acc_score*100]})

model_ev
colors = ['red','green','blue','gold','silver','yellow','orange',]

plt.figure(figsize=(12,5))

plt.title("barplot Represent Accuracy of different models")

plt.xlabel("Accuracy %")

plt.ylabel("Algorithms")

plt.bar(model_ev['Model'],model_ev['Accuracy'],color = colors)

plt.show()
scv=StackingCVClassifier(classifiers=[xgb,knn,svc],meta_classifier= svc,random_state=42)

scv.fit(X_train,y_train)

scv_predicted = scv.predict(X_test)

scv_conf_matrix = confusion_matrix(y_test, scv_predicted)

scv_acc_score = accuracy_score(y_test, scv_predicted)

print("confussion matrix")

print(scv_conf_matrix)

print("\n")

print("Accuracy of StackingCVClassifier:",scv_acc_score*100,'\n')

print(classification_report(y_test,scv_predicted))