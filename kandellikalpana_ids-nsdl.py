import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("../input/traindata/Train_data.csv")
data.head(10)
data.describe()
data.info()
from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder

replace_map={'protocol_type' :{'tcp':1,'icmp':2,'udp':3}}
data.drop(['protocol_type'],axis=1,inplace=True)
replace_map={'xAttack' :{'normal':1,'r2l':2,'probe':3,'u2r':4,'dos':5}}
data.head(100)

from sklearn.model_selection import train_test_split

feature_col_names = ['duration','service','flag']

predicted_class_names = ['xAttack']

X= data[feature_col_names].values

Y= data[predicted_class_names].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,random_state=0)
from sklearn.svm import SVC

classifier = SVC(gamma='auto')

classifier.fit(X_train,Y_train)
prediction_test=classifier.predict(X_test)
print("accuracy on training data :{0:.4f}".format(classifier.score(X_test,Y_test)))
prediction_test=classifier.predict(X_test)

from sklearn.metrics import accuracy_score

score_svm1=accuracy_score(Y_test,prediction_test)

print(score_svm1)

print("Accuracy=",accuracy_score(Y_test,prediction_test)*100)
from sklearn.metrics import confusion_matrix

cm_svm=confusion_matrix(Y_test,prediction_test)
from sklearn.metrics import classification_report

print(classification_report(Y_test,prediction_test))

recall_svm= cm_svm[0][0]/(cm_svm[0][0] + cm_svm[0][1])

precision_svm = cm_svm[0][0]/(cm_svm[0][0]+cm_svm[1][1])

f1score_svm=((2 * recall_svm*precision_svm)/(recall_svm +precision_svm))

recall_svm,precision_svm,f1score_svm
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=3)

clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)

rf_score = clf.score(X_test,Y_test)

print('random forest processing ,,,')

print('random forest Score: %000.2f %%' % rf_score)
from sklearn.metrics import confusion_matrix

cm_rf=confusion_matrix(Y_test,prediction_test)
from sklearn.metrics import classification_report

print(classification_report(Y_test,y_pred))

recall_rf= cm_rf[0][0]/(cm_rf[0][0] + cm_rf[0][1])

precision_rf= cm_rf[0][0]/(cm_rf[0][0]+cm_rf[1][1])

f1score_rf=((2 * recall_rf*precision_rf)/(recall_rf+precision_rf))

recall_rf,precision_rf,f1score_rf
from sklearn.naive_bayes import GaussianNB
nb_model=GaussianNB()

nb_model.fit(X_train,Y_train)
y_pred=nb_model.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix

cm_nb= confusion_matrix(Y_test, y_pred)
nb_score = nb_model.score(X_test, Y_test)

print('navie bayes processing ,,,')

print('naive Score: %.2f %%' % nb_score)
from sklearn.metrics import classification_report

print(classification_report(Y_test,y_pred))

recall_nb= cm_nb[0][0]/(cm_nb[0][0] + cm_nb[0][1])

precision_nb= cm_nb[0][0]/(cm_nb[0][0]+cm_nb[1][1])

f1score_nb=((2 * recall_nb*precision_nb)/(recall_nb+precision_nb))

recall_nb,precision_nb,f1score_nb
!pip3 install xgboost
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train,Y_train)
xgb_score = model.score(X_test, Y_test)

print('extreme gradient processing ,,,')

print('XGB Score: %.2f %%' % xgb_score)
y_pred = clf.predict(X_test)

print(y_pred)
from sklearn import metrics

print("accuracy:",metrics.accuracy_score(Y_test, y_pred)*100)
from sklearn.metrics import confusion_matrix

cm_xgb=confusion_matrix(Y_test,prediction_test)
from sklearn.metrics import classification_report

print(classification_report(Y_test,y_pred))

recall_xgb= cm_xgb[0][0]/(cm_xgb[0][0] + cm_xgb[0][1])

precision_xgb= cm_xgb[0][0]/(cm_xgb[0][0]+cm_xgb[1][1])

f1score_xgb=((2 * recall_xgb*precision_xgb)/(recall_xgb+precision_xgb))

recall_xgb,precision_xgb,f1score_xgb
from sklearn.ensemble import VotingClassifier

model1 = RandomForestClassifier(random_state=1)

model2 = XGBClassifier(random_state=1)

model = VotingClassifier(estimators=[('svm', model1), ('xgb', model2)], voting='hard')

model.fit(X_train,Y_train)

model.score(X_test,Y_test)
scores = [score_svm1,nb_score,rf_score,xgb_score]

algorithms = ["Support Vector Machine","Naive Bayes","Random Forest","XGBoost"]    



for i in range(len(algorithms)):

    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
scores = [score_svm1,nb_score,rf_score,xgb_score]

algorithms = ["Support Vector Machine","Naive Bayes","Random Forest","XGBoost"]   
import matplotlib.pyplot as plt

import seaborn as sns

sns.lineplot(algorithms,scores)

sns.set(rc={'figure.figsize':(15,8)})

plt.xlabel("Algorithms")

plt.ylabel("Accuracy score")



sns.lineplot(algorithms,scores)
sns.set(rc={'figure.figsize':(15,8)})

plt.xlabel("Algorithms")

plt.ylabel("Accuracy score")



sns.barplot(algorithms,scores)
results ={'Accuracy':[score_svm1,nb_score,rf_score,xgb_score],

          'recall':[recall_svm,recall_nb,recall_rf,recall_xgb],

          'precision':[precision_svm,precision_nb,precision_rf,precision_xgb],

          'f1-score':[f1score_svm,f1score_nb,f1score_rf,f1score_xgb]}

index=['svm','nb','rf','xgb']
results =pd.DataFrame(results,index=index)
fig =results.plot(kind='bar',title='Comaprison of models',figsize =(15,8)).get_figure()

fig.savefig('Final Result.png')
plt.figure(figsize=(20,20)) 

sns.heatmap(data.corr(), annot=True, fmt='.0%')
feature_col_names = ['duration','service','flag']

predicted_class_names = ['xAttack']

X= data[feature_col_names].values

y= data[predicted_class_names].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.35,random_state=0)
from sklearn.svm import SVC

classifier = SVC(gamma='auto')

classifier.fit(X_train,y_train)
prediction_test=classifier.predict(X_test)
print("accuracy on training data :{0:.4f}".format(classifier.score(X_test,y_test)))
prediction_test=classifier.predict(X_test)

from sklearn.metrics import accuracy_score

score_svm1=accuracy_score(y_test,prediction_test)

print(score_svm1)

print("Accuracy=",accuracy_score(y_test,prediction_test)*100)
from sklearn.metrics import confusion_matrix

cm_svm=confusion_matrix(y_test,prediction_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,prediction_test))

recall_svm= cm_svm[0][0]/(cm_svm[0][0] + cm_svm[0][1])

precision_svm = cm_svm[0][0]/(cm_svm[0][0]+cm_svm[1][1])

f1score_svm=((2 * recall_svm*precision_svm)/(recall_svm +precision_svm))

recall_svm,precision_svm,f1score_svm
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
rf_score = clf.score(X_test, y_test)

print('random forest processing ,,,')

print('random forest Score: %.2f %%' % rf_score)
from sklearn.metrics import confusion_matrix

cm_rf= confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

recall_rf= cm_rf[0][0]/(cm_rf[0][0] + cm_rf[0][1])

precision_rf= cm_rf[0][0]/(cm_rf[0][0]+cm_rf[1][1])

f1score_rf=((2 * recall_rf*precision_rf)/(recall_rf+precision_rf))

recall_rf,precision_rf,f1score_rf
from sklearn.naive_bayes import GaussianNB
nb_model=GaussianNB()

nb_model.fit(X_train,y_train)
y_pred = clf.predict(X_test)
nb_score = nb_model.score(X_test, y_test)

print('navie bayes processing ,,,')

print('naive Score: %.2f %%' % nb_score)
from sklearn.metrics import confusion_matrix

cm_nb= confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

recall_nb= cm_nb[0][0]/(cm_nb[0][0] + cm_nb[0][1])

precision_nb= cm_nb[0][0]/(cm_nb[0][0]+cm_nb[1][1])

f1score_nb=((2 * recall_nb*precision_nb)/(recall_nb+precision_nb))

recall_nb,precision_nb,f1score_nb
!pip3 install xgboost
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train,y_train)
xgb_score = model.score(X_test, y_test)

print('extreme gradient processing ,,,')

print('XGB Score: %.2f %%' % xgb_score)
y_pred = clf.predict(X_test)

from sklearn import metrics

print("accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
from sklearn.metrics import confusion_matrix

cm_xgb=confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

recall_xgb= cm_xgb[0][0]/(cm_xgb[0][0] + cm_xgb[0][1])

precision_xgb= cm_xgb[0][0]/(cm_xgb[0][0]+cm_xgb[1][1])

f1score_xgb=((2 * recall_xgb*precision_xgb)/(recall_xgb+precision_xgb))

recall_xgb,precision_xgb,f1score_xgb
from sklearn.ensemble import VotingClassifier

model1 = RandomForestClassifier(random_state=1)

model2 = XGBClassifier(random_state=1)

model = VotingClassifier(estimators=[('svm', model1), ('xgb', model2)], voting='hard')

model.fit(X_train,y_train)

model.score(X_test,y_test)
scores = [score_svm1,nb_score,rf_score,xgb_score]

algorithms = ["Support Vector Machine","Naive Bayes","Random Forest","XGBoost"]    



for i in range(len(algorithms)):

    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
scores = [score_svm1,nb_score,rf_score,xgb_score]

algorithms = ["Support Vector Machine","Naive Bayes","Random Forest","XGBoost"]    
import matplotlib.pyplot as plt

import seaborn as sns

sns.lineplot(algorithms,scores)

sns.set(rc={'figure.figsize':(15,8)})

plt.xlabel("Algorithms")

plt.ylabel("Accuracy score")



sns.lineplot(algorithms,scores)
sns.set(rc={'figure.figsize':(15,8)})

plt.xlabel("Algorithms")

plt.ylabel("Accuracy score")



sns.barplot(algorithms,scores)
results ={'Accuracy':[score_svm1,nb_score,rf_score,xgb_score],

          'recall':[recall_svm,recall_nb,recall_rf,recall_xgb],

          'precision':[precision_svm,precision_nb,precision_rf,precision_xgb],

          'f1-score':[f1score_svm,f1score_nb,f1score_rf,f1score_xgb]}

index=['svm','nb','rf','xgb']
results =pd.DataFrame(results,index=index)
fig =results.plot(kind='bar',title='Comaprison of models',figsize =(15,8)).get_figure()

fig.savefig('Final Result.png')

plt.figure(figsize=(20,20)) 

sns.heatmap(data.corr(), annot=True, fmt='.0%')

from sklearn.model_selection import train_test_split

feature_col_names = ['duration','service','flag']

predicted_class_names = ['xAttack']

X= data[feature_col_names].values

y= data[predicted_class_names].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=0)
from sklearn.svm import SVC

classifier = SVC(gamma='auto')

classifier.fit(X_train,y_train)
prediction_test=classifier.predict(X_test)

print("accuracy on training data :{0:.4f}".format(classifier.score(X_test,y_test)))
prediction_test=classifier.predict(X_test)

from sklearn.metrics import accuracy_score

score_svm1=accuracy_score(y_test,prediction_test)

print(score_svm1)

print("Accuracy=",accuracy_score(y_test,prediction_test)*100)
from sklearn.metrics import confusion_matrix

cm_svm=confusion_matrix(y_test,prediction_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,prediction_test))

recall_svm= cm_svm[0][0]/(cm_svm[0][0] + cm_svm[0][1])

precision_svm = cm_svm[0][0]/(cm_svm[0][0]+cm_svm[1][1])

f1score_svm=((2 * recall_svm*precision_svm)/(recall_svm +precision_svm))

recall_svm,precision_svm,f1score_svm
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

rf_score = clf.score(X_test, y_test)

print('random forest processing ,,,')

print('random forest Score: %.2f %%' % rf_score)
from sklearn.metrics import confusion_matrix

cm_rf= confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

recall_rf= cm_rf[0][0]/(cm_rf[0][0] + cm_rf[0][1])

precision_rf= cm_rf[0][0]/(cm_rf[0][0]+cm_rf[1][1])

f1score_rf=((2 * recall_rf*precision_rf)/(recall_rf+precision_rf))

recall_rf,precision_rf,f1score_rf
from sklearn.naive_bayes import GaussianNB
nb_model=GaussianNB()

nb_model.fit(X_train,y_train)
y_pred = clf.predict(X_test)
nb_score = nb_model.score(X_test, y_test)

print('navie bayes processing ,,,')

print('naive Score: %.2f %%' % nb_score)
from sklearn.metrics import confusion_matrix

cm_nb=confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

recall_nb= cm_nb[0][0]/(cm_nb[0][0] + cm_nb[0][1])

precision_nb= cm_nb[0][0]/(cm_nb[0][0]+cm_nb[1][1])

f1score_nb=((2 * recall_nb*precision_nb)/(recall_nb+precision_nb))

recall_nb,precision_nb,f1score_nb
!pip3 install xgboost
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train,y_train)

xgb_score = model.score(X_test, y_test)

print('extreme gradient processing ,,,')

print('XGB Score: %.2f %%' % xgb_score)

y_pred = clf.predict(X_test)
from sklearn import metrics

print("accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
from sklearn.metrics import confusion_matrix

cm_xgb=confusion_matrix(y_test,y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

recall_xgb= cm_xgb[0][0]/(cm_xgb[0][0] + cm_xgb[0][1])

precision_xgb= cm_xgb[0][0]/(cm_xgb[0][0]+cm_xgb[1][1])

f1score_xgb=((2 * recall_xgb*precision_xgb)/(recall_xgb+precision_xgb))

recall_xgb,precision_xgb,f1score_xgb
from sklearn.ensemble import VotingClassifier

model1 = RandomForestClassifier(random_state=1)

model2 = XGBClassifier(random_state=1)

model = VotingClassifier(estimators=[('svm', model1), ('xgb', model2)], voting='hard')

model.fit(X_train,y_train)

model.score(X_test,y_test)
scores = [score_svm1,nb_score,rf_score,xgb_score]

algorithms = ["Support Vector Machine","Naive Bayes","Random Forest","XGBoost"]    



for i in range(len(algorithms)):

    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
scores = [score_svm1,nb_score,rf_score,xgb_score]

algorithms = ["Support Vector Machine","Naive Bayes","Random Forest","XGBoost"]    
import matplotlib.pyplot as plt

import seaborn as sns

sns.lineplot(algorithms,scores)

sns.set(rc={'figure.figsize':(15,8)})

plt.xlabel("Algorithms")

plt.ylabel("Accuracy score")



sns.lineplot(algorithms,scores)

sns.set(rc={'figure.figsize':(15,8)})

plt.xlabel("Algorithms")

plt.ylabel("Accuracy score")



sns.barplot(algorithms,scores)
results ={'Accuracy':[score_svm1,nb_score,rf_score,xgb_score],

          'recall':[recall_svm,recall_nb,recall_rf,recall_xgb],

          'precision':[precision_svm,precision_nb,precision_rf,precision_xgb],

          'f1-score':[f1score_svm,f1score_nb,f1score_rf,f1score_xgb]}

index=['svm','nb','rf','xgb']
results =pd.DataFrame(results,index=index)
fig =results.plot(kind='bar',title='Comaprison of models',figsize =(15,8)).get_figure()

fig.savefig('Final Result.png')

plt.figure(figsize=(20,20)) 

sns.heatmap(data.corr(), annot=True, fmt='.0%')