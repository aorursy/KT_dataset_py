import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

data.head()
data.info()
data.columns
data.isnull().sum()

sns.set_style('darkgrid')

sns.countplot(x='diagnosis',data=data)
data.drop('Unnamed: 32',axis=1,inplace=True)

data.columns
data.shape
sns.heatmap(data.corr())


x=data.iloc[:,2:32].values

y=data.iloc[:,1].values
print(y)


from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

y=lb.fit_transform(y)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

x_train=sc_x.fit_transform(x_train)

x_test=sc_x.transform(x_test)
x.shape

from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve,roc_auc_score
from sklearn.linear_model import LogisticRegression

lg=LogisticRegression()

lg.fit(x_train,y_train)

y_pred=lg.predict(x_test)

#auc_roc=metrics.classification_report(y_test,y_pred)

#print(auc_roc)
def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    if train:

        print("Train Result:")

        print("------------")

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))

        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        print("ROC Score: \n {}\n".format(roc_auc_score(y_train, clf.predict(X_train))))



        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')

        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))

        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

        print("---------------------------------------------------------------------------------------------")

        

    elif train==False:

        print("Test Result:")

        print("-----------")

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))

        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))   

        print("ROC Score: \n {}\n".format(roc_auc_score(y_test, clf.predict(X_test))))

        print("---------------------------------------------------------------------------------------------")
print_score(lg,x_train,y_train,x_test,y_test,train=True)

print_score(lg,x_train,y_train,x_test,y_test,train=False)
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'Logistic Regression AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=10)

knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

print(roc_auc_score(y_test,y_pred))
error_rate=[]

for i in range(1,40):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_i=knn.predict(x_test)

    #print(pred_i)

    #print(y_test)

    error_rate.append(np.mean(pred_i!=y_test))

error_rate


plt.figure(figsize=(12,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title("error rate vs k-value")

plt.xlabel('k')

plt.ylabel('error rate')
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=15)

knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

print_score(knn,x_train,y_train,x_test,y_test,train=True)

print_score(knn,x_train,y_train,x_test,y_test,train=False)
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'KNN AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=10,criterion='entropy')

rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)



print_score(rf,x_train,y_train,x_test,y_test,train=True)

print_score(rf,x_train,y_train,x_test,y_test,train=False)
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'Random Foresrt AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.svm import SVC

model=SVC()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)



from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))


from sklearn.model_selection import GridSearchCV

parameters=[{'C':[1,10,100,1000],'kernel':['linear']},

           {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.001,0.0001]},

           ]

grid_search=GridSearchCV(estimator=model,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)

grid_search=grid_search.fit(x_train,y_train)
best_accuracy=grid_search.best_score_

print(best_accuracy)
best_param=grid_search.best_params_

print(best_param)
grid_search_prediction=grid_search.predict(x_test)

print_score(grid_search,x_train,y_train,x_test,y_test,train=True)

print_score(grid_search,x_train,y_train,x_test,y_test,train=False)
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, grid_search_prediction)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'Random Foresrt AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
import keras

from keras.models import Sequential

from keras.layers import Dense
classifier=Sequential()

classifier.add(Dense(units=32, kernel_initializer='uniform',activation='relu',input_dim=30))

classifier.add(Dense(units=6, kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(units=1, kernel_initializer='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

classifier.fit(x_train,y_train,batch_size=10,epochs=100)

y_pred=classifier.predict(x_test)>0.5
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

print(cm)