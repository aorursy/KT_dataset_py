import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import os
data=pd.read_csv('../input/creditcard.csv')
print(type(data))

#print(datadf.columns)
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
labels = 'No Fraud', 'Fraud'
sizes = [count_classes[1]/(count_classes[1]+count_classes[0]), count_classes[0]/(count_classes[1]+count_classes[0])]
explode = (0, 0.5,)  # only "explode" the 2nd slice (i.e. 'Fraud')
colors = ['orange', 'darkblue']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, colors=colors, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=45)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Fraud class repartition")
plt.show()
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


data['normAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
Y=data['Class']
X=data.drop(['Time','Amount','Class'],axis=1)
## Stratify /!\
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.3, random_state=42,stratify=Y)

#return 
# print(Y_train.isnull().sum().sum())


from sklearn import metrics

sgd_clf=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=42, shuffle=True,
       tol=None, verbose=0, warm_start=False)

sgd_clf.fit(X_train, Y_train) 
Y_train_predicted=sgd_clf.predict(X_train)
Y_test_predicted=sgd_clf.predict(X_test)


from sklearn.model_selection import cross_validate

## Cross Validation
scoring = ['precision', 'recall']
scores = cross_validate(sgd_clf, X_train, Y_train, scoring=scoring, cv=5, return_train_score=False,)




from sklearn.metrics import precision_score, recall_score

print(" ### TRAINING SET ###")
print("Recall : " + str(scores["test_recall"].mean()) +"  | Precision : " +str(scores["test_precision"].mean()))
print(" ### TEST SET ###")
print("Recall : " + str(recall_score(Y_test,Y_test_predicted)) +"  | Precision : " +str(precision_score(Y_test,Y_test_predicted)))

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(Y_test,Y_test_predicted)

class_names = [0,1]
plt.figure()
plot_confusion_matrix(confusion
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
from sklearn.utils import shuffle

Train_Data= pd.concat([X_train, Y_train], axis=1)
## On peut encore faire des trucs cool ici tu dois separer les training et etst data 1  et en dessous tu dois plus prendre de test data vu que tu en as deja tu utiliseras juste de la ceoss validation v
X_1 =Train_Data[ Train_Data["Class"]==1 ]
X_0=Train_Data[Train_Data["Class"]==0]

X_0=shuffle(X_0,random_state=42).reset_index(drop=True)
X_1=shuffle(X_1,random_state=42).reset_index(drop=True)

ALPHA=1.4

X_0=X_0.iloc[:round(len(X_1)*ALPHA),:]
data_d=pd.concat([X_1, X_0])

count_classes = pd.value_counts(data_d['Class'], sort = True).sort_index()
labels = 'No Fraud', 'Fraud'
sizes = [count_classes[1]/(count_classes[1]+count_classes[0]), count_classes[0]/(count_classes[1]+count_classes[0])]
explode = (0, 0.05,)  # only "explode" the 2nd slice (i.e. 'Fraud')
colors = ['orange', 'darkblue']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, colors=colors, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=45)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Fraud class repartition")
plt.show()

Y_d=Train_Data['Class']
X_d=Train_Data.drop(['Class'],axis=1)
Train_Data.head()

sgd_clf_d=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=42, shuffle=True,
       tol=None, verbose=0, warm_start=False)

sgd_clf_d.fit(X_d, Y_d) 

# Cross validation

scoring = ['precision', 'recall']
scores_d = cross_validate(sgd_clf, X_d, Y_d, scoring=scoring, cv=5, return_train_score=False)

Y_test_predicted=sgd_clf_d.predict(X_test)

print(" ### TRAINING SET ###")
print("Recall : " + str(scores_d["test_recall"].mean()) +"  | Precision : " +str(scores_d["test_precision"].mean()))
print(" ### TEST SET ###")
print("Recall : " + str(recall_score(Y_test,Y_test_predicted)) +"  | Precision : " +str(precision_score(Y_test,Y_test_predicted)))

confusion=confusion_matrix(Y_test,Y_test_predicted)


class_names = [0,1]
plt.figure()
plot_confusion_matrix(confusion
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
from sklearn.utils import shuffle

X_1 =Train_Data[ Train_Data["Class"]==1 ]
X_0=Train_Data[Train_Data["Class"]==0]

X_0=shuffle(X_0,random_state=42).reset_index(drop=True)
X_1=shuffle(X_1,random_state=42).reset_index(drop=True)

scoring = ['precision', 'recall']

ALPHA=0
alpha_array= np.array([])
precision_array= np.array([])
recall_array= np.array([])

d_alpha_array= np.array([])
d_precision_array= np.array([])
d_recall_array= np.array([])


                                                       
for ALPHA in np.arange(10,200,1):
    
    X_0=Train_Data[Train_Data["Class"]==0]
    X_0=X_0.iloc[:int(len(X_1)*ALPHA/10),:]
    
    data_d=pd.concat([X_1, X_0], axis=0)
    data_d=shuffle(data_d,random_state=42).reset_index(drop=True)
    Y_d=data_d['Class']
    X_d=data_d.drop(['Class'],axis=1)
    
    

    sgd_clf_d=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
           eta0=0.0, fit_intercept=True, l1_ratio=0.15,
           learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
           n_jobs=1, penalty='l2', power_t=0.5, random_state=42, shuffle=True,
           tol=None, verbose=0, warm_start=False)

    sgd_clf_d.fit(X_d, Y_d) 


    scores_d = cross_validate(sgd_clf, X_d, Y_d, scoring=scoring, cv=5, return_train_score=False)
    Y_predicted=sgd_clf_d.predict(X_test)
    
                                                           
    alpha_array=np.append(alpha_array,ALPHA/10)
    precision_array =np.append(precision_array,scores_d["test_precision"].mean())
    recall_array=np.append(recall_array,scores_d["test_recall"].mean())
    
scoreF1_array=(2*(recall_array*precision_array)/(recall_array+precision_array))


plt.plot(alpha_array, recall_array,label="Recall")
plt.plot(alpha_array,precision_array,label="Precision")
plt.plot(alpha_array,scoreF1_array,label="ScoreF1")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


    
Best_index=np.argmax(scoreF1_array)
print(" Best ALPHA :",alpha_array[Best_index]," Recall :",recall_array[Best_index]," Precision :", precision_array[Best_index])

ALPHA=alpha_array[Best_index]

X_1 =data[ data["Class"]==1 ]
X_0=data[data["Class"]==0]
X_0=shuffle(X_0,random_state=42).reset_index(drop=True)
X_1=shuffle(X_1,random_state=42).reset_index(drop=True)

X_0=data[data["Class"]==0]
X_0=X_0.iloc[:int(len(X_1)*ALPHA),:]
data_d=pd.concat([X_1, X_0])

data_d['normAmount']=StandardScaler().fit_transform(data_d['Amount'].values.reshape(-1,1))
Y_d=data_d['Class']
X_d=data_d.drop(['Time','Amount','Class'],axis=1)

X_d_train, X_d_test, Y_d_train, Y_d_test = train_test_split(X_d, Y_d , test_size=0.3, random_state=42)

sgd_clf_d=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
eta0=0.0, fit_intercept=True, l1_ratio=0.15,
learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
n_jobs=1, penalty='l2', power_t=0.5, random_state=42, shuffle=True,tol=None, verbose=0, warm_start=False)

sgd_clf_d.fit(X_d_train, Y_d_train) 

Y_test_predicted=sgd_clf_d.predict(X_test)

print("Recall : " + str(recall_score(Y_test,Y_test_predicted)) +"  | Precision : " +str(precision_score(Y_test,Y_test_predicted)))
confusion=confusion_matrix(Y_test,Y_test_predicted)

class_names = [0,1]
plt.figure()
plot_confusion_matrix(confusion
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()



ALPHA=15

X_1 =data[ data["Class"]==1 ]
X_0=data[data["Class"]==0]
X_0=shuffle(X_0,random_state=42).reset_index(drop=True)
X_1=shuffle(X_1,random_state=42).reset_index(drop=True)
len_X_1=len(X_1)
X_1=pd.concat([X_1,X_1, X_1])

X_0=data[data["Class"]==0]
X_0=X_0.loc[:np.round(len_X_1*ALPHA),:]
data_d=pd.concat([X_1, X_0])

data_d['normAmount']=StandardScaler().fit_transform(data_d['Amount'].values.reshape(-1,1))
Y_d=data_d['Class']
X_d=data_d.drop(['Time','Amount','Class'],axis=1)

X_d_train, X_d_test, Y_d_train, Y_d_test = train_test_split(X_d, Y_d , test_size=0.3, random_state=42)

sgd_clf_d=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
eta0=0.0, fit_intercept=True, l1_ratio=0.15,
learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
n_jobs=1, penalty='l2', power_t=0.5, random_state=42, shuffle=True,tol=None, verbose=0, warm_start=False)

sgd_clf_d.fit(X_d_train, Y_d_train) 

Y_test_predicted=sgd_clf_d.predict(X_test)

print("Recall : " + str(recall_score(Y_test,Y_test_predicted)) +"  | Precision : " +str(precision_score(Y_test,Y_test_predicted)))
confusion=confusion_matrix(Y_test,Y_test_predicted)

class_names = [0,1]
plt.figure()
plot_confusion_matrix(confusion
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


