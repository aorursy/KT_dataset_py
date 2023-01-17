from sklearn.datasets import load_digits

from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,mean_squared_error,confusion_matrix

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.model_selection import KFold,cross_val_score

import pandas as pd

import numpy as np

import os 

import sklearn.metrics as metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.naive_bayes import BernoulliNB,MultinomialNB,GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from scipy.spatial.distance import euclidean

from scikitplot.metrics import plot_roc_curve

plt.rcParams['figure.figsize']=[6,6]
x = np.load('../input/X.npy')

y = np.load('../input/Y.npy')
x.shape
imgs =[]

for x1 in x:

    imgs.append(x1.reshape(1,-1))
x1 = np.array(imgs).reshape(2062,4096)



x1.shape
y1 =[]

for h in range(len(y)):

    index=0

    for val in y[h]:

        if val!=1:

            index = index+1

        else:

            y1.append(index+1)



y1 = np.array(y1).reshape(-1,1)        

        
#Logistic Regression

reg = LogisticRegression()



x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size = 0.3,random_state = 42)

reg.fit(x_train,y_train)

y_predict = reg.predict(x_test)

acc_log = accuracy_score(y_test,y_predict)

acc_log = round(acc_log,2)

acc_log
#Decision Tree

cls = DecisionTreeClassifier(criterion='entropy',max_depth=19,min_samples_split=5)



cls.fit(x_train,y_train)

pred = cls.predict(x_test)

acc_tree = accuracy_score(y_test,pred)

acc_tree = round(acc_tree,2)

acc_tree
#Random Forest



rf = RandomForestClassifier(random_state=42,n_estimators=500,n_jobs=2)

rf.fit(x_train,y_train)



predict = rf.predict(x_test)

rf_acc = accuracy_score(predict,y_test)

rf_acc = round(rf_acc,2)

print(rf_acc)

important_features = rf.feature_importances_

#important_features

indices = np.argsort(important_features)[::-1]

indices


#Gridsearch CV

params1 = {'n_estimators':[int(x) for x in np.linspace(start=100,stop=1000,num=10)]}

rf_grid = GridSearchCV(rf,param_grid=params1,cv=3,scoring='accuracy')

rf_grid.fit(x_train,y_train)
rf_grid.best_params_
#Ada Boost



ab = AdaBoostClassifier(random_state=42,n_estimators=500)

ab.fit(x_train,y_train)

pcap = ab.predict(x_test)

ab_acc = accuracy_score(y_test,pcap)

ab_acc = round(ab_acc,2)

ab_acc
#Naive



bernouli = BernoulliNB()

bernouli.fit(x_train,y_train)

bernouli_pred = bernouli.predict(x_test)

bernouli_acc = accuracy_score(bernouli_pred,y_test)

bernouli_acc = round(bernouli_acc,2)

print(bernouli_acc)







multi = MultinomialNB()

multi.fit(x_train,y_train)

multi_pred = multi.predict(x_test)

multi_acc = accuracy_score(multi_pred,y_test)

multi_acc = round(multi_acc,2)

print(multi_acc)
g = np.array([acc_log,acc_tree,rf_acc,ab_acc,multi_acc,bernouli_acc])*100
c = np.array(['Logistic Regression','Decision Tree classifier','Random Forest Classifier','AdaBoost Classifier',

                         'MultinomialNB Classifier','BernouliNB Classifier'])
Classifier = pd.DataFrame(g,c)

Classifier.plot(kind ='bar',color='#FF9333')

plt.xlabel(c)



Classifier.columns = ['Accuracy']

Classifier
plot_roc_curve(y_test,rf.predict_proba(x_test),curves = 'each_class')
plot_roc_curve(predict,rf.predict_proba(x_test),curves = 'each_class')
kfold = KFold(n_splits=5,shuffle=True)

result = cross_val_score(rf,x_train,y_train,cv=kfold,scoring='accuracy')

result
result.mean()
a = plt.imread('../input/Assign.jpeg')

a.shape
plt.imshow(a)
a = a[:,:,0]

plt.imshow(a)
rf.predict(a.reshape(1,-1))
sample = pd.DataFrame(y_test,predict)
sample['original'] = sample.index 
sample = sample.reset_index()
sample =sample.drop(['index'],axis=1)
sample.columns = ['Predicted','original' ]

sample[(sample['Predicted']!=8) & (sample['original']==8)]