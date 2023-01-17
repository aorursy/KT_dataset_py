import pandas as pd
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#train=pd.read_csv("C://Users//ULTRON//Videos//ML competetions//mobile classification//train.csv")
train=pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
train.head()
train.isna().sum()
train.dtypes
from sklearn.ensemble import RandomForestClassifier
train.columns
##Simplest Random forest classifier 
X=train[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi']]
Y=train["price_range"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier() #We will not tune anythinga nd will use the default values for this
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
y_pred
train["price_range"].value_counts
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, y_pred)
from sklearn import metrics
metrics.accuracy_score(Y_test, y_pred)
##Setting up First feature that is n estimator for random forest 
print(train.shape)
##Setting the n-estimator as sqrt(2000)
import math
print(math.sqrt(2000))

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=45,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False) #We will not tune anythinga nd will use the default values for this
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, y_pred))
from sklearn import metrics
print(metrics.accuracy_score(Y_test, y_pred))
##Improve the accuracy by 1%
##No pass different estimator and calulate accuracy 
n_est=[30,45,90,100,150,200,250,300,350,400]
accuracy=[]
for i in n_est:
    clf=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=i,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False) #We will not tune anythinga nd will use the default values for this
    clf.fit(X_train,Y_train)
    y_pred=clf.predict(X_test)
#from sklearn.metrics import confusion_matrix
    print(confusion_matrix(Y_test, y_pred))
#from sklearn import metrics
    print("Accuracy for ",i,"estimator ",metrics.accuracy_score(Y_test, y_pred))
    accuracy.append(metrics.accuracy_score(Y_test, y_pred))
##To check the variabliey of accuracy as depend on the number of estimators
import matplotlib.pyplot as plt
plt.plot(n_est,accuracy)
plt.ylabel("Accuracy")
plt.xlabel("No of estimator")
crit=["gini","entropy"]
accuracy=[]
for i in crit:
    clf=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion=i, max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=150,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False) #We will not tune anythinga nd will use the default values for this
    clf.fit(X_train,Y_train)
    y_pred=clf.predict(X_test)
#from sklearn.metrics import confusion_matrix
    print(confusion_matrix(Y_test, y_pred))
#from sklearn import metrics
    print("Accuracy for ",i,"criterion ",metrics.accuracy_score(Y_test, y_pred))
    accuracy.append(metrics.accuracy_score(Y_test, y_pred))
max_depths = np.linspace(1, 32, 32, endpoint=True)
accuracy=[]
for i in max_depths:
    clf=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion="gini", max_depth=i, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=150,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False) #We will not tune anythinga nd will use the default values for this
    clf.fit(X_train,Y_train)
    y_pred=clf.predict(X_test)
#from sklearn.metrics import confusion_matrix
    print(confusion_matrix(Y_test, y_pred))
#from sklearn import metrics
    print("Accuracy for ",i,"depth ",metrics.accuracy_score(Y_test, y_pred))
    accuracy.append(metrics.accuracy_score(Y_test, y_pred))
## To plot accuracy vs depth of tree plot
import matplotlib.pyplot as plt
plt.plot(max_depths,accuracy)
plt.ylabel("Accuracy")
plt.xlabel("max_depths")
##To calucate the accuracy based on max feature 
fea=["auto","sqrt","log2"]
accuracy=[]
for i in fea:
    clf=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion="gini", max_depth=12, max_features=i,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=150,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False) #We will not tune anythinga nd will use the default values for this
    clf.fit(X_train,Y_train)
    y_pred=clf.predict(X_test)
#from sklearn.metrics import confusion_matrix
    print(confusion_matrix(Y_test, y_pred))
#from sklearn import metrics
    print("Accuracy for ",i,"featuere",metrics.accuracy_score(Y_test, y_pred))
    accuracy.append(metrics.accuracy_score(Y_test, y_pred))
## To plot accuracy vs feature of tree plot
import matplotlib.pyplot as plt
plt.bar(fea,accuracy)
plt.ylabel("Accuracy")
plt.xlabel("feature")
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'n_estimators': [ 100, 150],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [10,12,13],
    'criterion' :crit
}
rfc=RandomForestClassifier(random_state=42,oob_score=True)
C_rfc=GridSearchCV(estimator=rfc,param_grid=param_grid,cv=5)#Cross validation with score 5 
C_rfc.fit(X_train, Y_train)
C_rfc.best_params_
test=pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')
i_d=test["id"]
test=test.drop(["id"],axis=1)
X_train
clf=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion="gini", max_depth=12, max_features="auto",
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=150,
                       n_jobs=None, oob_score=True, random_state=None,
                       verbose=1, warm_start=False) #We will not tune anythinga nd will use the default values for this
clf.fit(X_train,Y_train)
y_pred=clf.predict(test)
len(y_pred)
final={"id":i_d,
      "Prediction":y_pred}
final=pd.DataFrame(final)
final
