%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from mpl_toolkits.mplot3d import Axes3D,proj3d

import numpy as np

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="darkgrid")
# Load Data set

data = pd.read_csv("../input/creditcard.csv")

data.Time = data.Time.apply(lambda x: np.ceil(float(x)/3600))



#split data set

train,test = train_test_split(data,test_size=.33)



#Grab fraud data

fraud = train.loc[train["Class"] == 1]

norm = train.loc[train["Class"] == 0]
f,ax = plt.subplots(1,1)

_ = sns.distplot(data.Time.loc[data.Class == 1],label="fraud",ax=ax)

_ = sns.distplot(data.Time.loc[data.Class == 0],label="Norm",ax=ax)

ax.legend()
fig,axs = plt.subplots(2,2,figsize=(15,15))

_ = sns.scatterplot(x=data.V15,y=data.V16,hue=data.Class,ax=axs[0,0]) # V15,V16 completely seperates the data  SVM or LDA

_ = sns.scatterplot(x=data.V15,y=data.V17,hue=data.Class,ax=axs[0,1])

_ = sns.scatterplot(x=data.V15,y=data.V14,hue=data.Class,ax=axs[1,0])

_ = sns.scatterplot(x=data.V15,y=data.V18,hue=data.Class,ax=axs[1,1])
# Lets perform a cross_valuidation 

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier



rf = RandomForestClassifier(n_estimators=10,class_weight={0:.9})

bg = BaggingClassifier()





lb = LabelBinarizer()

y_train = data['Class']

x = data[data.keys()[0:-1]]



rf_cv = cross_val_score(rf,X=x,y=y_train,cv=5,scoring='precision')

bg_cv = cross_val_score(bg,X=x,y=y_train,cv=5,scoring='precision')



print("rf pre: {}, Average: {}".format(rf_cv,np.average(rf_cv)))

print("bg pre: {}, Average: {}".format(bg_cv,np.average(bg_cv)))
#Bagging Classifier with Stratified K-Fold CV and upsampling

from sklearn.model_selection import StratifiedKFold

from sklearn.utils import resample

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

from sklearn.metrics import average_precision_score

from sklearn.ensemble import BaggingClassifier



cv = StratifiedKFold(n_splits = 10,shuffle=True)

bg = BaggingClassifier(n_jobs=4)

bg_accs = []

bg_recall = []

bg_pre = []

for train_idx,test_idx in cv.split(X=train.iloc[:,:-1],y=train.Class):

    ktrain = train.iloc[train_idx]

    ktest = train.iloc[test_idx]

    

    #Upsample fraud cases

    kfraud = ktrain.loc[ktrain["Class"] == 1]

    knorm = ktrain.loc[ktrain["Class"]==0]

    

    krefraud = resample(kfraud,n_samples=int((2/3)*len(knorm)))

    upktrain = pd.concat([krefraud,knorm]).sample(frac=1)

    

    bg.fit(X=upktrain[upktrain.keys()[0:-1]],y=upktrain.Class)

    y_pred = bg.predict(ktest[ktest.keys()[0:-1]])

    bg_accs.append(accuracy_score(ktest.Class,y_pred))

    bg_recall.append(recall_score(ktest.Class,y_pred))

    bg_pre.append(average_precision_score(ktest.Class,y_pred))
##### Precision-recall curve

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import classification_report

v_y = bg.predict_proba(test[test.keys()[0:-1]])

pre,rec,thr = precision_recall_curve(test.Class,v_y[:,1],pos_label=1)

ax = sns.lineplot(x=rec,y=pre,label=thr)

ax.set(xlabel="Recall",ylabel="Precision")

print(classification_report(test.Class,[1 if i > .45 else 0 for i in v_y[:,1]]))
#Random forest with Stratified KFold CV and upsampling

from sklearn.model_selection import StratifiedKFold

from sklearn.utils import resample

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

from sklearn.metrics import average_precision_score

from sklearn.ensemble import RandomForestClassifier



cv = StratifiedKFold(n_splits = 10,shuffle=True)

rf = RandomForestClassifier(n_estimators=10,class_weight={0:.9})

rf_accs = []

rf_recall = []

rf_pre = []

for train_idx,test_idx in cv.split(X=train.iloc[:,:-1],y=train.Class):

    ktrain = train.iloc[train_idx]

    ktest = train.iloc[test_idx]

    

    #Upsample fraud cases

    kfraud = ktrain.loc[ktrain["Class"] == 1]

    knorm = ktrain.loc[ktrain["Class"]==0]

    

    krefraud = resample(kfraud,n_samples=int((2/3)*len(knorm)))

    upktrain = pd.concat([krefraud,knorm]).sample(frac=1)

    

    rf.fit(X=upktrain[upktrain.keys()[0:-1]],y=upktrain.Class)

    y_pred = rf.predict(ktest[ktest.keys()[0:-1]])

    rf_accs.append(accuracy_score(ktest.Class,y_pred))

    rf_recall.append(recall_score(ktest.Class,y_pred))

    rf_pre.append(average_precision_score(ktest.Class,y_pred))
##### Precision-recall curve

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import classification_report

v_y = rf.predict_proba(test[test.keys()[0:-1]])

pre,rec,thr = precision_recall_curve(test.Class,v_y[:,1],pos_label=1)

ax = sns.lineplot(rec,pre)

ax.set(xlabel="Recall",ylabel="Precision")

print(classification_report(test.Class,[1 if i > .29 else 0 for i in v_y[:,1]]))