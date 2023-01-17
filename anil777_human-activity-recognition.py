import numpy as np

import pandas as pd 



train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

train.head()
train.shape
import sklearn 

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

features = train.iloc[:,0:562]

label = train["Activity"]

clf = ExtraTreesClassifier()

clf = clf.fit(features,label)

model = SelectFromModel(clf,prefit=True)

New_features = model.transform(features)

print(New_features.shape)
from sklearn.svm import LinearSVC

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, label)

model_2 = SelectFromModel(lsvc, prefit=True)

New_features_2 = model_2.transform(features)

print(New_features_2.shape)

# Loading Models 



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

Classifiers = [DecisionTreeClassifier(),RandomForestClassifier(n_estimators=200),GradientBoostingClassifier(n_estimators=200)]
#Without Feature Selection 



from sklearn.metrics import accuracy_score

import timeit

test_features= test.iloc[:,0:562]

Time_1=[]

Model_1=[]

Out_Accuracy_1=[]



for clf in Classifiers:

    start_time = timeit.default_timer()

    fit=clf.fit(features,label)

    pred=fit.predict(test_features)

    elapsed = timeit.default_timer() - start_time

    Time_1.append(elapsed)

    Model_1.append(clf.__class__.__name__)

    Out_Accuracy_1.append(accuracy_score(test['Activity'],pred))

    

    
test_features= model.transform(test.iloc[:,0:562])



Time_2=[]

Model_2=[]

Out_Accuracy_2=[]



for clf in Classifiers:

    start_time = timeit.default_timer()

    fit=clf.fit(New_features,label)

    pred=fit.predict(test_features)

    elapsed = timeit.default_timer() - start_time

    Time_2.append(elapsed)

    Model_2.append(clf.__class__.__name__)

    Out_Accuracy_2.append(accuracy_score(test['Activity'],pred))

test_features= model_2.transform(test.iloc[:,0:562])



Time_3=[]

Model_3=[]

Out_Accuracy_3=[]



for clf in Classifiers:

    start_time = timeit.default_timer()

    fit=clf.fit(New_features_2,label)

    pred=fit.predict(test_features)

    elapsed = timeit.default_timer() - start_time

    Time_3.append(elapsed)

    Model_3.append(clf.__class__.__name__)

    Out_Accuracy_3.append(accuracy_score(test['Activity'],pred))
import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline



ind =  np.arange(3)   # the x locations for the groups

width = 0.1       # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(ind, Out_Accuracy_1, width, color='r')

rects2 = ax.bar(ind + width, Out_Accuracy_2, width, color='y')

rects3 = ax.bar(ind + width + width ,Out_Accuracy_3, width, color='b')

ax.set_ylabel('Accuracy')

ax.set_title('Accuracy by Models and Selection Process')

ax.set_xticks(ind + width)

ax.set_xticklabels(Model_3,rotation=45)

plt.show()
