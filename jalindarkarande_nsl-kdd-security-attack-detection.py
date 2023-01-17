import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib as matplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_json('file://localhost/kaggle/input/nslkdd-in-json-format/NSL-KDD-Train.json',lines=True)
train.head()
le_prototype = LabelEncoder().fit(train['protocol_type'])
train['protocol_type'] =le_prototype.transform(train['protocol_type'])
lb_prototype = LabelBinarizer().fit(train['protocol_type'])
train['protocol_type'] = lb_prototype.transform(train['protocol_type'])
le_service = LabelEncoder().fit(train['service'])
train['service'] =le_service.transform(train['service'])
lb_service = LabelBinarizer().fit(train['service'])
train['service'] = lb_service.transform(train['service'])
le_flag = LabelEncoder().fit(train['flag'])
train['flag'] =le_flag.transform(train['flag'])
le_class = LabelEncoder().fit(train['class'])
train['class'] =le_class.transform(train['class'])
train.head()
x = train.drop('class', axis=1)
y = train.loc[:,['class']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=908)
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
x = X_train
y = y_train['class'].ravel()

clf1 = DecisionTreeClassifier() 
clf2 = RandomForestClassifier(n_estimators=25, random_state=1)
clf3 = GradientBoostingClassifier()
ET = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False)

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('et',ET)], voting='hard') 

for clf, label in zip([clf1, clf2, clf3,ET, eclf], ['DecisionTreeClassifier', 'Random Forest', 'GradientBoostingClassifier','ExtraTreesClassifier', 'Ensemble']): 
    tmp = clf.fit(x,y)
    pred = clf.score(X_test,y_test)
    print("Acc: %0.2f [%s]" % (pred,label))
### Lets test on Testing dataset
test = pd.read_json('file://localhost/kaggle/input/nslkdd-in-json-format/NSL-KDD-Test.json',lines=True)
test['class'] =le_class.transform(test['class'])
test['flag'] =le_flag.transform(test['flag'])
test['service'] =le_service.transform(test['service'])
test['protocol_type'] =le_prototype.transform(test['protocol_type'])
x = test.drop('class', axis=1)
y = test.loc[:,['class']]
for clf, label in zip([clf1, clf2, clf3,ET, eclf], ['DecisionTreeClassifier', 'Random Forest', 'GradientBoostingClassifier','ExtraTreesClassifier', 'Ensemble']): 
    pred = clf.score(x,y)
    print("Acc: %0.2f [%s]" % (pred,label))