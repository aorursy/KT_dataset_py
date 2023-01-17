# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

train_data.head()
test_data=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

test_data.head()
test_x=test_data.copy().drop(columns=['ID_code'])

test_x.head()
train_y=train_data['target']

train_x=train_data.copy().drop(columns=['target','ID_code'])

train_x.head()
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

clf1 = tree.DecisionTreeClassifier()

clf1_={"clf":clf1,"name":"Decision tree"}

clf2 = RandomForestClassifier(n_estimators=10)

clf2_={"clf":clf2,"name":"Random forest"}

clf3= AdaBoostClassifier(n_estimators=100)

clf3_={"clf":clf3,"name":"Ada boost"}

#from sklearn import svm

#clf2= svm.SVC(kernel='rbf')

#clf2_={"clf":clf2,"name":"Support vector machine"}

clfs=[clf1_,clf2_,clf3_]
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report,roc_auc_score



skf = StratifiedKFold(n_splits=2,random_state=5)

skf.get_n_splits(train_x,train_y)

best_auc_avg=0

for clf_ in clfs:

    print(clf_["name"])

    clf=clf_["clf"]

    aucList=[]

    for train_index, test_index in skf.split(train_x, train_y):

        X_train_split, X_test_split = train_x.iloc[train_index], train_x.iloc[test_index]

        y_train_split, y_test_split = train_y[train_index], train_y[test_index]

        positive=np.sum(y_train_split==1)

        print("Split size {} train samples, {} positive".format(X_train_split.shape, positive))

        clf.fit(X_train_split,y_train_split)

        y_pred_split=clf.predict(X_test_split)

        print(classification_report(y_test_split,y_pred_split))

        auc_score=roc_auc_score(y_test_split,y_pred_split)

        print("Roc AUC:{}".format(auc_score))

        aucList.append(auc_score)

    auc=np.asarray(aucList)

    print(auc.shape)

    print(auc)

    auc_avg=np.mean(auc)

    print("ROC AUC AVG:{}".format(auc_avg))

    if(auc_avg>best_auc_avg):

        best_clf_=clf_

        best_auc_avg=auc_avg

        

print("Best classifier was {}".format(best_clf_["name"]))

    
print(best_clf_["name"])

clf=best_clf_["clf"]
import lightgbm as lgb

params = {}

params['learning_rate'] = 0.003

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['metric'] = 'auc'

params['sub_feature'] = 0.5

params['num_leaves'] = 10

params['min_data'] = 50

params['max_depth'] = 10

params['verbosity'] = 1
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report,roc_auc_score



skf = StratifiedKFold(n_splits=2,random_state=5)

skf.get_n_splits(train_x,train_y)

clf_={"clf":None,"name":"LightGbm"}

print(clf_["name"])

aucList=[]

predictions = np.zeros(len(test_x))

train_predictions = np.zeros(len(train_x))

for train_index, test_index in skf.split(train_x, train_y):

    X_train_split, X_test_split = train_x.iloc[train_index], train_x.iloc[test_index]

    y_train_split, y_test_split = train_y[train_index], train_y[test_index]

    d_train = lgb.Dataset(X_train_split, label=y_train_split)

    d_test=lgb.Dataset(X_test_split, label=y_test_split)

    positive=np.sum(y_train_split==1)

    print("Split size {} train samples, {} positive".format(X_train_split.shape, positive))

    clf = lgb.train(params, d_train, 5000,verbose_eval=500,early_stopping_rounds=100,valid_sets=[d_train,d_test])

    y_pred_split=clf.predict(X_test_split,num_iteration=clf.best_iteration)

    #print(classification_report(y_test_split,y_pred_split))

    auc_score=roc_auc_score(y_test_split,y_pred_split)

    print("Roc AUC:{}".format(auc_score))

    aucList.append(auc_score)

    # Combine predictions from all folds and average

    predictions += clf.predict(test_x, num_iteration=clf.best_iteration) / skf.n_splits

    train_predictions +=clf.predict(train_x, num_iteration=clf.best_iteration) / skf.n_splits

auc=np.asarray(aucList)

print(auc.shape)

print(auc)

auc_avg=np.mean(auc)

print("ROC AUC AVG:{}".format(auc_avg))

auc_combined=roc_auc_score(train_y,train_predictions)

print("ROC AUC Combined:{}".format(auc_combined))



if(auc_combined>best_auc_avg):

    best_clf_=clf_

    best_auc_avg=auc_combined

        

print("Best classifier was {}".format(best_clf_["name"]))
test_y=predictions
sub_df=pd.DataFrame({"ID_code":test_data["ID_code"].values})

sub_df["target"]=test_y

sub_df.to_csv("submission.csv",index=False)