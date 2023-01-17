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
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,auc,roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from imblearn import under_sampling 
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks

from imblearn.combine import SMOTEENN

from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.pipeline import Pipeline
from matplotlib import pyplot



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier

df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv',index_col=0,header=0)
df.head()
df.target.value_counts()
corr = df.corr()
f = plt.figure(figsize=(19, 15))
plt.matshow(corr, fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
gs = gridspec.GridSpec(89, 1)
plt.figure(figsize=(16,28*4))
for i, col in enumerate(df[df.iloc[:,:].columns]):
    ax5 = plt.subplot(gs[i])
    sns.distplot(df[col][df.target == 1], bins=50, color='r')
    sns.distplot(df[col][df.target == 0], bins=50, color='g')
    ax5.set_xlabel('')
    ax5.set_title('feature: ' + str(col))
plt.show()
cols = [0,22,24,30,32,33,48,51,52,53,55,58,60,61,62,63,64,65,66,67,69,71,74,79,82,86,87,68,29,28,27,26,25,20,19,16,12,10,9,8,6]
print(len(cols))
drop_list = []
for col in cols:
  drop_list.append('col_'+str(col))

def split_data(df, drop_list,split):
    df = df.drop(drop_list,axis=1)
    # print(df.columns)
    #test train split time
    from sklearn.model_selection import train_test_split
    y = df['target'].values #target
    X = df.drop(['target'],axis=1).values #features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split,
                                                    random_state=42, stratify=y)

    
    # scaler = preprocessing.StandardScaler()
    # x_train_scaled = scaler.fit_transform(X_train)
    # X_train = pd.DataFrame(x_train_scaled)

    # x_test_scaled = scaler.fit_transform(X_test)
    # X_test = pd.DataFrame(x_test_scaled)
    # print("train-set size: ", len(y_train),
    #   "\ntest-set size: ", len(y_test))
    # print("fraud cases in test-set: ", sum(y_test))
    return X_train, X_test, y_train, y_test
def print_scores(y_test,y_pred):
    # print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
    # print("recall score: ", recall_score(y_test,y_pred))
    # print("precision score: ", precision_score(y_test,y_pred))
    # print("f1 score: ", f1_score(y_test,y_pred))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    print("accuracy score: ", accuracy_score(y_test,y_pred))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred)))
def get_predictions(clf, X_train, y_train, X_test):
    # create classifier
    clf = clf
    # fit it to training data
    clf.fit(X_train,y_train)
    # predict using test data
    y_pred = clf.predict(X_test)
    # Compute predicted probabilities: y_pred_prob
    # y_pred_prob = clf.predict_proba(X_test)
    #for fun: train-set predictions
    # train_pred = clf.predict(X_train)
    # print('train-set confusion matrix:\n', confusion_matrix(y_train,train_pred)) 
    return y_pred
drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list,0.05)
print('Undersampling')
under = RandomUnderSampler(sampling_strategy={0:36000},random_state=0)
# y_train = df.target
# X_train = df.drop(columns='target')
x,y = under.fit_resample(X_train,y_train)

model = LogisticRegression(max_iter=10000,class_weight='balanced',random_state=0,solver='saga',n_jobs=-1,C=1000)
model.fit(x,y)
roc_auc_score(y_test,model.predict(X_test))
y_pred_prob = model.predict_proba(X_test)
roc_auc_score(y_test,y_pred_prob[:,1])
drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list,0.05)
print('Undersampling')
under = RandomUnderSampler(sampling_strategy={0:7500},random_state=0)
# y_train = df.target
# X_train = df.drop(columns='target')
x,y = under.fit_resample(X_train,y_train)


# rf = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state = 0,
#                             # class_weight='balanced'
#                             # oob_score = True
#                             ) 
rf = RandomForestClassifier(n_estimators = 1000, criterion = "gini", max_depth = 5,
                                  max_features = "auto", min_samples_leaf = 0.005,
                                  min_samples_split = 0.005, n_jobs = -1, random_state = 0) 
rf.fit(x,y)
y_pred_prob = model.predict_proba(X_test)
roc_auc_score(y_test,y_pred_prob[:,1])
drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list,0.2)
print('Undersampling')
under = RandomUnderSampler(sampling_strategy={0:7200},random_state=0)
seed = 7
x,y = under.fit_resample(X_train,y_train)

# fit model no training data
model = XGBClassifier(n_jobs=-1,n_estimators=1000,random_state=0,booster='gbtree',scale_pos_weight=6)
model.fit(np.array(x), y,eval_metric='auc')
# make predictions for test data
y_pred = model.predict(np.array(X_test))
predictions = y_pred
# predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
y_pred_prob = model.predict_proba(X_test)
roc_auc_score(y_test,y_pred_prob[:,1])
Counter(y)
drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list,0.2)
print('Undersampling')
under = RandomUnderSampler(sampling_strategy={0:1500},random_state=0)
# y_train = df.target
# X_train = df.drop(columns='target')
x,y = under.fit_resample(X_train,y_train)
print(Counter(y))
level0 = [('lr',LogisticRegression(max_iter=10000,random_state=0,class_weight='balanced',n_jobs=-1,solver='saga',C=0.1)),
          ('dt',DecisionTreeClassifier(random_state=0,class_weight='balanced')),
          # ('bc',BaggingClassifier(n_estimators=1000,random_state=0)),
          ('knn',KNeighborsClassifier()),
          ('rf',RandomForestClassifier(n_estimators=400,random_state=0,class_weight='balanced',n_jobs=-1,criterion='entropy')),
          ('abc',AdaBoostClassifier(n_estimators=1000,random_state=0,learning_rate=0.01)),
          ('svm',SVC(random_state=0,class_weight='balanced')),
          ('xgb',XGBClassifier(n_estimators=200,class_weight='balanced',n_jobs=-1,scale_pos_weight=1.25,booster='gblinear'))]
# level1 = XGBClassifier(n_estimators=500,random_state=0)
level1 = LogisticRegression(max_iter=10000,class_weight='balanced',random_state=0,solver='saga',n_jobs=-1)
# level1 = RandomForestClassifier(n_estimators=600,random_state=0,class_weight='balanced',n_jobs=-1,max_depth=8,criterion='entropy')
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10,n_jobs=-1)
model.fit(x,y)
y_pred_prob = model.predict_proba(X_test)
roc_auc_score(y_test,y_pred_prob[:,1])
drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list)
print('Undersampling')
under = RandomUnderSampler(sampling_strategy={0:100000},random_state=0)
x,y = under.fit_resample(X_train,y_train)
for est in [0.01,0.1,1,10,1000]:
  # for dep in [0.01,0.1,1]:
    print("Estimators: ",est, " Depth: ",dep)
    # print()
    model = LogisticRegression(C=est,class_weight='balanced',n_jobs=-1,max_iter=10000,random_state=0)
    model.fit(x,y)
    y_pred = model.predict(X_test)
    print('ROC on Absolute')
    print(roc_auc_score(y_test,y_pred))
    y_pred_prob = model.predict_proba(X_test)
    print('ROC on Probabilities')
    print(roc_auc_score(y_test,y_pred_prob[:,1]))
    print()
    print()
drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list)
# X_train = df.drop(columns='target')
# y_train = df.target
print('Undersampling')
under = RandomUnderSampler(sampling_strategy={0:60000},random_state=0)
x,y = under.fit_resample(X_train,y_train)
print(Counter(y))
for est in [200,400,600,800]:
  for dep in [5,6,8,10]:
    print("Estimators: ",est, " Depth: ",dep)
    # print()
    model = RandomForestClassifier(n_estimators = est, criterion = "entropy", max_depth = dep,
                                  max_features = "auto", min_samples_leaf = 0.005,class_weight='balanced',
                                  min_samples_split = 0.005, n_jobs = -1, random_state = 0) 
    model.fit(x,y)
    y_pred = model.predict(X_test)
    print('ROC on Absolute')
    print(roc_auc_score(y_test,y_pred))
    y_pred_prob = model.predict_proba(X_test)
    print('ROC on Probabilities')
    print(roc_auc_score(y_test,y_pred_prob[:,1]))
    print()
    print()


level0 = [
          ('lr',LogisticRegression(C=1,max_iter=10000,random_state=0,class_weight='balanced',n_jobs=-1,solver='saga')),
          ('dt',DecisionTreeClassifier(random_state=0,class_weight='balanced')),
          ('knn',KNeighborsClassifier()),
          ('rf',RandomForestClassifier(n_estimators=500,random_state=0,class_weight='balanced',n_jobs=-1,criterion='entropy')),
          ('abc',AdaBoostClassifier(n_estimators=1000,random_state=0,learning_rate=0.1)),
          ('svm',SVC(random_state=0,class_weight='balanced')),
          ('xgb',XGBClassifier(n_estimators=1000,n_jobs=-1,booster='gblinear',class_weight='balanced',learning_rate=0.1,random_state=0,max_depth=50,scale_pos_weight=1,subsample=1))]
# level1 = XGBClassifier(n_estimators=500,n_jobs=-1,booster='gblinear',class_weight='balanced',learning_rate=0.01,random_state=0,max_depth=10,scale_pos_weight=1.25,subsample=1)
# level1 = LogisticRegression(C=1,max_iter=10000,class_weight='balanced',random_state=0,solver='saga')
level1 = RandomForestClassifier(n_estimators=600,random_state=0,class_weight='balanced',n_jobs=-1,max_depth=8,criterion='entropy')
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10,n_jobs=-1)
model.fit(x,y)
df_test = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
df_test.head()
drop_list = []
X_Test = df_test.iloc[:,1:].drop(columns=drop_list)
preds = model.predict_proba(np.array(X_Test))
df_test['target'] = preds[:,1]
df_test.loc[:,['id','target']].to_csv('/kaggle/output/Results.csv',index=False)