import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import Binarizer



import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

print(train['Outcome'].value_counts())

train.head()
train.info()
train.isnull().sum()
features = train.columns.difference(['Outcome'])

label = 'Outcome'



X = train[features]

y = train[label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,

                                                   random_state=156, stratify = y)
lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train)

pred = lr_clf.predict(X_test)
pred
lr_clf.predict_proba(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

pred_proba
def get_clf_eval(y_test, pred = None, pred_proba = None):

    confusion = confusion_matrix(y_test, pred)

    accuracy = accuracy_score(y_test, pred)

    precision = precision_score(y_test, pred)

    recall = recall_score(y_test, pred)

    f1 = f1_score(y_test, pred)

    roc_auc = roc_auc_score(y_test, pred_proba)



    print('오차행렬')

    print(confusion)

    print('정확도:{0: .4f}, 정밀도:{1: .4f}, 재현율:{2: .4f}),F1:{3: .4f},AUC:{4: .4f}'.format(accuracy, precision, recall, f1, roc_auc))
get_clf_eval(y_test, pred, pred_proba)
def precision_recall_curve_plot(y_test, pred_proba_c1):

    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    

    plt.figure(figsize=(8,6))

    thresholds_boundary = thresholds.shape[0]

    plt.plot(thresholds, precisions[0:thresholds_boundary], linestyle='--', label = 'precision')

    plt.plot(thresholds, recalls[0:thresholds_boundary],label = 'recall')

    

    start, end = plt.xlim()

    plt.xticks(np.round(np.arange(start, end, 0.1),2))

    

    plt.xlabel('Threshold Value'); plt.ylabel('Precision and Recall Value')

    plt.legend(); plt.grid()

    plt.show()
precision_recall_curve_plot(y_test, pred_proba)
plt.hist(train['Glucose'], bins=10)
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']



total_count = train['Glucose'].count()



for features in zero_features:

    zero_count = train[train[features] == 0][features].count()

    print('{0} 0건수는 {1}, 퍼센트는 {2:.2f} %'.format(features, zero_count, 100*zero_count/total_count))
mean_zero_features = train[zero_features].mean()
mean_zero_features
train[zero_features] = train[zero_features].replace(0, mean_zero_features)
for features in zero_features:

    zero_count = train[train[features] == 0][features].count()

    print('{0} 0 건수는 {1}, 퍼센트는 {2:.2f} %'.format(features, zero_count, 100*zero_count/total_count))
features = train.columns.difference(['Outcome'])

label = 'Outcome'



X = train[features]

y = train[label]
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
pd.DataFrame(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 156, stratify = y)
lr_reg = LogisticRegression()

lr_reg.fit(X_train, y_train)

pred = lr_reg.predict(X_test)

pred_proba = lr_reg.predict_proba(X_test)



get_clf_eval(y_test, pred, pred_proba[:, 1])
from sklearn.preprocessing import Binarizer



def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):

    #thresholds 리스트 객체내의 값을 차례로 iteration하면서 Evaluation 수행

    for custom_threshold in thresholds:

        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)

        custom_predict = binarizer.transform(pred_proba_c1)

        print('임계값: ',custom_threshold)

        get_clf_eval(y_test, custom_predict, pred_proba_c1)
thresholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.5]

pred_proba = lr_reg.predict_proba(X_test)

get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1,1), thresholds)
binarizer = Binarizer(threshold=0.45)
pred_th_045 = binarizer.fit_transform(pred_proba[:, 1].reshape(-1, 1))



get_clf_eval(y_test, pred_th_045, pred_proba[:, 1])