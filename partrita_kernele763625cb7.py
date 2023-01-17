# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline 



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
data = pd.read_csv('../input/diabetes.csv')

data.head()
data.info()
x = data.iloc[:, :-1]

y = data.iloc[:, -1]



X_train, X_test, y_train, y_test = train_test_split(

    x, y, test_size=0.2, random_state=42, stratify=y)

def get_clf_eval(y_test, pred):

    confusion = confusion_matrix(y_test, pred)

    accuracy = accuracy_score(y_test, pred)

    precision = precision_score(y_test, pred)

    recall = recall_score(y_test, pred)

    f1 = f1_score(y_test, pred)

    print(

        '정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1점수: {3:.4f}'.format(

            accuracy, precision, recall, f1))
lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train)

pred_y = lr_clf.predict(X_test)

get_clf_eval(y_test, pred_y)
def precision_recall_curve_plot(y_test, pred_prova_c1):

    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    plt.figure(figsize=(8,6))

    threshold_boundary =thresholds.shape[0]

    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')

    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')

    start, end = plt.xlim()

    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    plt.xlabel('Threshold value')

    plt.legend()

    plt.grid()

    plt.show()



pred_proba_c1 = lr_clf.predict_proba(X_test)[:,1]

precision_recall_curve_plot(y_test, pred_proba_c1)
data.describe()
plt.hist(data['Glucose'], bins=10)
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI']

total_count = data['Glucose'].count()



for feature in zero_features:

    zero_count = data[data[feature] == 0][feature].count()

    print('{0} 0 건수는 {1}, 퍼센트는 {2:.2f}%'.format(feature, zero_count, 100*zero_count/total_count))

    
mean_zero_features = data[zero_features].mean()

data[zero_features] = data[zero_features].replace(0, mean_zero_features)
X = data.iloc[:, :-1]

y = data.iloc[:, -1]



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)



lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train)

pred = lr_clf.predict(X_test)

get_clf_eval(y_test, pred)
from sklearn.preprocessing import Binarizer





def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):

    for threshold in thresholds:

        binarizer = Binarizer(threshold=threshold).fit(pred_proba_c1)

        custom_predict = binarizer.transform(pred_proba_c1)

        print('임곗값: ', threshold)

        get_clf_eval(y_test, custom_predict)



thresholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.50]

pred_proba = lr_clf.predict_proba(X_test)

get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1, 1),thresholds)
binarizer = Binarizer(threshold=0.3)

pred_th_03 = binarizer.fit_transform(pred_proba[:,1].reshape(-1,1))

get_clf_eval(y_test, pred_th_03)