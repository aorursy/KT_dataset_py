import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import math
import pandas as pd 
import matplotlib.pyplot as plt

import seaborn as sns
import sklearn.metrics as metrics
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv",index_col=0)
df_test = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-test.csv",index_col=0)
df_train.head()
#标签1比例
df_train['SeriousDlqin2yrs'].mean()
#标签1总数
df_train['SeriousDlqin2yrs'].sum()
import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['SeriousDlqin2yrs'],axis=1), df_train['SeriousDlqin2yrs'], test_size=0.2, random_state=42)
def plot_AUC(model,X_test,y_test):
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

model1 = AdaBoostClassifier(random_state=42)
model1.fit(X_train.fillna(0),y_train)
# calculate the fpr and tpr for all thresholds of the classification
plot_AUC(model1,X_test.fillna(0),y_test)
disp = plot_confusion_matrix(model1, X_test.fillna(0), y_test,
                                 display_labels=[0,1],
                                 cmap=plt.cm.Blues,
                                 values_format='' 
                                 )
disp.ax_.set_title('confusion matrix')

from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train.fillna(0), y_train)
print('bad rate is: ',y_train_balanced.mean())
model2 =AdaBoostClassifier(random_state=42)
model2.fit(X_train_balanced,y_train_balanced)
plot_AUC(model2,X_test.fillna(0),y_test)
#混淆矩阵
y_pred = model2.predict(X_test.fillna(0))
metrics.confusion_matrix(y_test,y_pred)
from imblearn.ensemble import EasyEnsembleClassifier
model3 = EasyEnsembleClassifier(n_estimators=20, random_state=42, base_estimator=AdaBoostClassifier(random_state=42))
model3.fit(X_train.fillna(0),y_train)
plot_AUC(model3,X_test.fillna(0),y_test)
disp = plot_confusion_matrix(model3, X_test.fillna(0), y_test,
                                 display_labels=[0,1],
                                 values_format='',
                                 cmap=plt.cm.Blues
                                 )
disp.ax_.set_title('confusion matrix')
# 模拟极度不平衡情况，约0.35%的样本为1，其余为0
total_1 = int(df_train['SeriousDlqin2yrs'].sum()*0.05)
df_train_extreme = pd.concat([df_train[df_train['SeriousDlqin2yrs']==1].sample(total_1,random_state=42),
                              df_train[df_train['SeriousDlqin2yrs']==0]])
X_train_ex, X_test_ex, y_train_ex, y_test_ex = train_test_split(
                                                    df_train_extreme.drop(['SeriousDlqin2yrs'],axis=1), 
                                                    df_train_extreme['SeriousDlqin2yrs'], test_size=0.2, 
                                                    random_state=42)
#数据分布上接近
print(y_train_ex.mean(),y_test_ex.mean())

model_a = AdaBoostClassifier(random_state=42)
model_a.fit(X_train_ex.fillna(0),y_train_ex)
plot_AUC(model_a,X_test_ex.fillna(0),y_test_ex)
disp = plot_confusion_matrix(model_a, X_test_ex.fillna(0), y_test_ex,
                                 display_labels=[0,1],
                                 cmap=plt.cm.Blues
                                 )
disp.ax_.set_title('confusion matrix')
y_pred = model_a.predict(X_test_ex.fillna(0))
metrics.confusion_matrix(y_test_ex,y_pred)
model_b = EasyEnsembleClassifier(n_estimators=20, random_state=42, 
                                 base_estimator=AdaBoostClassifier(random_state=42))
model_b.fit(X_train_ex.fillna(0),y_train_ex)
plot_AUC(model_b,X_test_ex.fillna(0),y_test_ex)
disp = plot_confusion_matrix(model_b, X_test_ex.fillna(0), y_test_ex,
                                 display_labels=[0,1],
                                 cmap=plt.cm.Blues
                                 )
disp.ax_.set_title('confusion matrix')
y_pred = model_b.predict(X_test_ex.fillna(0))
metrics.confusion_matrix(y_test_ex,y_pred)