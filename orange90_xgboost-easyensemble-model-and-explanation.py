import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
df_train.info()
df_train.describe()
#尝试过SMOTE去过采样，但没啥效果，就注释掉了
# from imblearn.over_sampling import SMOTE 
# sm = SMOTE(random_state=42)
# X_train, y_train = sm.fit_resample(X_train, y_train)
# print('bad rate is: ',y_train.mean())
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

model1 = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
model1.fit(X_train,y_train)
#在验证集上看性能
model1.score(X_test,y_test)
# calculate the fpr and tpr for all thresholds of the classification
plot_AUC(model1,X_test,y_test)
#混淆矩阵
y_pred = model1.predict(X_test)
metrics.confusion_matrix(y_test,y_pred)
weight = int(y_train.count()/y_train.sum())
model2 = xgb.XGBClassifier(objective="binary:logistic", random_state=42,scale_pos_weight = weight)
model2.fit(X_train,y_train)
model2.score(X_test,y_test)
plot_AUC(model2,X_test,y_test)
#混淆矩阵
y_pred = model2.predict(X_test)
metrics.confusion_matrix(y_test,y_pred)

from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train.fillna(0), y_train)
print('bad rate is: ',y_train_balanced.mean())
model3 = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
model3.fit(X_train_balanced,y_train_balanced)
model3.score(X_test,y_test)
plot_AUC(model3,X_test,y_test)
#混淆矩阵
y_pred = model3.predict(X_test)
metrics.confusion_matrix(y_test,y_pred)
param = {'min_child_weight': 10.0,
'objective': 'binary:logistic',
'max_depth': 5,
'eval_metric': 'auc',
'max_delta_step': 1.8,
'colsample_bytree': 0.4,
'subsample': 0.8,
'eta': 0.025,
'gamma': 0.65,
'num_boost_round' : 391
        }
from imblearn.ensemble import EasyEnsembleClassifier
X_train.head()
model4 = EasyEnsembleClassifier(n_estimators=20, random_state=42, base_estimator=xgb.XGBClassifier(objective="binary:logistic",random_state=42))
model4.fit(X_train.fillna(0),y_train)
model4.score(X_test.fillna(0),y_test)
plot_AUC(model4,X_test.fillna(0),y_test)

import shap
final_model = model4
for x in final_model.estimators_:
    print(x['classifier'].feature_importances_)
explainer = shap.TreeExplainer(final_model)

shap_values = explainer.shap_values(X_train)
print(shap_values.shape)

shap.summary_plot(shap_values,X_train)
shap.summary_plot(shap_values,X_train,plot_type='bar')
shap.dependence_plot('RevolvingUtilizationOfUnsecuredLines', shap_values,X_train, interaction_index=None, show=False)
sample = X_test.sample(1,random_state=42)
sample
final_model.predict_proba(sample)
shap.initjs()
shap_value_sample = explainer.shap_values(sample)
shap.force_plot(explainer.expected_value, shap_value_sample, sample)
df_test.head()

model_final = EasyEnsembleClassifier(n_estimators=50, random_state=42, base_estimator=xgb.XGBClassifier().set_params(**param))
model_final.fit(df_train.drop(['SeriousDlqin2yrs'],axis=1), df_train['SeriousDlqin2yrs'])

result = model_final.predict_proba(df_test.drop('SeriousDlqin2yrs',axis=1))
result = [x[1] for x in result]
df_result = pd.DataFrame({'Id':df_test['Unnamed: 0'].tolist(), 'Probability':result})
df_result.head()
df_result.to_csv('submission_credit_3.csv', index=False)
