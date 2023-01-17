# os
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data viz
from matplotlib import pyplot as plt
plt.style.use('seaborn')
import seaborn as sns


# metrics
from sklearn import metrics

# preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

# modelling
import xgboost as xgb
train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
train[:3]
x_cols = ['Gender', 'Age', 'Driving_License', 'Region_Code',
       'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
       'Policy_Sales_Channel', 'Vintage']

categoricals = ['Vehicle_Age','Vehicle_Damage','Gender']
train.hist(figsize=(25,10));
bins = np.arange(15,105,15)
labels = [f'{bins[x-1]}-{bins[x]}' for x in range(1,len(bins))]
pd.crosstab(train.Gender,pd.cut(train.Age,bins=bins,labels=labels)).plot.bar(rot=0);
bins = np.arange(15,105,15)
labels = [f'{bins[x-1]}-{bins[x]}' for x in range(1,len(bins))]
pd.crosstab(train.Vehicle_Age,pd.cut(train.Age,bins=bins,labels=labels)).plot.bar(rot=0);
train_,test_ = train_test_split(train, test_size=0.33,random_state=42, stratify=train.Response)
train_ = train_.copy()
test_ = test_.copy()

encods = dict()
for x in categoricals:
    unq = train_[x].unique()
    encods[x] = dict(zip(unq,list(range(len(unq)+1))))
    encods[f'rev_{x}'] = {v:k for k,v in encods[x].items()}
    train_[x] = train_[x].replace(encods[x])
    test_[x] = test_[x].replace(encods[x])
'train',train_.Response.value_counts() / len(train_), 'test', test_.Response.value_counts() / len(test_)
from collections import Counter
counter = Counter(train_.Response)
# estimate scale_pos_weight value
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)
xgc = xgb.XGBClassifier(n_estimators=1000,scale_pos_weight=estimate)
xgc.fit(train_[x_cols],train_.Response)
test_['pred'] = xgc.predict(test_[x_cols])
print(metrics.classification_report(test_.Response,test_.pred))
sns.heatmap(metrics.confusion_matrix(test_.Response,test_.pred),annot=True,fmt='d');
import shap
%%time
ex = shap.TreeExplainer(xgc)
shap_values = ex.shap_values(test_[x_cols],test_.Response)
shap.summary_plot(shap_values,test_[x_cols])
shap.summary_plot(shap_values,test_[x_cols],plot_type='bar')