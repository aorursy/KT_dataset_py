import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
test = pd.read_csv('/kaggle/input/hivprogression/test_data.csv')
train = pd.read_csv('/kaggle/input/hivprogression/training_data.csv')
train[:3]
train.Resp.value_counts().plot.bar(rot=0);
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
import xgboost as xgb
x_cols = ['VL-t0', 'CD4-t0']
train_,test_ = train_test_split(train[x_cols + ['Resp']],test_size=0.33,random_state=42,stratify=train.Resp)
'train:',train_.Resp.value_counts() / len(train_),'test:',test_.Resp.value_counts() / len(test_)
params_grid = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5],
    'n_estimators': [100,300,600,1000]
}
%%time
xgc = xgb.XGBClassifier()
grid = GridSearchCV(xgc,params_grid,cv=3,verbose=1000,n_jobs=5)
grid.fit(train_[x_cols],train_.Resp)
results = test_.copy()
results['y_pred'] = grid.best_estimator_.predict(test_[x_cols])
print(metrics.classification_report(results.Resp,results.y_pred))
sns.heatmap(metrics.confusion_matrix(results.Resp,results.y_pred),annot=True,fmt='d');
xgb.plot_importance(grid.best_estimator_);
%%time
xgc = xgb.XGBClassifier()
grid = GridSearchCV(xgc,params_grid,cv=3,verbose=1000,n_jobs=5)
grid.fit(train[x_cols],train.Resp)
results = test.copy()
results['ResponderStatus'] = grid.best_estimator_.predict(test[x_cols])
results[['PatientID','ResponderStatus']].to_csv('submission.csv',index=False)