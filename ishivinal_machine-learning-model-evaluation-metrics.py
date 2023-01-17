 !pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887 2>/dev/null 1>/dev/null
import warnings

warnings.filterwarnings("ignore")
from fastai.structured import train_cats,proc_df
import numpy as np 

import pandas as pd 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split
!ls ../input/house-prices-advanced-regression-techniques/train.csv
train = pd.read_csv("../input/titanic/train.csv")

train2 = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train_cats(train)

train_cats(train2)
train.columns
train2.columns
train.Age=train.Age.fillna(train.Age.median())
df_c,y_c,_=proc_df(train,"Survived")
df_r,y_r,_=proc_df(train2,"SalePrice")
X_train, X_test, y_train, y_test = train_test_split(df_c, y_c, test_size=0.3)
m = RandomForestClassifier(n_jobs=-1)

m.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
confusion_matrix(y_test, m.predict(X_test))
m.score(X_test,y_test)
accuracy_score(y_test,m.predict(X_test))
from sklearn.metrics import classification_report
print(classification_report(y_test, m.predict(X_test)))
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

%matplotlib inline
fpr, tpr, thresholds = roc_curve(y_test, m.predict_proba(X_test)[:,1])



plt.plot(fpr, tpr, label='ROC curve')

plt.plot([0, 1], [0, 1], 'k--', label='Random guess')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.xlim([-0.02, 1])

plt.ylim([0, 1.02])

plt.legend(loc="lower right")
from sklearn.metrics import log_loss
log_loss(y_test,m.predict_proba(X_test))
y_r=np.log(y_r)
X_train, X_test, y_train, y_test = train_test_split(df_r, y_r, test_size=0.3)
m = RandomForestRegressor(n_jobs=-1)

m.fit(X_train, y_train)
y_pred=m.predict(X_test)
from sklearn.metrics import mean_squared_error
round(mean_squared_error(y_test,y_pred),3)
from math import sqrt
round(sqrt(mean_squared_error(y_test,y_pred)),3)
from sklearn.metrics import mean_absolute_error
round(mean_absolute_error(y_test,y_pred),3)
round(sqrt(mean_absolute_error(y_test,y_pred)),3)
from sklearn.metrics import mean_squared_log_error
round(mean_squared_log_error( y_test, y_pred ),4)
round(np.sqrt(mean_squared_log_error( y_test, y_pred )),4)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
r2
n =len(X_train)
r2_adj =1- (1-r2)*(n-1)/(n-(13+1))
r2_adj
from sklearn.model_selection import KFold,StratifiedKFold
n_fold = 10

folds = KFold(n_splits=n_fold, shuffle=True)
test_pred_proba = np.zeros((df_c.shape[0], 2))

accuracy = []

    

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_c, y_c)):

        X_train, X_valid = df_c.iloc[train_idx], df_c.iloc[valid_idx]

        y_train, y_valid = y_c[train_idx], y_c[valid_idx]

        

        model = RandomForestClassifier()

        model.fit(X_train, y_train)



        y_pred_valid = model.predict(X_valid)

        accuracy.append(accuracy_score(y_valid,y_pred_valid))



accuracy
n_fold = 10

folds = StratifiedKFold(n_splits=n_fold, shuffle=True)
accuracy = []

    

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_c, y_c)):

        X_train, X_valid = df_c.iloc[train_idx], df_c.iloc[valid_idx]

        y_train, y_valid = y_c[train_idx], y_c[valid_idx]

        

        model = RandomForestClassifier()

        model.fit(X_train, y_train)



        y_pred_valid = model.predict(X_valid)

        accuracy.append(accuracy_score(y_valid,y_pred_valid))
accuracy
from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection import cross_val_score
loocv = LeaveOneOut()

m = RandomForestClassifier(n_jobs=-1)

results = cross_val_score(m, df_c, y_c, cv=loocv)
results
results.mean()
from sklearn.model_selection import ShuffleSplit
kfold = ShuffleSplit(n_splits=10, test_size=0.3)

results = cross_val_score(m, df_c, y_c, cv=kfold)
results
results.mean()