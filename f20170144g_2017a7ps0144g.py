!ls /kaggle/input/
import pandas as pd

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,auc,roc_auc_score

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import numpy as np

from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler
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
from collections import Counter
df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv',index_col=0,header=0)
Y = df['target']

X = df.drop(['target'], axis=1)
x_train,x_test, y_train, y_test = train_test_split(X,Y,test_size=0.1,random_state=0,shuffle=True)
under = RandomUnderSampler(sampling_strategy={0:10000},random_state=0)

x_train,y_train = under.fit_resample(x_train,y_train)
from sklearn.preprocessing import normalize

x_train = normalize(x_train)

x_test = normalize(x_test)
level0 = [('logreg',LogisticRegression(max_iter=10000,random_state=0,class_weight='balanced',solver='saga', n_jobs=-1)),

          ('ranfor',RandomForestClassifier(n_estimators=1000,random_state=0,criterion='entropy',class_weight='balanced',n_jobs=-1)),

          ('adaclf',AdaBoostClassifier(n_estimators=1000,random_state=0,learning_rate=0.1)),

          ('xgbclf',XGBClassifier(n_estimators=1000, booster='gblinear', scale_pos_weight=7, class_weight='balanced',n_jobs=-1))]

level1 = LogisticRegression(class_weight='balanced',random_state=0,solver='saga',n_jobs=-1)

model = StackingClassifier(estimators=level0, final_estimator=level1, n_jobs=-1)

model.fit(x_train,y_train)
df_test = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
X_test = df_test.drop(['id'],axis=1)

X_test = normalize(X_test)
y_pred_final = model.predict_proba(X_test)
df_test['target'] = y_pred_final[:,1]

df_test.loc[:,['id','target']].to_csv('2017A7PS0144G.csv',index=False)
