import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
sns.set_style('whitegrid')
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time
import warnings
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore", category=DeprecationWarning)
%matplotlib inline
plif_training_test = pd.read_csv('../input/lta4h-plif-active-inactive/lta4h_plif_active_inactive.csv')
le = LabelEncoder()
y_encoded = le.fit_transform(plif_training_test['activity'])
plif_training_test['activity'] = y_encoded

scaler = MinMaxScaler()
plif_training_test_scale = scaler.fit_transform(plif_training_test)
X_train, X_test, y_train, y_test = train_test_split(plif_training_test.drop('activity', axis=1), plif_training_test['activity'], test_size=0.25)
xgb1 = XGBClassifier(max_depth=40, learning_rate=0.01, n_estimators=200, gamma=0, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)
training_start = time.perf_counter()
xgb1.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = xgb1.predict(X_test)
prediction_end = time.perf_counter()
acc_xgb1 = (preds == y_test).sum().astype(float) / len(preds)*100
xgb1_train_time = training_end-training_start
xgb1_prediction_time = prediction_end-prediction_start

xgb1_cv = XGBClassifier(max_depth=40, learning_rate=0.01, n_estimators=200, gamma=0, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)
scores1 = cross_val_score(xgb1_cv, X_train, y_train, cv=10, scoring = "accuracy")
mean1 = scores1.mean()
sd1 = scores1.std()
acc_xgb1
plif_predict = pd.read_csv('../input/lta4h-plif-predict/lta4h_plif_predict.csv')
le = LabelEncoder()
y_encoded = le.fit_transform(plif_predict['activity'])
plif_predict['activity'] = y_encoded

scaler = MinMaxScaler()
plif_pred_scale = scaler.fit_transform(plif_predict)
