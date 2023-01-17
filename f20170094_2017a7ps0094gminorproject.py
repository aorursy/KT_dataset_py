import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import seaborn as sns
%matplotlib inline
train_df = pd.read_csv("../input/minor-project-2020/train.csv")
test_df = pd.read_csv("../input/minor-project-2020/test.csv")
x = train_df.drop(["id", "target"], axis=1)
y = train_df["target"]
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state = 121)
train_data = pd.concat([x_train, y_train], axis=1)
negative = train_data[train_data.target==0]
positive = train_data[train_data.target==1]

# upsample minority
pos_upsampled = resample(positive,
 replace=True, # sample with replacement
 n_samples=len(negative), # match number in majority class
 random_state=27) # reproducible results# combine majority and upsampled minority
upsampled = pd.concat([negative, pos_upsampled])
upsampled = upsampled.sample(frac=1).reset_index(drop=True)
x_train = upsampled.drop(["target"], axis=1)
y_train = upsampled["target"]
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()              #Instantiate the scaler
x_train = scaler.fit_transform(x_train) 
x_val = scaler.transform(x_val)
# 0.66971, highest
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
brf = BalancedRandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
random_grid = {}
gs = RandomizedSearchCV(scoring='roc_auc', estimator = brf, param_distributions = random_grid, n_iter = 100, cv = 4, verbose=2, random_state=42, n_jobs = 4)
gs.fit(x_train, y_train)

# 0.66967
# from sklearn.ensemble import AdaBoostClassifier
# gs = AdaBoostClassifier(n_estimators=100, random_state=0)
# gs.fit(x_train, y_train)
y_val_pred1 = gs.predict_proba(x_val)
y_val_pred1
y_val_pred = y_val_pred1[:,1]
y_val_pred
# plot_confusion_matrix(gs, x_val, y_val, cmap = plt.cm.Blues)
from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-pastel')

FPR, TPR, _ = roc_curve(y_val, y_val_pred)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)
x_test = test_df.drop(["id"], axis=1)
x_test = scaler.transform(x_test)
y_pred1 = gs.predict_proba(x_test)
y_pred = y_pred1[:,1]
res_df = pd.DataFrame()
res_df["id"] = test_df["id"]
res_df["target"] = y_pred
# res_df[res_df["target"]==1]
res_df.to_csv("brf_basic25.csv", index=False)
