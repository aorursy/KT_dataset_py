# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from collections import Counter
import random

import sklearn
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import PolynomialFeatures
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import optuna

import matplotlib.pyplot as plt

random_state = 42
def save_submission(model, test_df, filename, probs=True):
    res1 = pd.DataFrame()
    if probs:
        res1["target"] = model.predict_proba(StandardScaler().fit_transform(test_df.drop(["id"], axis=1)))[:, 1]
    else:
        res1["target"] = model.predict(StandardScaler().fit_transform(test_df.drop(["id"], axis=1)))
    res1.set_index(test_df["id"], inplace=True)
    res1.to_csv(filename, index_label='id', header=["target"])
    
def get_roc_auc(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return roc_auc_score(y_test, y_pred)

def remove_outliers(df, threshold=5):
    tdf = df.drop(columns=["id", "target"], axis=0)
    scaled = StandardScaler().fit_transform(tdf)
    return df[((scaled > -threshold) & (scaled < threshold)).all(axis=1)]
    
df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')
test_df = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
df.describe()
c = Counter(df["target"])
print(f"Couts of labels: {c}")
print(f"Percent of minority labels: {(c[1]/c[0])*100:.4f}%")
print(f"Scale factor between majority and minority: {c[0]/c[1]:.4f}")
fig, axs = plt.subplots(ncols=9, nrows=10, figsize=(50, 50))
index = 0
axs = axs.flatten()
for k,v in df.drop(["id", "target"], axis=1).items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
X = df.drop(["id"], axis=1).drop(["target"], axis=1)
y = df.drop(["id"], axis=1)["target"]
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state, stratify=y)

split_scalar = StandardScaler()
scaled_X_train = split_scalar.fit_transform(X_train)
scaled_X_test = split_scalar.transform(X_test)

scaled_X_train.shape, scaled_X_test.shape, y_train.shape, y_test.shape
lg1 = LogisticRegression(solver='lbfgs', verbose=1, n_jobs=-1)
lg1.fit(scaled_X_train, y_train)
y_pred_disc = lg1.predict(scaled_X_test)
y_pred_prob = lg1.predict_proba(scaled_X_test)[:, 1]
print(f"AUC with 0/1 labels: {roc_auc_score(y_test, y_pred_disc):.5f}, with probs: {roc_auc_score(y_test, y_pred_prob):.5f}")
lg2 = LogisticRegression(solver='lbfgs', class_weight='balanced', verbose=1, n_jobs=-1)
lg2.fit(scaled_X_train, y_train)
y_pred_disc = lg2.predict(scaled_X_test)
y_pred_prob = lg2.predict_proba(scaled_X_test)[:, 1]
print(f"AUC with 0/1 labels: {roc_auc_score(y_test, y_pred_disc):.5f}, with probs: {roc_auc_score(y_test, y_pred_prob):.5f}")
lg3 = LogisticRegression(solver='saga', class_weight='balanced', verbose=1, n_jobs=-1)
lg3.fit(scaled_X_train, y_train)
y_pred_disc = lg3.predict(scaled_X_test)
y_pred_prob = lg3.predict_proba(scaled_X_test)[:, 1]
print(f"AUC with 0/1 labels: {roc_auc_score(y_test, y_pred_disc):.5f}, with probs: {roc_auc_score(y_test, y_pred_prob):.5f}")
lg3 = LogisticRegression(solver='saga', penalty='elasticnet', class_weight='balanced', verbose=1, n_jobs=-1, l1_ratio=0.5)
lg3.fit(scaled_X_train, y_train)
y_pred_disc = lg3.predict(scaled_X_test)
y_pred_prob = lg3.predict_proba(scaled_X_test)[:, 1]
print(f"AUC with 0/1 labels: {roc_auc_score(y_test, y_pred_disc):.5f}, with probs: {roc_auc_score(y_test, y_pred_prob):.5f}")
X_aug = np.concatenate([X, X**2], axis=1)

X_aug_train, X_aug_test, y_aug_train, y_aug_test = train_test_split(X_aug, y, test_size=0.33, random_state=random_state, stratify=y)

split_scalar = StandardScaler()
scaled_X_aug_train = split_scalar.fit_transform(X_aug_train)
scaled_X_aug_test = split_scalar.transform(X_aug_test)

scaled_X_aug_train.shape, scaled_X_aug_test.shape, y_aug_train.shape, y_aug_test.shape
lg5 = LogisticRegression(solver='lbfgs', class_weight='balanced', verbose=1, n_jobs=-1)
lg5.fit(scaled_X_aug_train, y_aug_train)
y_pred_disc = lg5.predict(scaled_X_aug_test)
y_pred_prob = lg5.predict_proba(scaled_X_aug_test)[:, 1]
print(f"AUC with 0/1 labels: {roc_auc_score(y_aug_test, y_pred_disc):.5f}, with probs: {roc_auc_score(y_aug_test, y_pred_prob):.5f}")
fs_dis_scores = []
fs_prob_scores = []
ks = [2, 5, 10, 20, 30, 40, 50, 80, 88]
for k in ks:
    fs = SelectKBest(score_func=f_classif, k=k)
    X_selected = fs.fit_transform(X, y)

    X_selected_train, X_selected_test, y_selected_train, y_selected_test = train_test_split(X_selected, y, test_size=0.33, random_state=random_state, stratify=y)

    split_scalar = StandardScaler()
    scaled_X_selected_train = split_scalar.fit_transform(X_selected_train)
    scaled_X_selected_test = split_scalar.transform(X_selected_test)

    lg = LogisticRegression(solver='lbfgs', class_weight='balanced', verbose=1, n_jobs=-1)
    lg.fit(scaled_X_selected_train, y_selected_train)
    
    y_pred_disc = lg.predict(scaled_X_selected_test)
    y_pred_prob = lg.predict_proba(scaled_X_selected_test)[:, 1]
    
    fs_dis_scores.append(roc_auc_score(y_selected_test, y_pred_disc))
    fs_prob_scores.append(roc_auc_score(y_selected_test, y_pred_prob))
    
    print(f"Evaluation for k={k} complete! 0/1 Score = {fs_dis_scores[-1]}, Prob Score = {fs_prob_scores[-1]}")

plt.plot(ks, fs_dis_scores, 'b-', label="0/1 score")
plt.plot(ks, fs_prob_scores, 'g-', label="Prob score")
plt.legend()
plt.grid()
def lr_objective(trial):
    k = trial.suggest_int("k", 10, 80)
    
    fs = SelectKBest(score_func=f_classif, k=k)
    X_selected = fs.fit_transform(X, y)
    X_selected_train, X_selected_test, y_selected_train, y_selected_test = train_test_split(X_selected, y, test_size=0.33, random_state=random_state, stratify=y)
    
    split_scalar = StandardScaler()
    scaled_X_selected_train = split_scalar.fit_transform(X_selected_train)
    scaled_X_selected_test = split_scalar.transform(X_selected_test)

    lg = LogisticRegression(solver='lbfgs', class_weight='balanced', verbose=1, n_jobs=-1)
    lg.fit(scaled_X_selected_train, y_selected_train)

    y_pred_prob = lg.predict_proba(scaled_X_selected_test)[:, 1]
    return roc_auc_score(y_selected_test, y_pred_prob)


study = optuna.create_study(direction="maximize")
study.optimize(lr_objective, n_trials=20)
fs = SelectKBest(score_func=f_classif, k=15)
X_selected = fs.fit_transform(X, y)

poly = PolynomialFeatures(2)
X_aug_selected = poly.fit_transform(X_selected)

X_aug_selected_train, X_aug_selected_test, y_aug_selected_train, y_aug_selected_test = train_test_split(X_aug_selected, y, test_size=0.33, random_state=random_state, stratify=y)

split_scalar = StandardScaler()
scaled_X_aug_selected_train = split_scalar.fit_transform(X_aug_selected_train)
scaled_X_aug_selected_test = split_scalar.transform(X_aug_selected_test)

lg7 = LogisticRegression(solver='lbfgs', class_weight='balanced', verbose=1, n_jobs=-1)
lg7.fit(scaled_X_aug_selected_train, y_aug_selected_train)
y_pred_disc = lg7.predict(scaled_X_aug_selected_test)
y_pred_prob = lg7.predict_proba(scaled_X_aug_selected_test)[:, 1]
print(f"AUC with 0/1 labels: {roc_auc_score(y_aug_selected_test, y_pred_disc):.5f}, with probs: {roc_auc_score(y_aug_selected_test, y_pred_prob):.5f}")
n_outs = []
tdf = df.drop(columns=["id", "target"], axis=0)
scaled = StandardScaler().fit_transform(tdf)
for threshold in range(1, 10):
    n_out = 800000 - sum(((scaled > -threshold) & (scaled < threshold)).all(axis=1))
    n_outs.append(n_out)
    print(f" Num of examples with feature values outside  of {threshold} sigmas: {n_out} ({(n_out / 800000) * 100:.2f}%)")
plt.plot(n_outs)
clean_df = remove_outliers(df)
clean_X = clean_df.drop(["id"], axis=1).drop(["target"], axis=1)
clean_y = clean_df.drop(["id"], axis=1)["target"]

clean_X_train, clean_X_test, clean_y_train, clean_y_test = train_test_split(clean_X, clean_y, test_size=0.33, random_state=random_state, stratify=clean_y)

split_scalar = StandardScaler()
scaled_clean_X_train = split_scalar.fit_transform(clean_X_train)
scaled_clean_X_test = split_scalar.transform(clean_X_test)

lg8 = LogisticRegression(solver='lbfgs', class_weight='balanced', verbose=1, n_jobs=-1)
lg8.fit(scaled_clean_X_train, clean_y_train)
y_pred_disc = lg8.predict(scaled_clean_X_test)
y_pred_prob = lg8.predict_proba(scaled_clean_X_test)[:, 1]
print(f"AUC with 0/1 labels: {roc_auc_score(clean_y_test, y_pred_disc):.5f}, with probs: {roc_auc_score(clean_y_test, y_pred_prob):.5f}")
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under)]
sampler = Pipeline(steps=steps)

scaled_X_r_train, y_r_train = sampler.fit_resample(scaled_X_train, y_train)

c = Counter(y_r_train)
print(scaled_X_r_train.shape, y_r_train.shape)
print(f"Couts of labels: {c}")
print(f"Percent of minority labels: {(c[1]/c[0])*100:.4f}%")
print(f"Scale factor between majority and minority: {c[0]/c[1]:.4f}")
lg9 = LogisticRegression(solver='lbfgs', class_weight='balanced', verbose=1, n_jobs=-1)
lg9.fit(scaled_X_r_train, y_r_train)
y_pred_disc = lg9.predict(scaled_X_test)
y_pred_prob = lg9.predict_proba(scaled_X_test)[:, 1]
print(f"AUC with 0/1 labels: {roc_auc_score(y_test, y_pred_disc):.5f}, with probs: {roc_auc_score(y_test, y_pred_prob):.5f}")
lg10 = LogisticRegression(solver='lbfgs', class_weight='balanced', verbose=1, n_jobs=-1)
lg10.fit(StandardScaler().fit_transform(X), y)
y_pred_disc = lg10.predict(scaled_X_test)
y_pred_prob = lg10.predict_proba(scaled_X_test)[:, 1]
print(f"AUC with 0/1 labels: {roc_auc_score(y_test, y_pred_disc):.5f}, with probs: {roc_auc_score(y_test, y_pred_prob):.5f}")
save_submission(lg10, test_df, "lg_full_data.csv")
xgb = XGBClassifier(n_jobs=-1, random_state=random_state)
xgb.fit(scaled_X_train, y_train, verbose=True, eval_set=[(scaled_X_test, y_test)], eval_metric="auc", early_stopping_rounds=5)
get_roc_auc(xgb, scaled_X_test, y_test)
xgb = XGBClassifier(n_jobs=-1, random_state=random_state, scale_pos_weight=445)
xgb.fit(scaled_X_train, y_train, verbose=True, eval_set=[(scaled_X_test, y_test)], eval_metric="auc", early_stopping_rounds=5)
get_roc_auc(xgb, scaled_X_test, y_test)
trainX = scaled_X_train
trainy = y_train
testX = scaled_X_test
testy = y_test

model = OneClassSVM(gamma='scale', nu=0.001)
trainX = trainX[trainy==0]
model.fit(trainX)
yhat = model.predict(testX)
testy[testy == 1] = -1
testy[testy == 0] = 1
roc_auc_score(testy, yhat)
dt = DecisionTreeClassifier()
dt = dt.fit(scaled_X_train, y_train)
get_roc_auc(dt, scaled_X_test, y_test)