# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import pandas as pd
import os

print(os.getcwd())

path_data = os.path.join("..", "input","default of credit card clients.xls")
df = pd.read_excel(path_data, header=1, index_col=0).rename(columns={"PAY_0": "PAY_1"})
df.head()
df[df.index.duplicated()]
df.describe()
df["LIMIT_BAL_log"] = np.log10(df.LIMIT_BAL)
df[[ 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6' ]].describe()
default = df[df["default payment next month"] == 1]
no_default = df[df["default payment next month"] == 0]
print(len(df), len(default), len(no_default))
print(len(default)/len(df), len(no_default)/ len(df),)
df.keys()
default["SEX"].hist()
no_default["SEX"].hist()
default["AGE"].hist()
no_default["AGE"].hist()
default["EDUCATION"].hist(bins=range(1,7))
no_default["EDUCATION"].hist(bins=range(1,7))
default["MARRIAGE"].hist()
no_default["MARRIAGE"].hist()
default["PAY_1"].hist(bins=range(-2,9))
no_default["PAY_1"].hist(bins=range(-2,9))
df[np.isnan(df.LIMIT_BAL_log)]
df.LIMIT_BAL_log.describe()
df.LIMIT_BAL_log.hist()
default.LIMIT_BAL_log.hist()
no_default.LIMIT_BAL_log.hist()
df.keys()
# Transformaciones

df["BILL_AMT1_P"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
df["BILL_AMT2_P"] = df["BILL_AMT2"] / df["LIMIT_BAL"]
df["BILL_AMT3_P"] = df["BILL_AMT3"] / df["LIMIT_BAL"]
df["BILL_AMT4_P"] = df["BILL_AMT4"] / df["LIMIT_BAL"]
df["BILL_AMT5_P"] = df["BILL_AMT5"] / df["LIMIT_BAL"]
df["BILL_AMT6_P"] = df["BILL_AMT6"] / df["LIMIT_BAL"]

delta = 60.5
p_l1 = (((df["PAY_AMT1"] + delta) / (df["BILL_AMT1"] + delta) + 1) - abs((df["PAY_AMT1"] + delta) / (df["BILL_AMT1"] + delta) - 1)) / 2
p_l2 = (((df["PAY_AMT2"] + delta) / (df["BILL_AMT2"] + delta) + 1) - abs((df["PAY_AMT2"] + delta) / (df["BILL_AMT2"] + delta) - 1)) / 2
p_l3 = (((df["PAY_AMT3"] + delta) / (df["BILL_AMT3"] + delta) + 1) - abs((df["PAY_AMT3"] + delta) / (df["BILL_AMT3"] + delta) - 1)) / 2
p_l4 = (((df["PAY_AMT4"] + delta) / (df["BILL_AMT4"] + delta) + 1) - abs((df["PAY_AMT4"] + delta) / (df["BILL_AMT4"] + delta) - 1)) / 2
p_l5 = (((df["PAY_AMT5"] + delta) / (df["BILL_AMT5"] + delta) + 1) - abs((df["PAY_AMT5"] + delta) / (df["BILL_AMT5"] + delta) - 1)) / 2
p_l6 = (((df["PAY_AMT6"] + delta) / (df["BILL_AMT6"] + delta) + 1) - abs((df["PAY_AMT6"] + delta) / (df["BILL_AMT6"] + delta) - 1)) / 2

df["PAY_AMT1_P"] = ((p_l1 - 1) + abs(p_l1 + 1)) / 2
df["PAY_AMT2_P"] = ((p_l2 - 1) + abs(p_l2 + 1)) / 2
df["PAY_AMT3_P"] = ((p_l3 - 1) + abs(p_l3 + 1)) / 2
df["PAY_AMT4_P"] = ((p_l4 - 1) + abs(p_l4 + 1)) / 2
df["PAY_AMT5_P"] = ((p_l5 - 1) + abs(p_l5 + 1)) / 2
df["PAY_AMT6_P"] = ((p_l6 - 1) + abs(p_l6 + 1)) / 2

df["DELTA_PAY_AMT1_P"] = df["PAY_AMT1_P"] - df["PAY_AMT2_P"]
df["DELTA_PAY_AMT2_P"] = df["PAY_AMT2_P"] - df["PAY_AMT3_P"]
df["DELTA_PAY_AMT3_P"] = df["PAY_AMT3_P"] - df["PAY_AMT4_P"]
df["DELTA_PAY_AMT4_P"] = df["PAY_AMT4_P"] - df["PAY_AMT5_P"]
df["DELTA_PAY_AMT5_P"] = df["PAY_AMT5_P"] - df["PAY_AMT6_P"]

df["DELTA_BILL_AMT1_P"] = df["BILL_AMT1_P"] - df["BILL_AMT2_P"]
df["DELTA_BILL_AMT2_P"] = df["BILL_AMT2_P"] - df["BILL_AMT3_P"]
df["DELTA_BILL_AMT3_P"] = df["BILL_AMT3_P"] - df["BILL_AMT4_P"]
df["DELTA_BILL_AMT4_P"] = df["BILL_AMT4_P"] - df["BILL_AMT5_P"]
df["DELTA_BILL_AMT5_P"] = df["BILL_AMT5_P"] - df["BILL_AMT6_P"]
df[["BILL_AMT1_P", "BILL_AMT2_P", "BILL_AMT3_P", "BILL_AMT4_P", "BILL_AMT5_P", "BILL_AMT6_P"]].describe()
df[["PAY_AMT1_P", "PAY_AMT2_P", "PAY_AMT3_P", "PAY_AMT4_P", "PAY_AMT5_P", "PAY_AMT6_P"]].describe()
fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2,3, figsize=(15, 5))

df.PAY_AMT1_P.hist(ax=ax1, bins=50)
df.PAY_AMT2_P.hist(ax=ax2, bins=50)
df.PAY_AMT3_P.hist(ax=ax3, bins=50)
df.PAY_AMT4_P.hist(ax=ax4, bins=50)
df.PAY_AMT5_P.hist(ax=ax5, bins=50)
df.PAY_AMT6_P.hist(ax=ax6, bins=50)

plt.tight_layout()
df[df.PAY_AMT1_P > 10000][["PAY_AMT1_P", "PAY_AMT1", "BILL_AMT1"]]
fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2,3, figsize=(15, 5))

df.BILL_AMT1_P.plot(ax=ax1, title="BILL_AMT1_P")
df.BILL_AMT2_P.plot(ax=ax2, title="BILL_AMT2_P")
df.BILL_AMT3_P.plot(ax=ax3, title="BILL_AMT3_P")
df.BILL_AMT4_P.plot(ax=ax4, title="BILL_AMT4_P")
df.BILL_AMT5_P.plot(ax=ax5, title="BILL_AMT5_P")
df.BILL_AMT6_P.plot(ax=ax6, title="BILL_AMT6_P")

plt.tight_layout()
fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2,3, figsize=(15, 5))

df.DELTA_PAY_AMT1_P.plot(ax=ax1, title="DELTA_PAY_AMT1_P")
df.DELTA_PAY_AMT2_P.plot(ax=ax2, title="DELTA_PAY_AMT2_P")
df.DELTA_PAY_AMT3_P.plot(ax=ax3, title="DELTA_PAY_AMT3_P")
df.DELTA_PAY_AMT4_P.plot(ax=ax4, title="DELTA_PAY_AMT4_P")
df.DELTA_PAY_AMT5_P.plot(ax=ax5, title="DELTA_PAY_AMT5_P")

plt.tight_layout()
fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2,3, figsize=(15, 5))

df.DELTA_BILL_AMT1_P.plot(ax=ax1, title="DELTA_BILL_AMT1_P")
df.DELTA_BILL_AMT2_P.plot(ax=ax2, title="DELTA_BILL_AMT2_P")
df.DELTA_BILL_AMT3_P.plot(ax=ax3, title="DELTA_BILL_AMT3_P")
df.DELTA_BILL_AMT4_P.plot(ax=ax4, title="DELTA_BILL_AMT4_P")
df.DELTA_BILL_AMT5_P.plot(ax=ax5, title="DELTA_BILL_AMT5_P")

plt.tight_layout()
def balance_log_buckets(limit_bill, f=1./2):
    centroides = [4, 4.5, 5, 5.5, 6]
    bk = np.array([np.exp(-1 * abs(limit_bill[0] - c) * f) for c in centroides])
    bk_n = bk / bk.sum() # Escala manual
    return bk_n
def age_buckets(age, f=1./10):
    centroides = [20, 30, 40, 50, 60, 70]
    bk = np.array([np.exp(-1 * abs(age[0] - c) * f) for c in centroides])
    bk_n = bk / bk.sum() # Escala manual
    return bk_n
v = np.vectorize(balance_log_buckets, signature="(1)->(5)",)
v(np.array([[4], [5.3]]))
df.keys()
cols = ['LIMIT_BAL_log', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
        'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
        'BILL_AMT1_P', 'BILL_AMT2_P', 'BILL_AMT3_P', 'BILL_AMT4_P', 'BILL_AMT5_P', 'BILL_AMT6_P', 
        'PAY_AMT1_P', 'PAY_AMT2_P', 'PAY_AMT3_P', 'PAY_AMT4_P', 'PAY_AMT5_P', 'PAY_AMT6_P', 
        'DELTA_PAY_AMT1_P', 'DELTA_PAY_AMT2_P', 'DELTA_PAY_AMT3_P', 'DELTA_PAY_AMT4_P', 'DELTA_PAY_AMT5_P',
        'DELTA_BILL_AMT1_P', 'DELTA_BILL_AMT2_P', 'DELTA_BILL_AMT3_P', 'DELTA_BILL_AMT4_P', 'DELTA_BILL_AMT5_P']
len(cols)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

limit_bill_transformer = FunctionTransformer(np.vectorize(balance_log_buckets, signature="(1)->(5)"))
age_transformer = FunctionTransformer(np.vectorize(age_buckets, signature="(1)->(6)"))
indentity_transformer = FunctionTransformer(np.vectorize(lambda x: x))



ct = ColumnTransformer([("LimitBillTransformer", limit_bill_transformer , [0]),
                        ("SexTransformer", OneHotEncoder(), [1]),
                        ("EducationTransformer", OneHotEncoder(), [2]),
                        ("MarriageTransformer", OneHotEncoder(),  [3]),
                        ("AgeTransformer", age_transformer, [4]),
                        
                        ("Pay1Transformer", indentity_transformer, [5]),
                        ("Pay2Transformer", indentity_transformer, [6]),
                        ("Pay3Transformer", indentity_transformer, [7]),
                        ("Pay4Transformer", indentity_transformer, [8]),
                        ("Pay5Transformer", indentity_transformer, [9]),
                        ("Pay6Transformer", indentity_transformer, [10]),
                        
                        ("BillAmountP1Transformer", indentity_transformer, [11]),
                        ("BillAmountP2Transformer", indentity_transformer, [12]),
                        ("BillAmountP3Transformer", indentity_transformer, [13]),
                        ("BillAmountP4Transformer", indentity_transformer, [14]),
                        ("BillAmountP5Transformer", indentity_transformer, [15]),
                        ("BillAmountP6Transformer", indentity_transformer, [16]),
                        
                        ("PayAmountP1Transformer", indentity_transformer, [17]),
                        ("PayAmountP2Transformer", indentity_transformer, [18]),
                        ("PayAmountP3Transformer", indentity_transformer, [19]),
                        ("PayAmountP4Transformer", indentity_transformer, [20]),
                        ("PayAmountP5Transformer", indentity_transformer, [21]),
                        ("PayAmountP6Transformer", indentity_transformer, [22]),

                        ("BillDeltaP1Transformer", indentity_transformer, [23]),
                        ("BillDeltaP2Transformer", indentity_transformer, [24]),
                        ("BillDeltaP3Transformer", indentity_transformer, [25]),
                        ("BillDeltaP4Transformer", indentity_transformer, [26]),
                        ("BillDeltaP5Transformer", indentity_transformer, [27]),
                        
                        ("PayDeltatP1Transformer", indentity_transformer, [28]),
                        ("PayDeltatP2Transformer", indentity_transformer, [29]),
                        ("PayDeltatP3Transformer", indentity_transformer, [30]),
                        ("PayDeltatP4Transformer", indentity_transformer, [31]),
                        ("PayDeltatP5Transformer", indentity_transformer, [32]),
                       ], remainder="passthrough")
from sklearn.utils import resample


def re_sample_x_y(X, y, n_samples, label_resample=1,): 
    
    X_y_train = np.concatenate([X, np.array([y]).T], axis=1)
    X_y_pos = X_y_train[X_y_train[:,-1] == label_resample]
    X_y_neg = X_y_train[X_y_train[:,-1] != label_resample]
    X_y_pos_resample = resample(X_y_pos, n_samples=n_samples) 
    X_y_ = np.concatenate([X_y_pos_resample, X_y_neg])            
    
    X_ = X_y_[:,:-1]
    y_ = X_y_[:,-1].astype(int)

    return X_, y_
positivos = df[df["default payment next month"] == 1]
negativos = df[df["default payment next month"] == 0] 
len(positivos)/len(df), len(negativos)/len(df)
int(len(negativos) / ( (1 / 0.8) - 1))
def resample_df(dataframe, n_sample, x_cols=cols, y_cols="default payment next month", pos_label=1):
    
    df_train_pos = dataframe[dataframe[y_cols] == pos_label]
    df_train_neg = dataframe[dataframe[y_cols] != pos_label]
    
    df_train_pos_resample = df_train_pos.sample(n_sample, replace=True)
    df_resample = pd.concat([df_train_pos_resample, df_train_neg], axis=0).sample(frac=1)
    
    X_ = df_resample[x_cols].values
    y_ = df_resample[y_cols].values
    
    return X_,y_
X = df[cols].values
y = df["default payment next month"].values
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.33)
# X_train = df_train[cols].values
# y_train = df_train["default payment next month"].values
X_train, y_train = resample_df(df_train, 93456)
len(y_train[y_train == 1])
X_test = df_test[cols].values
y_test = df_test["default payment next month"].values
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

pca = PCA(n_components=30)
csf_MLPC = MLPClassifier(hidden_layer_sizes=(100,))

piper = Pipeline([("transformer", ct), ("scaler", StandardScaler()), ("csf", csf_MLPC)])
piper.fit(X_train, y_train)    
score_train = piper.score(X_train, y_train)
score_test = piper.score(X_test, y_test)
print("score train: {}, socre test: {}".format(score_train, score_test))
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

pca = PCA(n_components=30)
csf_DTC = DecisionTreeClassifier(max_depth=3)
clf_ABC = AdaBoostClassifier(csf_DTC)

piper2 = Pipeline([("transformer", ct), ("scaler", StandardScaler()), ("csf", clf_ABC)])
piper2.fit(X_train, y_train)
score_train = piper2.score(X_train, y_train)
score_test = piper2.score(X_test, y_test)
print("score train: {}, socre test: {}".format(score_train, score_test))
from sklearn import metrics

fig, ax = plt.subplots(1, figsize=(15,5))
metrics.plot_roc_curve(piper, X_test, y_test, ax=ax)
metrics.plot_roc_curve(piper2, X_test, y_test, ax=ax)
piper2.named_steps.csf.estimators_
from sklearn import tree

fig, ax = plt.subplots(1, figsize=(15,5))
tree.plot_tree(piper2.named_steps.csf.estimators_[0], ax=ax) 
from sklearn.tree import export_text

r = export_text(piper2.named_steps.csf.estimators_[1])
print(r)
piper2.named_steps.csf.estimator_weights_
piper2.named_steps.csf.estimator_errors_
len_positives = [int(len(negativos) / ( (1 / p) - 1)) for p in [0.3, 0.4, 0.5, 0.8]]
len_positives
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics

info_fit = list()


for i, n_pos_samples in enumerate(len_positives):
    
    df_train, df_test = train_test_split(df, test_size=0.33)
    
    X_train, y_train = resample_df(df_train, n_pos_samples)
    
    X_test = df_test[cols].values
    y_test = df_test["default payment next month"].values
    
    piper.fit(X_train, y_train)
    
    score_train = piper.score(X_train, y_train)
    
    score_test = piper.score(X_test, y_test)
    
    y_predict = piper.predict(X_test)
    
    cm = confusion_matrix(y_test, y_predict, labels=[0,1])
    
    scores_proba = piper.predict_proba(X_test)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores_proba[:,1], pos_label=1)
    
    info_fit.append({"score_train": score_train, "score_test": score_test, "confusion_matrix": cm, "roc_curve": (fpr, tpr, thresholds)})
    
    print("iter done: {}, score train: {}, score test: {}".format(i,score_train,score_test))       
fig,ax = plt.subplots(1, figsize=(15,5))
ax.plot(info_fit[0]["roc_curve"][0], info_fit[0]["roc_curve"][1], label="proprocion: [0.30, 0.70]", marker=",", )
ax.plot(info_fit[1]["roc_curve"][0], info_fit[1]["roc_curve"][1], label="proprocion: [0.40, 0.60]", marker=",", )
ax.plot(info_fit[2]["roc_curve"][0], info_fit[2]["roc_curve"][1], label="proprocion: [0.50, 0.50]", marker=",", )
ax.plot(info_fit[3]["roc_curve"][0], info_fit[3]["roc_curve"][1], label="proprocion: [0.80, 0.20]", marker=",",)
plt.legend()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics

info_fit2 = list()


for i, n_pos_samples in enumerate(len_positives):
    
    df_train, df_test = train_test_split(df, test_size=0.33)
    
    X_train, y_train = resample_df(df_train, n_pos_samples)
    
    X_test = df_test[cols].values
    y_test = df_test["default payment next month"].values
    
    piper2.fit(X_train, y_train)
    
    score_train = piper2.score(X_train, y_train)
    
    score_test = piper2.score(X_test, y_test)
    
    y_predict = piper2.predict(X_test)
    
    cm = confusion_matrix(y_test, y_predict, labels=[0,1])
    
    scores_proba = piper2.predict_proba(X_test)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores_proba[:,1], pos_label=1)
    
    info_fit2.append({"score_train": score_train, "score_test": score_test, "confusion_matrix": cm, "roc_curve": (fpr, tpr, thresholds)})
    
    print("iter done: {}, score train: {}, score test: {}".format(i,score_train,score_test))    
fig,ax = plt.subplots(1, figsize=(15,5))
ax.plot(info_fit2[0]["roc_curve"][0], info_fit2[0]["roc_curve"][1], label="proprocion: [0.30, 0.70]", marker=",", )
ax.plot(info_fit2[1]["roc_curve"][0], info_fit2[1]["roc_curve"][1], label="proprocion: [0.40, 0.60]", marker=",", )
ax.plot(info_fit2[2]["roc_curve"][0], info_fit2[2]["roc_curve"][1], label="proprocion: [0.50, 0.50]", marker=",", )
ax.plot(info_fit2[3]["roc_curve"][0], info_fit2[3]["roc_curve"][1], label="proprocion: [0.80, 0.20]", marker=",",)
plt.legend()




