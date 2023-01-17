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
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import tqdm as tqdm



%matplotlib inline
df_train = pd.read_csv("../input/minor-project-2020/train.csv")

df_test = pd.read_csv("../input/minor-project-2020/test.csv")
X_test = df_test.drop(["id"],axis=1)

Y_train=df_train["target"]

X_train=df_train.drop(["id","target"],axis=1)

X_test.head()


corr = X_train.corr()

print (corr)

f, ax = plt.subplots(figsize=(11, 9))

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

#sns.heatmap(corr, mask=mask, cmap="Reds",square=True, ax=ax,vmax=1, vmin = -1, center=0,linewidths=.5, cbar_kws={"shrink": .5})

sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(240, 10, n=9),square=True, ax=ax,vmax=1,vmin=-1,center=0,linewidths=.5, cbar_kws={"shrink": .5})

#sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 40, as_cmap=True),square=True, ax=ax,vmax=1, vmin = -1, center=0,linewidths=.5, cbar_kws={"shrink": .5})
# cor_target = abs(cor["MEDV"])

# #Selecting highly correlated features

# relevant_features = cor_target[cor_target>0.5]

# relevant_features
from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler = MinMaxScaler() 

X_train = scaler.fit_transform(X_train) 

X_test = scaler.fit_transform(X_test)

from statistics import mean, stdev 

from sklearn import preprocessing 

from sklearn.model_selection import StratifiedKFold 

from sklearn import linear_model 

from sklearn import datasets  

from sklearn.metrics import roc_auc_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline

from tqdm import tqdm

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import train_test_split


# def get_redundant_pairs(df):

#     '''Get diagonal and lower triangular pairs of correlation matrix'''

#     pairs_to_drop = set()

#     cols = df.columns

#     for i in range(0, df.shape[1]):

#         for j in range(0, i+1):

#             pairs_to_drop.add((cols[i], cols[j]))

#     return pairs_to_drop



# def get_top_abs_correlations(df, n=5):

#     au_corr = df.corr().abs().unstack()

#     labels_to_drop = get_redundant_pairs(df)

#     au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

#     return au_corr[0:n]
# print("Top Absolute Correlations")

# num_correlation = 10

# top_correlations = get_top_abs_correlations(X_train, num_correlation)

# print(top_correlations)
model = []

names =[]

AUCdict = {}

result = {}
from xgboost import XGBClassifier

class_weight = int(Y_train.value_counts()[0]/Y_train.value_counts()[1])

model.append(XGBClassifier(scale_pos_weight=class_weight,seed=42))

names.append("XGBoost")

AUCdict[names[-1]]=[]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

for train_index, test_index in skf.split(X_train,Y_train): 

    x_train_fold, x_test_fold = X_train[train_index], X_train[test_index] 

    y_train_fold, y_test_fold = Y_train[train_index], Y_train[test_index] 

    model[-1].fit(x_train_fold, y_train_fold) 

    currAUC = roc_auc_score(y_test_fold, model[-1].predict(x_test_fold))

    AUCdict[names[-1]].append(currAUC)

    if (len(AUCdict[names[-1]])==1):

        result[names[-1]] = model[-1].predict(X_test)

    elif (currAUC > max(AUCdict[names[-1]])):result[names[-1]] = model[-1].predict(X_test)

print('Max ROC AUC in XGBoost: %.3f' % max(AUCdict[names[-1]]))
AnsXG = pd.DataFrame(df_test["id"],columns=["id"])

AnsXG["target"] =pd.Series(result[names[-1]])

AnsXG.to_csv('AnsXG.csv',index=False) 
over = SMOTE(sampling_strategy=0.4)

# under = RandomUnderSampler(sampling_strategy=1)

steps = [('over', over)]

pipeline = Pipeline(steps=steps)

X_train_Smote,Y_train_Smote  = pipeline.fit_resample(X_train, Y_train)
from sklearn.linear_model import LogisticRegressionCV

model.append(LogisticRegressionCV(random_state=0, n_jobs=-1,max_iter=1000))

names.append("LR")

AUCdict[names[-1]]=[]

l = []





skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for train_index, test_index in tqdm(skf.split(X_train_Smote,Y_train_Smote)): 

    x_train_fold, x_test_fold = X_train_Smote[train_index], X_train_Smote[test_index] 

    y_train_fold, y_test_fold = Y_train_Smote[train_index], Y_train_Smote[test_index] 

    model[-1].fit(x_train_fold, y_train_fold) 

    currAUC = roc_auc_score(y_test_fold, model[-1].predict(x_test_fold))

    AUCdict[names[-1]].append(currAUC)

    l.append((list)(model[-1].predict(X_test)))



print('Max ROC AUC in SMOTE AND Logistic Regression: %.3f' % max(AUCdict[names[-1]]))
ans=[]

ones =0

for j in range(200000):

    for i in range(5):

        if l[i][j]==1:

            ones =ones+1;

    if(ones >= 5-ones):ans.append(1)

    else:ans.append(0)

    ones = 0

AnsLR = pd.DataFrame(df_test["id"],columns=["id"])

AnsLR["target"] =pd.Series(ans)

print(AnsLR)

AnsLR.to_csv('AnsLR.csv',index=False) 
# from sklearn.ensemble import RandomForestClassifier

# class_weight = int(pd.Series(Y_train_Smote).value_counts()[0]/(pd.Series(Y_train_Smote)).value_counts()[1])

# model.append(RandomForestClassifier(n_estimators=150,class_weight= {0:1,1:class_weight}))

# names.append("RF")

# AUCdict[names[-1]]=[]

# l = []

# skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1)

# for train_index, test_index in tqdm(skf.split(X_train_Smote,Y_train_Smote),total=8): 

#     x_train_fold, x_test_fold = X_train_Smote[train_index], X_train_Smote[test_index] 

#     y_train_fold, y_test_fold = Y_train_Smote[train_index], Y_train_Smote[test_index] 

#     model[-1].fit(x_train_fold, y_train_fold) 

#     currAUC = roc_auc_score(y_test_fold, model[-1].predict(x_test_fold))

#     AUCdict[names[-1]].append(currAUC)

#     l.append((list)(model[-1].predict(X_test)))



# print('Max ROC AUC in SMOTE AND RF: %.3f' % max(AUCdict[names[-1]]))
# ans=[]

# ones =0

# for j in range(200000):

#     for i in range(5):

#         if l[i][j]==1:

#             ones =ones+1;

#     if(ones >= 5-ones):ans.append(1)

#     else:ans.append(0)

#     ones = 0



# ans = pd.Series(l)

# AnsRF1 = pd.DataFrame(df_test["id"],columns=["id"])

# AnsRF1["target"] =pd.Series(ans)


