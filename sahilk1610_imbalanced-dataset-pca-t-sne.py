
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
train = pd.read_csv('../input/anomaly-detection/Participants_Data_WH18/Train.csv')
train.head()
test = pd.read_csv("../input/anomaly-detection/Participants_Data_WH18/Test.csv")
test.head(2)
train["Class"].value_counts()
plt.hist(train["Class"])
for i in train.columns[:3]:
    plt.hist(train[i])
    plt.title(i)
    plt.show()
Sm = SMOTE()
X = train.drop("Class", axis = 1)
y = train["Class"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, stratify = y)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
x_train, y_train = Sm.fit_resample(X_train, y_train)
x_train.shape, y_train.shape, y_train.value_counts()
x_train_re, x_valid_re, y_train_re, y_valid_re = train_test_split(x_train, y_train, test_size = 0.1, stratify = y_train)
from sklearn.ensemble import RandomForestClassifier
import catboost
from sklearn.metrics import auc, roc_curve
def metric(preds, target):
    fpr, tpr, thresholds = roc_curve(target, preds)
    return auc(fpr, tpr)
Rf = RandomForestClassifier()
model_Rf = Rf.fit(x_train_re, y_train_re)
preds = model_Rf.predict(x_valid_re)
print(metric(preds, y_valid_re))

pca = PCA(n_components=3, random_state=52)
pca_result = pca.fit_transform(X)
print(pca.explained_variance_ratio_)

train_copy = train.copy()
train_copy['pca-one'] = pca_result[:,0]
train_copy['pca-two'] = pca_result[:,1] 
train_copy['pca-three'] = pca_result[:,2]
rndperm = np.random.permutation(train_copy.shape[0])
plt.figure(figsize=(10,8))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="Class",
    palette=sns.color_palette("hls", 2),
    data= train_copy.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
ax = plt.figure(figsize=(10,8)).gca(projection='3d')
ax.scatter(
    xs=train_copy.loc[rndperm,:]["pca-one"], 
    ys=train_copy.loc[rndperm,:]["pca-two"], 
    zs=train_copy.loc[rndperm,:]["pca-three"], 
    c=train_copy.loc[rndperm,:]["Class"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()
import time
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)
train_copy['tsne-2d-one'] = tsne_results[:,0]
train_copy['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(10,8))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="Class",
    palette=sns.color_palette("hls", 2),
    data=train_copy,
    legend="full",
    alpha=0.3
)
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(X)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
train_copy['tsne-pca50-one'] = tsne_pca_results[:,0]
train_copy['tsne-pca50-two'] = tsne_pca_results[:,1]
plt.figure(figsize=(16,4))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="Class",
    palette=sns.color_palette("hls", 2),
    data=train_copy,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="Class",
    palette=sns.color_palette("hls", 2),
    data=train_copy,
    legend="full",
    alpha=0.3,
    ax=ax2
)
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="Class",
    palette=sns.color_palette("hls", 2),
    data=train_copy,
    legend="full",
    alpha=0.3,
    ax=ax3
)
train = pd.read_csv('../input/anomaly-detection/Participants_Data_WH18/Train.csv')
train.head(2)
train = train.T.drop_duplicates().T
train.shape
plt.rcParams['figure.figsize'] = 20,6
plt.subplot(131)
sns.boxplot(train["Class"], train["feature_1"])
plt.subplot(132)
sns.boxplot(train["Class"], train["feature_2"])
plt.subplot(133)
sns.boxplot(train["Class"], train["feature_3"])

df = pd.DataFrame((train == 0).astype(int).sum(axis=0))
df
all_zero = df[df[0]>1761].index

train.drop(all_zero,axis=1,inplace=True)
train.info()
X = train.drop("Class", axis = 1)
y = train["Class"]

X_train, X_valid , y_train, y_valid = train_test_split(X, y, test_size = 0.2, stratify = y)
from xgboost import XGBClassifier
model = XGBClassifier(silent=True,
                      booster = 'gbtree',
                      scale_pos_weight=5,
                      learning_rate=0.01,  
                      colsample_bytree = 0.7,
                      subsample = 0.5,
                      max_delta_step = 3,
                      reg_lambda = 2,
                     objective='binary:logistic',
                      
                      n_estimators=818, 
                      max_depth=8,
                     )
%%time
eval_set = [(X_valid, y_valid)]
eval_metric = ["logloss"]
model.fit(X_train, y_train,early_stopping_rounds=50, eval_metric=eval_metric, eval_set=eval_set)
predictions = model.predict_proba(X_valid)[:, -1]

score = roc_auc_score(y_valid, predictions)
score
pca_3 = PCA(n_components=3)
pca_result_3 = pca_3.fit_transform(X)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
df_3 = pd.DataFrame(pca_result_3, columns=["pca1", 'pca2', 'pca3'])

df_3.head(2)
X_train, X_valid , y_train, y_valid = train_test_split(df_3, y, test_size = 0.2, stratify = y)
%%time
eval_set = [(X_valid, y_valid)]
eval_metric = ["logloss"]
model.fit(X_train, y_train,early_stopping_rounds=50, eval_metric=eval_metric, eval_set=eval_set)
predictions = model.predict_proba(X_valid)[:, -1]

score2 = roc_auc_score(y_valid, predictions)
score2
