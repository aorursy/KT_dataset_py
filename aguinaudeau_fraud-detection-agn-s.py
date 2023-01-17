# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from mlxtend.plotting import plot_decision_regions



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_curve, auc

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import LocalOutlierFactor

from sklearn.ensemble import IsolationForest



from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE
df = pd.read_csv('../input/creditcard.csv')
df.head()
df.info()
df.describe()
cat_col = ['Class']



cont_cols = list(df.columns)

cont_cols.remove('Class')
df['Class'].value_counts()
fraud_ratio = df['Class'].value_counts()[1] / df['Class'].value_counts().sum()

print(f'Fraud ratio in the dataset: {round(100 * fraud_ratio, 2)}%')
plt.figure()

sns.countplot(df['Class'])

plt.show()
fig = plt.figure(figsize=(50, 400))

fig.subplots_adjust(hspace=0.1, wspace=0.1)



# f= plt.figure()

# f.set_figheight(15)

# f.set_figwidth(15)



for i, col in enumerate(cont_cols):

    ax = fig.add_subplot(30, 3, (3 * i + 1))

    sns.kdeplot(df.loc[df['Class'] == 0, col], label = 'Class == 0')

    sns.kdeplot(df.loc[df['Class'] == 1, col], label = 'Class == 1')

    plt.xlabel(col)

    plt.ylabel('Density')

    plt.title(f'Distribution of {col} per class')

    

    ax2 = fig.add_subplot(30, 3, (3 * i + 2))

    sns.distplot(df[col])

    plt.title(f'Distribution of {col}')    

    

    ax3 = fig.add_subplot(30, 3, (3 * i + 3))

    sns.boxplot(df[col])

    plt.title(f'Boxplot of {col}') 

    

plt.rcParams.update({'font.size': 20})

#plt.tight_layout(pad = 2.5);
df_clean = df.copy()
df_clean['Time_hours'] = np.floor((df_clean['Time'] / 3600) % 24)

df_clean['Time_hours']
col = 'Time_hours'



plt.rcParams.update({'font.size': 10})



plt.figure()

sns.distplot(df_clean[col])

plt.title(f'Distribution of {col}') 

plt.show()



plt.figure()

sns.kdeplot(df_clean.loc[df_clean['Class'] == 0, col], label = 'Class == 0')

sns.kdeplot(df_clean.loc[df_clean['Class'] == 1, col], label = 'Class == 1')

plt.xlabel(col)

plt.ylabel('Density')

plt.title(f'Distribution of {col} per class')

plt.show()
col = 'Amount'

df_amount_under_5000 = df_clean.loc[df_clean['Amount'] > 5000]



plt.rcParams.update({'font.size': 10})



plt.figure()

sns.distplot(df_amount_under_5000[col])

plt.title(f'Distribution of {col}') 

plt.show()



plt.figure()

sns.kdeplot(df_amount_under_5000.loc[df_amount_under_5000['Class'] == 0, col], label = 'Class == 0')

sns.kdeplot(df_amount_under_5000.loc[df_amount_under_5000['Class'] == 1, col], label = 'Class == 1')

plt.xlabel(col)

plt.ylabel('Density')

plt.title(f'Distribution of {col} per class')

plt.show()
corr = df_clean.corr()
plt.figure(figsize=(20, 20))

sns.heatmap(corr.abs())

plt.show()
X = df_clean.drop(['Class'], axis=1)

y = df_clean['Class']

X.shape, y.shape
scaler = StandardScaler()

scaler.fit(X, y)

X_scaled = scaler.transform(X)

X_scaled.shape
%matplotlib inline



pca2 = PCA(n_components=2)

# first we perform mean normalization

X_centered = X_scaled - X_scaled.mean(axis=0)

pca2.fit(X_centered)

X_pca2 = pca2.transform(X_centered)



# Then we plot the results of PCA

plt.plot(X_pca2[y == 0, 0], X_pca2[y == 0, 1], 'o', label='Class 0')

plt.plot(X_pca2[y == 1, 0], X_pca2[y == 1, 1], 'o', label='Class 1')



plt.legend(loc=0)

plt.title('PCA projection')

plt.show()
pca3 = PCA(n_components=3)

# first we perform mean normalization



pca3.fit(X_centered)

X_pca3 = pca3.transform(X_centered)
# Then we plot the results of PCA

%matplotlib notebook



fig = plt.figure(figsize=(16, 16))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca3[y == 0, 0], X_pca3[y == 0, 1], X_pca3[y == 0, 2], 'o', label='Class 0', alpha=0.1)

ax.scatter(X_pca3[y == 1, 0], X_pca3[y == 1, 1], X_pca3[y == 1, 2], 'x', label='Class 1')



plt.legend(loc=0)

plt.title('PCA projection')

plt.show()
X_sample = X_scaled[:1000, :]

y_sample = y[:1000]

y_sample.shape, X_tsne2.shape
tsne2 = TSNE(n_components=2)



# Here we perform the t-SNE

X_tsne2 = tsne2.fit_transform(X_sample)



# Then we plot the results of t-SNE

plt.plot(X_tsne2[y == 0, 0], X_tsne2[y_sample == 0, 1], 'o', label='Class 0')

plt.plot(X_tsne2[y == 1, 0], X_tsne2[y_sample == 1, 1], 'o', label='Class 1')

plt.legend(loc=0)

plt.title('t-SNE projection')

plt.show()
tsne3 = TSNE(n_components=3)



# Here we perform the t-SNE

X_tsne3 = tsne3.fit_transform(X_sample)



# Then we plot the results of t-SNE

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



ax.scatter(X_tsne3[y == 0, 0], X_tsne3[y == 0, 1], X_tsne3[y == 1, 2], 'o', label='Class 0')

ax.scatter(X_tsne3[y == 1, 0], X_tsne3[y == 1, 1], X_tsne3[y == 1, 2], 'o', label='Class 1')

ax.legend(loc=0)



plt.title('t-SNE projection')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0, test_size=0.2)
lr1 = LogisticRegression()



lr1.fit(X_train, y_train)

y_pred = lr1.predict(X_test)

y_pred_proba = lr1.predict_proba(X_test)[:, 1]

score = f1_score(y_test, y_pred)

precisions_lr1, recalls_lr1, thresholds = precision_recall_curve(y_test, y_pred_proba)

auc_score = auc(recalls_lr1, precisions_lr1)

print(f'f1 score: {round(score, 2)}, auprc score :{round(auc_score, 2)}');
print(classification_report(y_test, y_pred))
rus = RandomUnderSampler(sampling_strategy='auto', return_indices=False, random_state=0, replacement=False, ratio=0.01)

X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
X_resampled.shape
np.unique(y_resampled, return_counts=True)
lr2 = LogisticRegression()



lr2.fit(X_resampled, y_resampled)

y_pred = lr2.predict(X_test)

y_pred_proba = lr2.predict_proba(X_test)[:, 1]

score = f1_score(y_test, y_pred)

precisions_lr2, recalls_lr2, thresholds = precision_recall_curve(y_test, y_pred_proba)

auc_score = auc(recalls_lr2, precisions_lr2)

print(f'f1 score: {round(score, 2)}, auprc score :{round(auc_score, 10)}');
print(classification_report(y_test, y_pred))
smote = SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=5, m_neighbors='deprecated', out_step='deprecated', kind='deprecated', svm_estimator='deprecated', n_jobs=1, ratio=0.01)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
y.shape
np.unique(y_resampled, return_counts=True)
lr3 = LogisticRegression()



lr3.fit(X_resampled, y_resampled)

y_pred = lr3.predict(X_test)

y_pred_proba = lr3.predict_proba(X_test)[:, 1]

score = f1_score(y_test, y_pred)

precisions_lr3, recalls_lr3, thresholds = precision_recall_curve(y_test, y_pred_proba)

auc_score = auc(recalls_lr3, precisions_lr3)

print(f'f1 score: {round(score, 2)}, auprc score :{round(auc_score, 2)}');
print(classification_report(y_test, y_pred))
X_sample = X_scaled[:1000, :]

y_sample = y[:1000]
# fit the model for outlier detection (default)

lof = LocalOutlierFactor(n_neighbors=30, contamination=0.01)

# use fit_predict to compute the predicted labels of the training samples

# (when LOF is used for outlier detection, the estimator has no predict,

# decision_function and score_samples methods).

y_pred = lof.fit_predict(X_sample)

y_pred[y_pred == 1] = 0

y_pred[y_pred == -1] = 1

n_errors = (y_pred != y_sample).sum()

X_scores = lof.negative_outlier_factor_

lof.offset_
y_pred.sum()
print(classification_report(y_sample, y_pred))
[0]  * 2 + [1] * 3
# fit the model for outlier detection (default)

lof = LocalOutlierFactor(n_neighbors=30, contamination=0.05, novelty=True)

# use fit_predict to compute the predicted labels of the training samples

# (when LOF is used for outlier detection, the estimator has no predict,

# decision_function and score_samples methods).

X_train = X_sample[y_sample==0]

lof.fit(X_train)



X_test_inliers = X_scaled[1000:1500, :][y[1000:1500] == 0]

X_test_outliers = X_scaled[1000:,:][y[1000:] == 1]

y_test = np.array([0] * len(X_test_inliers) + [1] * len(X_test_outliers))

X_test = np.concatenate([X_test_inliers, X_test_outliers])


y_pred = lof.predict(X_test)

y_pred[y_pred == 1] = 0

y_pred[y_pred == -1] = 1

n_errors = (y_pred != y_test).sum()

X_scores = lof.negative_outlier_factor_

print(n_errors)
y_pred.sum()
print(classification_report(y_test, y_pred))
precision_recall_curve(y_test, y_pred)
lof.__dict__
print("prediction errors: {}".format(n_errors))

print("Negative LOF scores: {}".format(lof.negative_outlier_factor_))

print("Offset (threshold to consider sample as anomaly or not): {}".format(lof.offset_))
IF = IsolationForest(contamination=0.002, behaviour="new") #contamination: amount of outliers to determine threshold value 

#for outlier detection ; behaviour new is to match scikit-learn decision_function method in other anomaly detection algorithms 

LOF = LocalOutlierFactor(n_neighbors=30, contamination=0.002, novelty=True) #novelty: allows to find new kind of outliers



# Fit the data

IF.fit(X_train)

LOF.fit(X_train)



# Perform the prediction

ypredIF = IF.predict(X_test)

ypredLOF = LOF.predict(X_test)



# Warning, those algorithm return -1 for anomaly and 1 for normal!!

ypredIF[ypredIF == 1] = 0

ypredIF[ypredIF == -1] = 1

ypredLOF[ypredLOF == 1] = 0

ypredLOF[ypredLOF == -1] = 1



# Print the results

print("##################\nIsolation Forest results:")

print(classification_report(y_test, ypredIF))



print("##################\nLOF results:")

print(classification_report(y_test, ypredLOF))
IF.__dict__
%matplotlib inline



plt.figure()

plt.plot(recalls_lr1, precisions_lr1, label='lr1')

plt.plot(recalls_lr2, precisions_lr2, label='lr2')

plt.plot(recalls_lr3, precisions_lr3, label='lr3')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend()

plt.title('Precision Recall Curve')

plt.show()