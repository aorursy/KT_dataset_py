# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
original_df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
original_df.head()
print("Count of classes available")
print(pd.Index(original_df['Class']).value_counts())
sns.set(style="darkgrid")
ax = sns.countplot(x="Class", data=original_df)
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = original_df['Amount'].values
time_val = original_df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='b')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='g')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])


plt.show()
from sklearn.preprocessing import RobustScaler
rob_scaler = RobustScaler()
original_df['scaled_amount'] = rob_scaler.fit_transform(original_df['Amount'].values.reshape(-1,1))
original_df['scaled_time'] = rob_scaler.fit_transform(original_df['Time'].values.reshape(-1,1))
original_df.drop(['Time','Amount'], axis=1, inplace=True)
# Shuffling our data
original_df.sample(frac=1).head()
from sklearn.manifold import TSNE
def plot_graph_tsne(X,y):
    pca_2d = TSNE(n_components=2, random_state=42).fit_transform(X.values)
    colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y]
    kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}
    fig = plt.Figure(figsize=(12,6))
    plt.scatter(pca_2d[:, 0],pca_2d[:, 1], c=colors, **kwarg_params, label="Fraud")
    plt.legend()
    sns.despine()
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
X = original_df.drop('Class', axis=1)
y = original_df['Class']
sampler = RandomUnderSampler(sampling_strategy='auto')
X_rs, y_rs = sampler.fit_sample(X, y)
print('Resampled dataset shape %s' % Counter(y_rs))
plot_graph_tsne(X_rs, y_rs)

df_under = pd.concat([X_rs, y_rs], axis=1)
df_under.head()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(50,30))      
sns.heatmap(df_under.corr(), annot=True, cmap = 'coolwarm', ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix of UnderSampled Data", fontsize=14)

       
sns.heatmap(original_df.corr(), cmap = 'coolwarm', ax=ax2)
ax2.set_title("Imbalanced Correlation Matrix of Original Data", fontsize=14)
f, axes = plt.subplots(ncols=7, figsize=(60,8))
sns.boxplot(x="Class", y="V3", data=df_under,  ax=axes[0])
axes[0].set_title('V3 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V9", data=df_under,  ax=axes[1])
axes[1].set_title('V9 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V10", data=df_under,  ax=axes[2])
axes[2].set_title('V10 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V12", data=df_under,  ax=axes[3])
axes[3].set_title('V12 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=df_under,  ax=axes[4])
axes[4].set_title('V14 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V16", data=df_under,  ax=axes[5])
axes[5].set_title('V16 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V17", data=df_under,  ax=axes[6])
axes[6].set_title('V17 vs Class Negative Correlation')

plt.show()
f, axes = plt.subplots(ncols=4, figsize=(20,4))
sns.boxplot(x="Class", y="V2", data=df_under,  ax=axes[0])
axes[0].set_title('V2 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=df_under,  ax=axes[1])
axes[1].set_title('V4 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V11", data=df_under,  ax=axes[2])
axes[2].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V19", data=df_under,  ax=axes[3])
axes[3].set_title('V19 vs Class Positive Correlation')

plt.show()
from scipy.stats import norm
fig, ax = plt.subplots(1, 4, figsize=(50,8))

V2 = df_under['V2'].values
V4 = df_under['V4'].values
V11 = df_under['V11'].values
V19 = df_under['V19'].values

sns.distplot(V2, ax=ax[0], fit=norm, color='b')
ax[0].set_title('Distribution of V2', fontsize=8)
ax[0].set_xlim([min(V2), max(V2)])

sns.distplot(V4, ax=ax[1], fit=norm, color='g')
ax[1].set_title('Distribution of V4', fontsize=8)
ax[1].set_xlim([min(V4), max(V4)])

sns.distplot(V11, ax=ax[2], fit=norm, color='g')
ax[2].set_title('Distribution of V11', fontsize=8)
ax[2].set_xlim([min(V11), max(V11)])

sns.distplot(V19, ax=ax[3], fit=norm, color='g')
ax[3].set_title('Distribution of V19', fontsize=8)
ax[3].set_xlim([min(V19), max(V19)])

plt.show()
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df_under[["V3","V9","V10","V12","V14","V2","V4","V11","V19"]]))
df_under_filtered = df_under[(z < 3).all(axis=1)]
print("Before outlier removal",df_under.shape)
print("Remaining after outlier removal",df_under_filtered.shape)
sns.set(style="darkgrid")
ax = sns.countplot(x="Class", data=df_under_filtered)
Q1 = df_under[["V3","V9","V10","V12","V14","V2","V4","V11","V19"]].quantile(0.25)
Q3 = df_under[["V3","V9","V10","V12","V14","V2","V4","V11","V19"]].quantile(0.75)
IQR = Q3 - Q1
df_under_out = df_under[~((df_under < (Q1 - 1.5 * IQR)) |(df_under > (Q3 + 1.5 * IQR))).any(axis=1)]
print("Remaining after outlier removal",df_under_out.shape)
sns.set(style="darkgrid")
ax = sns.countplot(x="Class", data=df_under_out)
f, axes = plt.subplots(ncols=9, figsize=(60,8))
sns.boxplot(x="Class", y="V3", data=df_under_filtered,  ax=axes[0])
axes[0].set_title('V3 Reduced Outlier')
axes[0].annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

sns.boxplot(x="Class", y="V9", data=df_under_filtered,  ax=axes[1])
axes[1].set_title('V9 Reduced Outlier')

sns.boxplot(x="Class", y="V10", data=df_under_filtered,  ax=axes[2])
axes[2].set_title('V10 Reduced Outlier')

sns.boxplot(x="Class", y="V12", data=df_under_filtered,  ax=axes[3])
axes[3].set_title('V12 Reduced Outlier')

sns.boxplot(x="Class", y="V14", data=df_under_filtered,  ax=axes[4])
axes[4].set_title('V14 Reduced Outlier')

sns.boxplot(x="Class", y="V2", data=df_under_filtered,  ax=axes[5])
axes[5].set_title('V2 Reduced Outlier')

sns.boxplot(x="Class", y="V4", data=df_under_filtered,  ax=axes[6])
axes[6].set_title('V4 Reduced Outlier')

sns.boxplot(x="Class", y="V11", data=df_under_filtered,  ax=axes[7])
axes[7].set_title('V11 Reduced Outlier')

sns.boxplot(x="Class", y="V19", data=df_under_filtered,  ax=axes[8])
axes[8].set_title('V19 Reduced Outlier')


plt.show()
X = df_under_filtered.drop('Class', axis=1)
y = df_under_filtered['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)
X_test.shape
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
import xgboost
from sklearn import svm, tree
from sklearn import metrics

classifiers = []
nb_model = GaussianNB()
classifiers.append(("Gausian Naive Bayes Classifier",nb_model))
lr_model= LogisticRegression()
classifiers.append(("Logistic Regression Classifier",lr_model))
# sv_model = svm.SVC()
# classifiers.append(sv_model)
dt_model = tree.DecisionTreeClassifier()
classifiers.append(("Decision Tree Classifier",dt_model))
rf_model = RandomForestClassifier()
classifiers.append(("Random Forest Classifier",rf_model))
xgb_model = xgboost.XGBClassifier()
classifiers.append(("XG Boost Classifier",xgb_model))
lda_model = LinearDiscriminantAnalysis()
classifiers.append(("Linear Discriminant Analysis", lda_model))
gp_model =  GaussianProcessClassifier()
classifiers.append(("Gaussian Process Classifier", gp_model))
ab_model =  AdaBoostClassifier()
classifiers.append(("AdaBoost Classifier", ab_model))

cv_scores = []
names = []
for name, clf in classifiers:
    print(name)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
    y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
    print("Model Score : ",clf.score(X_test, y_pred))
    print("Number of mislabeled points from %d points : %d"% (X_test.shape[0],(y_test!= y_pred).sum()))
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    cv_scores.append(scores)
    names.append(name)
    print("Cross validation scores : ",scores.mean())
    confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
    print("Confusion Matrix \n",confusion_matrix)
    classification_report = metrics.classification_report(y_test,y_pred)
    print("Classification Report \n",classification_report)
