import numpy as np
import pandas as pd
import time
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

# Visualization Libraries
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import plotly.graph_objs as go
from plotly.offline import iplot

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import collections

# Other Important Libraries
from sklearn.model_selection import train_test_split  # used for splitting the dataset
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")

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
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df.shape
df.info()
df.describe()
# Let's see the percentage of data having fraud and non-fraud transaction
print("Frauds", round(df['Class'].value_counts()[1]/len(df)*100, 3), '% of the dataset')
print("No Frauds", round(df['Class'].value_counts()[0]/len(df)*100, 3), '% of the dataset')
labels = df["Class"].value_counts()[:10].index
values = df["Class"].value_counts()[:10].values

colors=['#2678bf', '#98adbf']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent",
                            insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()
fig, ax = plt.subplots(1,2, figsize=(8,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title("Distribution of Tranaction Amount")
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='g')
ax[1].set_title('Distribution of Transaction Time')
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()
# We know that most of the data has already been scaled so we need to scale the columns 
# that are not scaled i.e. Amount and Time

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

# Since we have added a new columns in place of Amount and and time so we need to drop the
# existing columns

df = df.drop(['Time', 'Amount'], axis = 1)
df.head()
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis = 1, inplace = True)
df.insert(0, 'scaled_amount', scaled_amount) # now scaled_amount is at the 1st position of columns list
df.insert(1, 'scaled_time', scaled_time) # now scaled_time is at the 2nd position of the columns lis

df.head()
# We already know from the above that our datasets contains
# Frauds: 0.173 % of the dataset
# No Frauds: 99.827 % of the dataset

X = df.drop('Class', axis = 1)
y = df['Class']

SS = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in SS.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
    
# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed

train_uniques_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-'*100)

print('Label Distribution : \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))
# Since we need equal number of both fraud and non-fraud data so we need to first shuffle 
# to take equal amount of non-fraud datas

df = df.sample(frac=1)

# amount of fraud classes 492 rows, so there must be an equal amount of non-fraud classes too

fraud_df = df.loc[df['Class']==1]
non_fraud_df = df.loc[df['Class']==0][:492]  # its not necessary to pick the 1st 492 to non-fraud classes we can take it from anywhere

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# now shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()
new_df.shape
print('Classe Distribution in the Dataset Subsample')
print(new_df['Class'].value_counts()/len(new_df))


labels = new_df["Class"].value_counts()[:10].index
values = new_df["Class"].value_counts()[:10].values

colors=['#2678bf', '#98adbf']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent",
                            insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()
f, (ax1, ax2) = plt.subplots(2,1, figsize=(28,24))

# for the original dataset
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Original Dataset Correlation Matrix \n (This won't be used for reference)")

sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title("Subsample Correlation Matrix \n (This will be use for reference)", fontsize=14)
plt.show()
from scipy.stats import norm
f, (ax1,ax2,ax3, ax4) = plt.subplots(1,4, figsize=(20,6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class']==1].values # trying to get to know how much V14 is affecting the chances of being Fraud
sns.distplot(v14_fraud_dist, ax=ax1, fit=norm, color='r')
ax1.set_title('V14 Distribution \n (Fraud Transaction)', fontsize=14)

v12_fraud_dist = new_df['V12'].loc[new_df['Class']==1].values # trying to get to know how much V12 is affecting the chances of being Fraud
sns.distplot(v12_fraud_dist, ax=ax2, fit=norm, color='g')
ax2.set_title('V12 Distribution \n (Fraud Transaction)')

v10_fraud_dist = new_df['V10'].loc[new_df['Class']==1].values # trying to get to know how much V10 is affecting the chances of being Fraud
sns.distplot(v10_fraud_dist, ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V10 Distribution \n (Fraud Transaction)')

v17_fraud_dist = new_df['V17'].loc[new_df['Class']==1].values # trying to get to know how much V17 is affecting the chances of being Fraud
sns.distplot(v10_fraud_dist, ax=ax4, fit=norm)
ax4.set_title('V17 Distribution \n (Fraud Transaction)')

plt.show()
from scipy.stats import norm
f, (ax1,ax2,ax3, ax4) = plt.subplots(1,4, figsize=(20,6))

v2_fraud_dist = new_df['V2'].loc[new_df['Class']==1].values # trying to get to know how much V2 is affecting the chances of being Fraud
sns.distplot(v2_fraud_dist, ax=ax1, fit=norm, color='r')
ax1.set_title('V2 Distribution \n (Fraud Transaction)', fontsize=14)

v4_fraud_dist = new_df['V4'].loc[new_df['Class']==1].values # trying to get to know how much V4 is affecting the chances of being Fraud
sns.distplot(v4_fraud_dist, ax=ax2, fit=norm, color='g')
ax2.set_title('V4 Distribution \n (Fraud Transaction)')

v11_fraud_dist = new_df['V11'].loc[new_df['Class']==1].values # trying to get to know how much V11 is affecting the chances of being Fraud
sns.distplot(v11_fraud_dist, ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V11 Distribution \n (Fraud Transaction)')

v19_fraud_dist = new_df['V19'].loc[new_df['Class']==1].values # trying to get to know how much V19 is affecting the chances of being Fraud
sns.distplot(v19_fraud_dist, ax=ax4, fit=norm)
ax4.set_title('V19 Distribution \n (Fraud Transaction)')

plt.show()
# Rmoving the V14 Outliers

v14_fraud = new_df['V14'].loc[new_df['Class']==1].values
q25 = np.percentile(v14_fraud, 25)  # Calculating the 25 percentile
q75 = np.percentile(v14_fraud, 75)  # Calculating the 75 percentile

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75-q25
print('iqr: {}'.format(v14_iqr))

v14_cut_off = v14_iqr*1.5
v14_lower = q25-v14_cut_off # setting the lower limit
v14_upper = q75+v14_cut_off # setting the upper limit
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('Features V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V14 outliers:{}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)



# Removing the V12 Outliers

v12_fraud = new_df['V12'].loc[new_df['Class']==1].values
q25 = np.percentile(v12_fraud, 25)  # Calculating the 25 percentile
q75 = np.percentile(v12_fraud, 75)  # Calculating the 75 percentile

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v12_iqr = q75-q25
print('iqr: {}'.format(v12_iqr))

v12_cut_off = v12_iqr*1.5
v12_lower = q25-v12_cut_off # setting the lower limit
v12_upper = q75+v12_cut_off # setting the upper limit
print('Cut Off: {}'.format(v12_cut_off))
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print('Features V12 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V12 outliers:{}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)



# Removing the V10 Outliers

v10_fraud = new_df['V10'].loc[new_df['Class']==1].values
q25 = np.percentile(v10_fraud, 25)  # Calculating the 25 percentile
q75 = np.percentile(v10_fraud, 75)  # Calculating the 75 percentile

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v10_iqr = q75-q25
print('iqr: {}'.format(v10_iqr))

v10_cut_off = v10_iqr*1.5
v10_lower = q25-v10_cut_off # setting the lower limit
v10_upper = q75+v10_cut_off # setting the upper limit
print('Cut Off: {}'.format(v10_cut_off))
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print('Features V10 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V10 outliers:{}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)




f, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,6))

colors=['#2678bf', '#98adbf']

# Feature V14
sns.boxplot(x="Class", y="V14", data=new_df, ax=ax1,palette = colors)
ax1.set_title("Feature V14 \n Reduction of Outliers")
ax1.annotate("Fewer extreme \n Outliers", xy=(0.98, -17.5), xytext=(0,-12), arrowprops=dict(facecolor='black'))

# Feature V12
sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2, palette=colors)
ax2.set_title("Feature V12 \n Reduction of Outliers")
ax2.annotate("Fewer extreme \n Outliers", xy=(0.98, -17.3), xytext=(0,-12), arrowprops=dict(facecolor='black'))

# Feature V10
sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3, palette=colors)
ax2.set_title("Feature V10 \n Reduction of Outliers")
ax2.annotate("Fewer extreme \n Outliers", xy=(0.98, -17.3), xytext=(0,-12), arrowprops=dict(facecolor='black'))


plt.show()
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# Let's check how much time the given dimensionlity reduction algorithms take

# Implementing T-SNE 
t0 = time.time()  # Initial Time
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()  # Final Time
time_diff1 = t1-t0
print("T-SNE took {:.2} s".format(time_diff1))

# PCA Implementation
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
time_diff2 = t1-t0
print("PCA took {:.2} s".format(time_diff2))

# TruncatedSVD Implementation
t0 = time.time()
X_reduced_pca = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
t1 = time.time()
time_diff2 = t1-t0
print("Truncated SVD took {:.2} s".format(time_diff2))



# Undersampling before cross validation (prone to overfit)
X = new_df.drop('Class', axis=1)
y = new_df['Class']
# Since we have already scaled our data we shpuld split our training and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
# Implementing the Classifiers

classifiers = {
    "LogisticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}
# let's try to get the score using the croos-validation

from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=7)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 3) * 100, "% accuracy score")
# Let's use GridSearchCV to find the best parameters for each of the classifiers described above
from sklearn.model_selection import GridSearchCV

# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)

# The below line will help us to find the best parameters automatically
log_reg = grid_log_reg.best_estimator_

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)

# The below line will help us to find the best parameters automatically
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)

# The below line will help us to find the best parameters automatically
svc = grid_svc.best_estimator_


# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)

# The below line will help us to find the best parameters automatically
tree_clf = grid_tree.best_estimator_
# Now calculating the accuracy in the Overfitting cases


log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=7)
print("Logistic Regression Cross Validation Score: ", round(log_reg_score.mean()*100,3).astype(str)+'%')

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=7)
print("Knears Neighbors Cross Validation Score:", round(knears_score.mean()*100,3).astype(str)+'%')

svc_score = cross_val_score(svc, X_train, y_train, cv=7)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean()*100, 3).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=7)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean()*100, 3).astype(str) + '%')
# We will undersample during cross validating


undersample_X = df.drop('Class', axis=1)
undersample_y = df['Class']

for train_index, test_index in SS.split(undersample_X, undersample_y):
    print("Train:", train_index, "Test:", test_index)
    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]
    
undersample_Xtrain = undersample_Xtrain.values
undersample_Xtest = undersample_Xtest.values
undersample_ytrain = undersample_ytrain.values
undersample_ytest = undersample_ytest.values 

undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []

# Implementing NearMiss Technique 
# Distribution of NearMiss 

X_nearmiss, y_nearmiss = NearMiss().fit_sample(undersample_X.values, undersample_y.values)
print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))


# Cross Validating the right way

for train, test in SS.split(undersample_Xtrain, undersample_ytrain):
    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg) # SMOTE happens during Cross Validation not before..
    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])
    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])
    
    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))
    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))
    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))
# Let's Plot LogisticRegression Learning Curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    # First Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    
    # Second Estimator 
    train_sizes, train_scores, test_scores = learning_curve(
        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)
    ax2.set_xlabel('Training size (m)')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend(loc="best")
    
    # Third Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)
    ax3.set_xlabel('Training size (m)')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.legend(loc="best")
    
    # Fourth Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)
    ax4.set_xlabel('Training size (m)')
    ax4.set_ylabel('Score')
    ax4.grid(True)
    ax4.legend(loc="best")
    return plt
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict

# Creating a dataframe with all the scores and the classifier names

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=7, method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=7)

svc_pred = cross_val_predict(svc, X_train, y_train, cv=7, method="decision_function")

tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=7)
from sklearn.metrics import roc_auc_score

print("Logistic Regression: ", roc_auc_score(y_train, log_reg_pred))
print("KNears Neighbors: ", roc_auc_score(y_train, knears_pred))
print("Support Vector Classifier: ", roc_auc_score(y_train, svc_pred))
print("Decision Tree Classifier: ", roc_auc_score(y_train, tree_pred))
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)

def logistic_roc_curve(log_fpr, log_tpr):
    plt.figure(figsize=(12,8))
    plt.title('Logistic Regression ROC Curve', fontsize=16)
    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01,1,0,1])
    
    
logistic_roc_curve(log_fpr, log_tpr)
plt.show()
def knear_roc_curve(knear_fpr, knear_tpr):
    plt.figure(figsize=(12,8))
    plt.title('KNears Neighbors ROC Curve', fontsize=16)
    plt.plot(knear_fpr, knear_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01,1,0,1])
    
    
knear_roc_curve(knear_fpr, knear_tpr)
plt.show()
def svc_roc_curve(svc_fpr, svc_tpr):
    plt.figure(figsize=(12,8))
    plt.title('Support Vector Classifier ROC Curve', fontsize=16)
    plt.plot(svc_fpr, svc_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01,1,0,1])
    
    
svc_roc_curve(svc_fpr, svc_tpr)
plt.show()
def tree_roc_curve(tree_fpr, tree_tpr):
    plt.figure(figsize=(12,8))
    plt.title('Decision Tree ROC Curve', fontsize=16)
    plt.plot(tree_fpr, tree_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01,1,0,1])
    
    
tree_roc_curve(tree_fpr, tree_tpr)
plt.show()
from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
y_pred = log_reg.predict(X_train)

# Overfitting Case
print('---' * 45)
print('Overfitting: \n')
print('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))
print('Precision Score: {:.2f}'.format(precision_score(y_train, y_pred)))
print('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))
print('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))
print('---' * 45)

# How it should look like
print('---' * 45)
print('How it should be:\n')
print("Accuracy Score: {:.2f}".format(np.mean(undersample_accuracy)))
print("Precision Score: {:.2f}".format(np.mean(undersample_precision)))
print("Recall Score: {:.2f}".format(np.mean(undersample_recall)))
print("F1 Score: {:.2f}".format(np.mean(undersample_f1)))
print('---' * 45)

undersample_y_score = log_reg.decision_function(original_Xtest)
from sklearn.metrics import average_precision_score

undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      undersample_average_precision))
