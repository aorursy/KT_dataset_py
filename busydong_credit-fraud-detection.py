# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.preprocessing import StandardScaler, RobustScaler


# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('../input/creditcard.csv')
df.head()
df.describe()
# Good No Null Values!
df.isnull().sum().max()
df.columns
# The classes are heavily skewed we need to solve this issue later.
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='g')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()
fig, ax = plt.subplots(2, 1, figsize=(16,4),sharex=True)

ax[0].scatter(df[df['Class']==1]['Time'],df[df['Class']==1]['Amount'], alpha=0.4, c='r')
ax[0].set_title('Fraud')

ax[1].scatter(df[df['Class']==0]['Time'],df[df['Class']==0]['Amount'], alpha=0.4)
ax[1].set_title('non-Fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.show()
# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df_scaled = df.copy()

df_scaled['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df_scaled['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df_scaled.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = df_scaled['scaled_amount']
scaled_time = df_scaled['scaled_time']

df_scaled.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df_scaled.insert(0, 'scaled_amount', scaled_amount)
df_scaled.insert(1, 'scaled_time', scaled_time)

df_scaled.head()
df_scaled.describe()
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

# Lets shuffle the data before creating the subsamples

df_scaled_shuffled = df_scaled.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df_scaled.loc[df_scaled['Class'] == 1]

non_fraud_df = df_scaled_shuffled.loc[df_scaled_shuffled['Class'] == 0][:fraud_df['Class'].count()]

equal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
equal_distributed_df = equal_distributed_df.sample(frac=1, random_state=42)

equal_distributed_df.head()
# equal_distributed_df['Class'].value_counts()
sns.countplot('Class', data=equal_distributed_df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()
# Make sure we use the subsample in our correlation

# f, ax2 = plt.subplots(1, 1, figsize=(12,10))
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,18))

# Entire DataFrame
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)


sub_sample_corr = equal_distributed_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()
f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Class", y="V17", data=equal_distributed_df, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=equal_distributed_df, ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V12", data=equal_distributed_df, ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V10", data=equal_distributed_df, ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

plt.show()
f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="Class", y="V11", data=equal_distributed_df, ax=axes[0])
axes[0].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=equal_distributed_df, ax=axes[1])
axes[1].set_title('V4 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V2", data=equal_distributed_df, ax=axes[2])
axes[2].set_title('V2 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V19", data=equal_distributed_df, ax=axes[3])
axes[3].set_title('V19 vs Class Positive Correlation')

plt.show()
# New_df is from the random undersample data (fewer instances)
X = equal_distributed_df.drop('Class', axis=1)
y = equal_distributed_df['Class']


# T-SNE Implementation
t0 = time.time()
X_reduced_tsne = TSNE(n_components=3, random_state=42).fit_transform(X.values)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))

# PCA Implementation
t0 = time.time()
X_reduced_pca = PCA(n_components=3, random_state=42).fit_transform(X.values)
t1 = time.time()
print("PCA took {:.2} s".format(t1 - t0))

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=3, algorithm='randomized', random_state=42).fit_transform(X.values)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1 - t0))
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)
fig = plt.figure(figsize=(24,6))
ax = fig.add_subplot(1,3,1, projection='3d')
# ax.set_zlabel("x_composite_3")
# ax.set_zlabel("x_composite_3")

blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1],X_reduced_tsne[:,2], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1],X_reduced_tsne[:,2], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax.set_title('t-SNE', fontsize=14)

# ax.grid(True)

ax.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax = fig.add_subplot(1,3,2, projection='3d')
ax.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1],X_reduced_pca[:,2], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1],X_reduced_pca[:,2], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax.set_title('PCA', fontsize=14)

# ax2.grid(True)

ax.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax = fig.add_subplot(1,3,3, projection='3d')
ax.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1],X_reduced_svd[:,2], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1],X_reduced_svd[:,2], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax.set_title('Truncated SVD', fontsize=14)

# ax.grid(True)

ax.legend(handles=[blue_patch, red_patch])

plt.show()
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
outlier_fraction = sum(y==1)/sum(y==0)
state = np.random.RandomState(42)


classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1, random_state=state)
   
}
X = df_scaled.drop('Class', axis=1)
y = df_scaled['Class']

# n_outliers = len(Fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        y_pred = clf.fit_predict(X)
    else:    
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != y).sum()
    # Run Classification Metrics
    print("{} total error: {}".format(clf_name,n_errors))
    print("AUCROC core :")
    print(roc_auc_score(y,y_pred))
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))
# Our data is already scaled we should split our training and test sets
X = df_scaled.drop('Class', axis=1)
y = df_scaled['Class']


# This is explicitly used for undersampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

print('-' * 50)
print('Label Distributions for Original:')
print(y.value_counts(normalize = True))
print('-' * 50)
print('Label Distributions for Train:')
print(y_train.value_counts(normalize = True))
print('-' * 50)
print('Label Distributions for Test:')
print(y_test.value_counts(normalize = True))
# Undersample traini data, keeping test data unchanged
print("Before undersampling")
print(y_train.value_counts())

# Using near miss algorithm
X_nearmiss, y_nearmiss = NearMiss().fit_sample(X_train.values, y_train.values)

print("Near Miss undersampling count: ")
print(Counter(y_nearmiss))

# Using random under sampling
minorityN = (y_train==1).sum()

majority_indices = X_train[y_train==0].index
random_indices = np.random.choice(majority_indices, minorityN, replace=False) # use the low-frequency group count to randomly sample from high-frequency group
X_train_subsample = pd.concat([X_train.loc[random_indices],X_train[y_train==1]]).values
y_train_subsample =  pd.concat([y_train.loc[random_indices],y_train[y_train==1]]).values

print("Random Undersampling count: ")
print(Counter(y_train_subsample))

# Using SMOTE upsampling
X_SMOTE, y_SMOTE = SMOTE(sampling_strategy='minority').fit_sample(X_train.values, y_train.values)
print("SMOTE upsampling count: ")
print(Counter(y_SMOTE))
def model_and_performance(original_Xtrain, original_ytrain, original_Xtest, original_ytest, cv_model):

    
    model = cv_model.fit(original_Xtrain, original_ytrain)
    best_est = cv_model.best_estimator_
    prediction = best_est.predict(original_Xtrain)
    prediction_test = best_est.predict(original_Xtest)
    
    accuracy=accuracy_score(original_ytrain,prediction)
    precision=precision_score(original_ytrain, prediction)
    recall=recall_score(original_ytrain, prediction)
    f1=f1_score(original_ytrain, prediction)
    auc=roc_auc_score(original_ytrain, prediction)
        
    accuracy_test=accuracy_score(original_ytest,prediction_test)
    precision_test=precision_score(original_ytest, prediction_test)
    recall_test=recall_score(original_ytest, prediction_test)
    f1_test=f1_score(original_ytest, prediction_test)
    auc_test=roc_auc_score(original_ytest, prediction_test)
        
    print("Train accuracy:",accuracy,"  test accuracy: ", accuracy_test)
    print("Train precision:",precision,"  test precision: ", precision_test)
    print("Train recall:",recall,"  test recall: ", recall_test)
    print("Train f1:",f1,"  test f1: ", f1_test)
    print("Train auc:",auc,"  test auc: ", auc_test)

def model_and_performance_5fold(original_Xtrain, original_ytrain, original_Xtest, original_ytest, cv_model):
    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    accuracy_lst = []
    precision_lst = []
    recall_lst = []
    f1_lst = []
    auc_lst = []
    
    accuracy_lst_test = []
    precision_lst_test = []
    recall_lst_test = []
    f1_lst_test = []
    auc_lst_test = []
    for train, test in sss.split(original_Xtrain, original_ytrain):
        model = cv_model.fit(original_Xtrain[train], original_ytrain[train])
        best_est = cv_model.best_estimator_
        prediction = best_est.predict(original_Xtrain[test])
        prediction_test = best_est.predict(original_Xtest)
    
        accuracy_lst.append(accuracy_score(original_ytrain[test],prediction))
        precision_lst.append(precision_score(original_ytrain[test], prediction))
        recall_lst.append(recall_score(original_ytrain[test], prediction))
        f1_lst.append(f1_score(original_ytrain[test], prediction))
        auc_lst.append(roc_auc_score(original_ytrain[test], prediction))
        
        accuracy_lst_test.append(accuracy_score(original_ytest,prediction_test))
        precision_lst_test.append(precision_score(original_ytest, prediction_test))
        recall_lst_test.append(recall_score(original_ytest, prediction_test))
        f1_lst_test.append(f1_score(original_ytest, prediction_test))
        auc_lst_test.append(roc_auc_score(original_ytest, prediction_test))
        
    print("Train validation average accuracy:",np.mean(accuracy_lst),"  test accuracy: ", np.mean(accuracy_lst_test))
    print("Train validation average precision:",np.mean(precision_lst),"  test precision: ", np.mean(precision_lst_test))
    print("Train validation average recall:",np.mean(recall_lst),"  test recall: ", np.mean(recall_lst_test))
    print("Train validation average f1:",np.mean(f1_lst),"  test f1: ", np.mean(f1_lst_test))
    print("Train validation average auc:",np.mean(auc_lst),"  test auc: ", np.mean(auc_lst_test))
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)
print("-"*20 + "Near Miss Algorithm + Logistic" + "-"*20 )
t0 = time.time()
model_and_performance_5fold(X_nearmiss,y_nearmiss,X_test, y_test,rand_log_reg)
print("model took {:.2} s".format(t1 - t0))

print("-"*20 + "Random Undersampling + Logistic" + "-"*20 )
t0 = time.time()
model_and_performance_5fold(X_train_subsample,y_train_subsample,X_test, y_test,rand_log_reg)
t1 = time.time()
print("model took {:.2} s".format(t1 - t0))


print("-"*20 + "SMOTE upsampling + Logistic" + "-"*20 )
t0 = time.time()
model_and_performance_5fold(X_SMOTE,y_SMOTE,X_test, y_test,rand_log_reg)
t1 = time.time()
print("model took {:.2} s".format(t1 - t0))


log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
rand_log_reg = RandomizedSearchCV(LogisticRegression(class_weight='balanced'), log_reg_params, n_iter=4)

print("-"*20 + "Original Train + Logistic (with balanced class weight)" + "-"*20 )
t0 = time.time()
model_and_performance_5fold(X_train.values,y_train.values,X_test, y_test,rand_log_reg)
t1 = time.time()
print("model took {:.2} s".format(t1 - t0))
# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)
print("-"*20 + "Random downsampling + Logistic" + "-"*20 )
t0 = time.time()
model_and_performance_5fold(X_train_subsample,y_train_subsample,X_test, y_test,rand_log_reg)
t1 = time.time()
print("model took {:.2} s".format(t1 - t0))

# KNears
knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree']}
rand_knears = RandomizedSearchCV(KNeighborsClassifier(), knears_params, n_iter=4)
print("-"*20 + "Random downsampling + KNearest" + "-"*20 )
t0 = time.time()
model_and_performance_5fold(X_train_subsample,y_train_subsample,X_test, y_test,rand_knears)
t1 = time.time()
print("model took {:.2} s".format(t1 - t0))

# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
rand_svc = RandomizedSearchCV(SVC(), svc_params, n_iter=4)
print("-"*20 + "Random downsampling + Support Vector Classifier" + "-"*20 )
t0 = time.time()
model_and_performance_5fold(X_train_subsample,y_train_subsample,X_test, y_test,rand_svc)
t1 = time.time()
print("model took {:.2} s".format(t1 - t0))

# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
rand_tree = RandomizedSearchCV(DecisionTreeClassifier(), tree_params, n_iter=4)
print("-"*20 + "Random downsampling + DecisionTreeClassifier" + "-"*20 )
t0 = time.time()
model_and_performance_5fold(X_train_subsample,y_train_subsample,X_test, y_test,rand_tree)
t1 = time.time()
print("model took {:.2} s".format(t1 - t0))

# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)
print("-"*20 + "SMOTE upsampling + Logistic" + "-"*20 )
t0 = time.time()
model_and_performance(X_SMOTE,y_SMOTE,X_test, y_test,rand_log_reg)
t1 = time.time()
print("model took {:.2} s".format(t1 - t0))


# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
rand_tree = RandomizedSearchCV(DecisionTreeClassifier(), tree_params, n_iter=4)
print("-"*20 + "SMOTE upsampling + DecisionTreeClassifier" + "-"*20 )
t0 = time.time()
model_and_performance(X_SMOTE,y_SMOTE,X_test, y_test,rand_tree)
t1 = time.time()
print("model took {:.2} s".format(t1 - t0))

# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV


# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_SMOTE, y_SMOTE)
# We automatically get the logistic regression with the best parameters.
log_reg = grid_log_reg.best_estimator_
print(log_reg)

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(n_neighbors=2), knears_params)
grid_knears.fit(X_train_subsample, y_train_subsample)
# KNears best estimator
knears_neighbors = grid_knears.best_estimator_
print(knears_neighbors)


# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train_subsample, y_train_subsample)

# SVC best estimator
svc = grid_svc.best_estimator_
print(svc)


# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_SMOTE, y_SMOTE)

# tree best estimator
tree_clf = grid_tree.best_estimator_
print(tree_clf)
# Overfitting Case

t0 = time.time()
log_reg_score = cross_val_score(log_reg, X_SMOTE, y_SMOTE, cv=5, scoring='roc_auc')
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')
t1 = time.time()
print("Logistic Regression took {:.2} s".format(t1 - t0))

t0 = time.time()
knears_score = cross_val_score(knears_neighbors, X_train_subsample, y_train_subsample, cv=5, scoring='roc_auc')
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')
t1 = time.time()
print("Knears Neighbors took {:.2} s".format(t1 - t0))

t0 = time.time()
svc_score = cross_val_score(svc, X_train_subsample, y_train_subsample, cv=5, scoring='roc_auc')
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')
t1 = time.time()
print("Support Vector Classifier took {:.2} s".format(t1 - t0))

t0 = time.time()
tree_score = cross_val_score(tree_clf, X_SMOTE, y_SMOTE, cv=5, scoring='roc_auc')
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
t1 = time.time()
print("DecisionTree Classifier took {:.2} s".format(t1 - t0))
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
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
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
plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, X_train_subsample, y_train_subsample, (0.87, 1.01), cv=cv, n_jobs=4)
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
# Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = log_reg.predict(X_SMOTE)
log_reg_pred_test = log_reg.predict(X_test)


knears_pred = knears_neighbors.predict(X_train_subsample)
knears_pred_test = knears_neighbors.predict(X_test)


svc_pred = svc.predict(X_train_subsample)
svc_pred_test = svc.predict(X_test)


tree_pred = tree_clf.predict(X_SMOTE)
tree_pred_test = tree_clf.predict(X_test)



print('Logistic Regression:')
print("train: ",roc_auc_score(y_SMOTE, log_reg_pred))
print("test: ",roc_auc_score(y_test, log_reg_pred_test))
print(confusion_matrix(y_test, log_reg_pred_test))
print(classification_report(y_test, log_reg_pred_test))
print('KNears Neighbors: ')
print("train: ",roc_auc_score(y_train_subsample, knears_pred))
print("test: ",roc_auc_score(y_test, knears_pred_test))
print(confusion_matrix(y_test, knears_pred_test))
print(classification_report(y_test, knears_pred_test))
print('Support Vector Classifier: ')
print("train: ",roc_auc_score(y_train_subsample, svc_pred))
print("test: ",roc_auc_score(y_test, svc_pred_test))
print(confusion_matrix(y_test, svc_pred_test))
print(classification_report(y_test, svc_pred_test))
print('Decision Tree Classifier: ')
print("train: ",roc_auc_score(y_SMOTE, tree_pred))
print("test: ",roc_auc_score(y_test, tree_pred_test))
print(confusion_matrix(y_test, tree_pred_test))
print(classification_report(y_test, tree_pred_test))
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

n_inputs = X_train.shape[1]

undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
undersample_model.summary()
undersample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
undersample_model.fit(X_train_subsample, y_train_subsample, validation_split=0.2, batch_size=25, \
                      epochs=20, shuffle=True, verbose=1, callbacks=[callback])
undersample_predictions = undersample_model.predict(X_test, batch_size=200, verbose=0)
undersample_fraud_predictions = undersample_model.predict_classes(X_test, batch_size=200, verbose=0)
print(confusion_matrix(y_test, undersample_fraud_predictions))
print(classification_report(y_test, undersample_fraud_predictions))


n_inputs = X_SMOTE.shape[1]
n_inputs = X_SMOTE.shape[1]

oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
oversample_model.fit(X_SMOTE, y_SMOTE, validation_split=0.2, batch_size=300, epochs=20, \
                     shuffle=True, verbose=1, callbacks=[callback])

oversample_predictions = oversample_model.predict(X_test, batch_size=200, verbose=0)
oversample_fraud_predictions = oversample_model.predict_classes(X_test, batch_size=200, verbose=0)
print('Logistic Regression:')
print("test: ",roc_auc_score(y_test, log_reg_pred_test))
print(confusion_matrix(y_test, log_reg_pred_test))
print(classification_report(y_test, log_reg_pred_test))
print('KNears Neighbors: ')
print("test: ",roc_auc_score(y_test, knears_pred_test))
print(confusion_matrix(y_test, knears_pred_test))
print(classification_report(y_test, knears_pred_test))
print('Support Vector Classifier: ')
print("test: ",roc_auc_score(y_test, svc_pred_test))
print(confusion_matrix(y_test, svc_pred_test))
print(classification_report(y_test, svc_pred_test))
print('Decision Tree Classifier: ')
print("test: ",roc_auc_score(y_test, tree_pred_test))
print(confusion_matrix(y_test, tree_pred_test))
print(classification_report(y_test, tree_pred_test))
print("-"*100)
print("ML model:")
print("Undersampling performance:")
print("aucroc: ", roc_auc_score(y_test, undersample_fraud_predictions))
print(confusion_matrix(y_test, undersample_fraud_predictions))
print(classification_report(y_test, undersample_fraud_predictions))
print("\nOversampling performance:")
print("aucroc: ", roc_auc_score(y_test, oversample_fraud_predictions))
print(confusion_matrix(y_test, oversample_fraud_predictions))
print(classification_report(y_test, oversample_fraud_predictions))
