import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



#classifier libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import collections



#other libraries

from imblearn.datasets import fetch_datasets

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/creditcard.csv")

df.head(5)
df.describe()
#means no null values, so its good

df.isnull().sum().max()
df.columns
#checking how many classes are 0 and how many 1

print("fraud class is ", round(df['Class'].value_counts()[1]/len(df) * 100, 2), "% of the dataset")

print("non fraud class is ", round(df['Class'].value_counts()[00]/len(df) * 100, 2), "% of the dataset")
colors = ["#0101DF","#DF0101"]

sns.countplot("Class", data=df, palette=colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
fig, ax = plt.subplots(1, 2, figsize=(18,4))



amount_val = df['Amount'].values

time_val = df['Time'].values



sns.distplot(amount_val, ax=ax[0], color='r')

ax[0].set_title('Distribution of Transaction Amount', fontsize=14)

ax[0].set_xlim([min(amount_val), max(amount_val)])



sns.distplot(time_val, ax=ax[1], color='b')

ax[1].set_title('Distribution of Transaction Time', fontsize=14)

ax[1].set_xlim([min(time_val), max(time_val)])

from sklearn.preprocessing import StandardScaler, RobustScaler



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



df["scaled_time"] = rob_scaler.fit_transform(df["Time"].values.reshape(-1,1))

df["scaled_amount"] = rob_scaler.fit_transform(df["Amount"].values.reshape(-1,1))



df.drop(["Time","Amount"], axis=1, inplace=True)
scaled_amount = df["scaled_amount"]

scaled_time = df["scaled_time"]



df.drop(["scaled_amount","scaled_time"], axis=1, inplace=True)



df.insert(0, "scaled_amount", scaled_amount)

df.insert(1, "scaled_time", scaled_time)



df.head()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')



X = df.drop("Class", axis=1)

y = df["Class"]



sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for train_index, test_index in sss.split(X,y):

    print("train", train_index, "test", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    

original_Xtrain = original_Xtrain.values

original_ytrain = original_ytrain.values

original_Xtest = original_Xtest.values

original_ytest = original_ytest.values



train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)

test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print('-' * 100)



print('Label Distributions: \n')

print(train_counts_label/ len(original_ytrain))

print(test_counts_label/ len(original_ytest))
df = df.sample(frac=1)



fraud_df = df.loc[df["Class"]==1]

non_fraud_df = df.loc[df["Class"]==0][:492]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()
print('Distribution of the Classes in the subsample dataset')

print(new_df['Class'].value_counts()/len(new_df))







sns.countplot('Class', data=new_df, palette=colors)

plt.title('Equally Distributed Classes', fontsize=14)

plt.show()
f, (ax1,ax2) = plt.subplots(2, 1, figsize=(18,12))



# entire dataframe

corr = df.corr()

sns.heatmap(corr, cmap="coolwarm_r", annot_kws={"size":20}, ax=ax1)

ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)



# subsample correlation

sub_sample_corr = new_df.corr()

sns.heatmap(sub_sample_corr, cmap="coolwarm_r", annot_kws={"size":20}, ax=ax2)

ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)



plt.show()
f, axes = plt.subplots(ncols=4, figsize=(16,4))



sns.boxplot(x="Class", y="V17", data=new_df, palette=colors, ax=axes[0])

axes[0].set_title("V17 vs class negative correlation")



sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[1])

axes[1].set_title("V14 vs class negative correlation")



sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[2])

axes[2].set_title("V12 vs class negative correlation")



sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[3])

axes[3].set_title("V10 vs class negative correlation")



plt.show()
f, axes = plt.subplots(ncols=4, figsize=(16,4))



# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)

sns.boxplot(x="Class", y="V11", data=new_df, palette=colors, ax=axes[0])

axes[0].set_title('V11 vs Class Positive Correlation')



sns.boxplot(x="Class", y="V4", data=new_df, palette=colors, ax=axes[1])

axes[1].set_title('V4 vs Class Positive Correlation')





sns.boxplot(x="Class", y="V2", data=new_df, palette=colors, ax=axes[2])

axes[2].set_title('V2 vs Class Positive Correlation')





sns.boxplot(x="Class", y="V19", data=new_df, palette=colors, ax=axes[3])

axes[3].set_title('V19 vs Class Positive Correlation')



plt.show()
from scipy.stats import norm



f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,4))



v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values

sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')

ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)



v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values

sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')

ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)





v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values

sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')

ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)



plt.show()
# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)

v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

v14_iqr = q75 - q25

print('iqr: {}'.format(v14_iqr))



v14_cut_off = v14_iqr * 1.5

v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off

print('Cut Off: {}'.format(v14_cut_off))

print('V14 Lower: {}'.format(v14_lower))

print('V14 Upper: {}'.format(v14_upper))



outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]

print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V10 outliers:{}'.format(outliers))



new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)

print('----' * 44)



# -----> V12 removing outliers from fraud transactions

v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)

v12_iqr = q75 - q25



v12_cut_off = v12_iqr * 1.5

v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off

print('V12 Lower: {}'.format(v12_lower))

print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]

print('V12 outliers: {}'.format(outliers))

print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))

print('----' * 44)



# Removing outliers V10 Feature

v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)

v10_iqr = q75 - q25



v10_cut_off = v10_iqr * 1.5

v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off

print('V10 Lower: {}'.format(v10_lower))

print('V10 Upper: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]

print('V10 outliers: {}'.format(outliers))

print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))
f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))



colors = ['#B3F9C5', '#f9c5b3']

# Boxplots with outliers removed

# Feature V14

sns.boxplot(x="Class", y="V14", data=new_df,ax=ax1, palette=colors)

ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)

ax1.annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)



# Feature 12

sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2, palette=colors)

ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)

ax2.annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)



# Feature V10

sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3, palette=colors)

ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)

ax3.annotate('Fewer extreme \n outliers', xy=(0.95, -16.5), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)





plt.show()
# New_df is from the random undersample data (fewer instances)

X = new_df.drop('Class', axis=1)

y = new_df['Class']





# T-SNE Implementation

t0 = time.time()

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)

t1 = time.time()

print("T-SNE took {:.2} s".format(t1 - t0))



# PCA Implementation

t0 = time.time()

X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)

t1 = time.time()

print("PCA took {:.2} s".format(t1 - t0))



# TruncatedSVD

t0 = time.time()

X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)

t1 = time.time()

print("Truncated SVD took {:.2} s".format(t1 - t0))
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))

# labels = ['No Fraud', 'Fraud']

f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)





blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')

red_patch = mpatches.Patch(color='#AF0000', label='Fraud')





# t-SNE scatter plot

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax1.set_title('t-SNE', fontsize=14)



ax1.grid(True)



ax1.legend(handles=[blue_patch, red_patch])



# PCA scatter plot

ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax2.set_title('PCA', fontsize=14)



ax2.grid(True)



ax2.legend(handles=[blue_patch, red_patch])



# TruncatedSVD scatter plot

ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax3.set_title('Truncated SVD', fontsize=14)



ax3.grid(True)



ax3.legend(handles=[blue_patch, red_patch])



plt.show()
# Undersampling before cross validating (prone to overfit)

X = new_df.drop('Class', axis=1)

y = new_df['Class']
# Our data is already scaled we should split our training and test sets

from sklearn.model_selection import train_test_split



# This is explicitly used for undersampling.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Turn the values into an array for feeding the classification algorithms.

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
# Let's implement simple classifiers



classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
# Wow our scores are getting even high scores even when applying cross validation.

from sklearn.model_selection import cross_val_score





for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_



knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(X_train, y_train)

# KNears best estimator

knears_neighbors = grid_knears.best_estimator_



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, y_train)



# SVC best estimator

svc = grid_svc.best_estimator_



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, y_train)



# tree best estimator

tree_clf = grid_tree.best_estimator_
# Overfitting Case



log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')





knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)

print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')



svc_score = cross_val_score(svc, X_train, y_train, cv=5)

print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')



tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)

print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
# We will undersample during cross validating

undersample_X = df.drop('Class', axis=1)

undersample_y = df['Class']



for train_index, test_index in sss.split(undersample_X, undersample_y):

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

# Distribution of NearMiss (Just to see how it distributes the labels we won't use these variables)

X_nearmiss, y_nearmiss = NearMiss().fit_sample(undersample_X.values, undersample_y.values)

print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))

# Cross Validating the right way



for train, test in sss.split(undersample_Xtrain, undersample_ytrain):

    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg) # SMOTE happens during Cross Validation not before..

    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])

    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])

    

    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))

    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))

    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))

    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))

    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))