import numpy as np
import pandas as pd

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, ShuffleSplit, learning_curve, cross_val_predict, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, StratifiedShuffleSplit, cross_val_score

# imblearn
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced

from collections import Counter
import collections
import os
import warnings
warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:,.4f}'.format

df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
def eda_categ_feat_desc_plot(series_categorical, title = ""):
    """Generate 2 plots: barplot with quantity and pieplot with percentage. 
       @series_categorical: categorical series
       @title: optional
    """
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    
    fig, ax = plt.subplots(figsize = (12,4), ncols=2, nrows=1) # figsize = (width, height)
    if(title != ""):
        fig.suptitle(title, fontsize=18)
        fig.subplots_adjust(top=0.8)

    s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])
    for index, row in val_concat.iterrows():
        s.text(row.name, row['quantity'], row['quantity'], color='black', ha="center")

    s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),
                             labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],
                             title="Percentage Plot")

    ax[1].set_ylabel('')
    ax[0].set_title('Quantity Plot')

    plt.show()
import itertools

# Create a confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print("Columns List:")
print(list(df.columns))
eda_categ_feat_desc_plot(df['Class'], 'Rows By Class: No Fraud = 0 | Fraud = 1')
df.describe().T
import missingno as msno
sns.heatmap(df.isnull(), cbar=False)
# Good No Null Values!
df.isnull().sum().max()
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()
f, (ax1, ax2) = plt.subplots(ncols=2,  figsize=(16, 5), sharex=False)

map_feat_ax = {'Amount': ax1, 'Time': ax2}

for key, value in map_feat_ax.items():
    sns.boxplot(x=df[key], ax=value)
    
f.suptitle("Box Plot to 'Amount' and 'Time'", fontsize=18)
    
plt.show()
# Show Distriupution of Time and Amount

list_columns = ['Amount', 'Time']

list_describes = []
for f in list_columns:
    list_describes.append(df[f].describe())

df_describes = pd.concat(list_describes, axis = 1)
df_describes  
from sklearn.preprocessing import StandardScaler, RobustScaler

rob_scaler = RobustScaler() # Reduce influence of outliers in scaling using IQR (Inter Quartile Range)

df['Amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.head(1)
f, (ax1, ax2) = plt.subplots(ncols=2,  figsize=(16, 5), sharex=False)

map_feat_ax = {'Amount': ax1, 'Time': ax2}

for key, value in map_feat_ax.items():
    sns.boxplot(x=df[key], ax=value)
    
f.suptitle("Box Plot to 'Amount' and 'Time' after Scaled", fontsize=18)
    
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold

# Separate dataset in train test
X = df.drop('Class', axis=1)
y = df['Class']

# Separa os dados de maneira estratificada (mantendo as proporções originais)
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

# De forma iterativa, no final teremos separao 20% para teste e 80% para treinamento
for train_index, test_index in sss.split(X, y):
    original_Xtrain, original_Xtest = X.iloc[train_index].values, X.iloc[test_index].values
    original_ytrain, original_ytest = y.iloc[train_index].values, y.iloc[test_index].values
    
# calculate to check if the 2 sub-set (train,test) have the smae proportion of rows with classes 0 (No fraud) and 1 (fraud)
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
prop_train = train_counts_label/ len(original_ytrain)
prop_test = test_counts_label/ len(original_ytest)
original_size = len(X)

# Print Restult to cofirm that has the same proportion
print("Split DataFrame in Train and Test\n")
print("Original Size:", '{:,d}'.format(original_size))
print("\nTrain: must be 80% of dataset:\n", 
      "X_train:", '{:,d}'.format(len(original_Xtrain)), '{:.2%}'.format(len(original_Xtrain)/original_size),
      "| y_train:", '{:,d}'.format(len(original_ytrain)), '{:.2%}'.format(len(original_ytrain)/original_size),
            "\n => Classe 0 (No Fraud):", train_counts_label[0],  '{:.2%}'.format(prop_train[0]), 
            "\n => Classe 1 (Fraud):   ", train_counts_label[1], '{:.2%}'.format(prop_train[1]),
      "\n\nTest: must be 20% of dataset:\n",
      "X_test:", '{:,d}'.format(len(original_Xtest)), '{:.2%}'.format(len(original_Xtest)/original_size),
      "| y_test:", '{:,d}'.format(len(original_ytest)), '{:.2%}'.format(len(original_ytest)/original_size),
              "\n => Classe 0 (No Fraud)", test_counts_label[0], '{:.2%}'.format(prop_test[0]),
              "\n => Classe 1 (Fraud)   ",test_counts_label[1], '{:.2%}'.format(prop_test[1])
     )
# frac = 1 means all data
df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1] # df where is class = 1
non_fraud_df = df.loc[df['Class'] == 0][:492] # df where class = 0 (no fraud) limited by coutn masx of fraude 492

# join 2 df to make a datacet balanced
normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows to shrunbel y=1 and y=0 (else will bi sorted (thasi is abnormal))
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.shape
eda_categ_feat_desc_plot(new_df['Class'], 'Random Under-Sampling: to correct unbalanced  (Fraud = 1; No Fraud = 0)')
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

# Entire DataFrame (Unbalanced)
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, cmap='coolwarm_r', annot=True, annot_kws={'size':8}, ax=ax1, mask=mask)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)

# HeatMap to new_df (Balanced)
sub_sample_corr = new_df.corr()

mask = np.zeros_like(sub_sample_corr)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot=True, annot_kws={'size':8}, ax=ax2, mask=mask)
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()
# Generate Ranking of correlations (boht positives, negatives)

corr = new_df.corr() # Show greater correlations both negative and positive
dict_to_rename = {0: "value", "level_0": "feat1", "level_1": "feat2"} # Rename DataFrame
s = corr.unstack().reset_index().rename(dict_to_rename, axis = 1) # Restructure dataframe

s['+|-'] = s['value']
s['value'] = s['value'].abs()

# remove rows thas like 'x' | 'x' 
s_to_drop = s[(s['feat1'] == s['feat2'])].index 
s = s.drop(s_to_drop).reset_index()

s = s[ s['feat1'] == 'Class' ].sort_values(by="value", ascending=False).drop("index", axis=1) 

# Biggest correlation with class
top_int = 10
s.head(top_int)
f, (axes, axes2) = plt.subplots(ncols=4, nrows=2, figsize=(20,10))

colors = ["#0101DF", "#DF0101"]

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Class", y="V17", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="Class", y="V11", data=new_df, palette=colors, ax=axes2[0])
axes2[0].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=new_df, palette=colors, ax=axes2[1])
axes2[1].set_title('V4 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V2", data=new_df, palette=colors, ax=axes2[2])
axes2[2].set_title('V2 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V19", data=new_df, palette=colors, ax=axes2[3])
axes2[3].set_title('V19 vs Class Positive Correlation')

plt.show()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

colors = ['b', 'r']

# V14
sns.boxplot(x='Class', y='V14', data = new_df, ax=ax1, palette=colors)
ax1.set_title('Show outliers to V14', fontsize=14)

# V12
sns.boxplot(x='Class', y='V12', data = new_df, ax=ax2, palette=colors)
ax2.set_title('Show outliers to V12', fontsize=14)

# V10
sns.boxplot(x='Class', y='V10', data = new_df, ax=ax3, palette=colors)
ax3.set_title('Show outliers to V10', fontsize=14)

plt.show()
# Remover os outliers de V14 (correlação negativa alta com a classe)
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
# Valores do quartil 25 e quartil 75
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('QUARTIL 25: {} | QUARTIL 75: {}'.format(q25, q75))
# Interquartile range
v14_iqr = q75 - q25
print('IQR: ', v14_iqr)

# Limiar
v14_cut_off = v14_iqr * 1.5
# Limite superior e inferior
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('LIMIAR: ', v14_cut_off)
print('V14 LIMITE INFERIOR', v14_lower)
print('V14 LIMITE SUPERIOR', v14_upper)

# Ouliers (fora os limites estabelecidos anteriormente)
outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('V14 QUANTIDADE DE OUTLIERS EM FRAUDES:', len(outliers))

# Novo dataframe sem os outliers
new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
print('----' * 20)


# Remover os outliers de V12
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('V12 LIMITE INFERIOR: {}'.format(v12_lower))
print('V12 LIMITE SUPERIOR: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]

print('V12 OUTLIERS: {}'.format(outliers))
print('V12 QUANTIDADE DE OUTLIERS EM FRAUDES: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)
print('NÚMERO DE INSTÂNCIAS APÓS A REMOÇÃO DOS OUTLIERS: {}'.format(len(new_df)))
print('----' * 20)


# Remover os outliers de V10

v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('V10 LIMITE INFERIOR: {}'.format(v10_lower))
print('V10 SUPERIOR: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]

print('V10 OUTLIERS: {}'.format(outliers))
print('V10 QUANTIDAADE DE OUTLIERS EM FRAUDES: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)


print('---' * 20)
print('NÚMERO DE INSTÂNCIAS APÓS A REMOÇÃO DOS OUTLIERS (Antes era 984): {}'.format(len(new_df)))
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

colors = ['b', 'r']

# V14
sns.boxplot(x='Class', y='V14', data = new_df, ax=ax1, palette=colors)
ax1.set_title('Show outliers to V14 after remove', fontsize=14)

# V12
sns.boxplot(x='Class', y='V12', data = new_df, ax=ax2, palette=colors)
ax2.set_title('Show outliers to V12 after remove', fontsize=14)

# V10
sns.boxplot(x='Class', y='V10', data = new_df, ax=ax3, palette=colors)
ax3.set_title('Show outliers to V10 after remove', fontsize=14)


plt.show()
# New_df is from the random undersample data (fewer instances)
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# T-SNE Implementation: Tae a time in comparision with others techniques
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

# In UnderSampling DataSet Balanced
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
    print("Classifiers: ", classifier.__class__.__name__,
          "\n\tHas a training score of", round(training_score.mean(), 2) * 100, "% accuracy score on ")
# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV

# Logistic Regression Best Estimators
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
log_reg = grid_log_reg.best_estimator_
print('Best Loggistic Params:', log_reg)

# KNears best estimator
knears_params = {"n_neighbors": list(range(2,5,1)), 
                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
knears_neighbors = grid_knears.best_estimator_
print('Best KNears Params:', knears_neighbors)

# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 
              'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)
svc = grid_svc.best_estimator_
print('Best SVM Params:', svc)

# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"],
               "max_depth": list(range(2,4,1)), 
               "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)
tree_clf = grid_tree.best_estimator_
print('Best Tree Params:', tree_clf)
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
# Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,
                             method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,
                             method="decision_function")

tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)

from sklearn.metrics import roc_auc_score

print("ROC AUC SCORE\n")

print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))

print("\nCROSS VALIDATION\n")

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)


def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)
plt.show()
from sklearn.metrics import confusion_matrix

y_pred = log_reg.predict(X_test)
y_pred

labels = ['No Fraud', 'Fraud']

confusion_mtx = confusion_matrix(y_test, y_pred)

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(confusion_mtx, labels, title="Random UnderSample - Logistic Regression \n Confusion Matrix")

from sklearn.metrics import classification_report

# Logistic Regression fitted using SMOTE technique
y_pred_log_reg = log_reg.predict(X_test)

# Other models fitted with UnderSampling
y_pred_knear = knears_neighbors.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_tree = tree_clf.predict(X_test)

log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
svc_cf = confusion_matrix(y_test, y_pred_svc)
tree_cf = confusion_matrix(y_test, y_pred_tree)

plt.show()

print('Logistic Regression:')
print(classification_report(y_test, y_pred_log_reg))

print('KNears Neighbors:')
print(classification_report(y_test, y_pred_knear))

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_svc))

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_tree))
# We will undersample during cross validating
undersample_X = df.drop('Class', axis=1)
undersample_y = df['Class']

# Divide Original DataSet in Train and Test
for train_index, test_index in sss.split(undersample_X, undersample_y):
    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]
    
undersample_Xtrain = undersample_Xtrain.values
undersample_Xtest = undersample_Xtest.values
undersample_ytrain = undersample_ytrain.values
undersample_ytest = undersample_ytest.values 

np.unique(undersample_ytrain, return_counts=True), np.unique(undersample_ytest, return_counts=True)
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict

# Com o modelo treinado no undersampling dataset balanced, testamos no ytrain
log_reg_pred = cross_val_predict(log_reg, undersample_Xtrain, undersample_ytrain, cv=5,
                             method="decision_function")

# Perceba, mesmo que tivermos scores altos, em tudo, somente analisando precision na 
print("LOGISTIC REGRESSION\n")

print('ROC AUC SCORE: ', roc_auc_score(undersample_ytrain, log_reg_pred))

log_reg_score = cross_val_score(log_reg, undersample_Xtrain, undersample_ytrain, cv=5)
print('Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')
y_pred_log_reg = log_reg.predict(undersample_Xtest)

log_reg_cf = confusion_matrix(undersample_ytest, y_pred_log_reg)

labels = ['No Fraud', 'Fraud']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(log_reg_cf, labels, title="Random UnderSample \n Confusion Matrix")
print('Logistic Regression - Classification Report\n')
print(classification_report(undersample_ytest, y_pred_log_reg))
print('Tamanho do X (treino): {} | Tamanho do y (treino): {}'.format(len(original_Xtrain), len(original_ytrain)))
print('Tamanho do X (teste): {} | Tamanho do y (teste): {}'.format(len(original_Xtest), len(original_ytest)))

# Lista para armazenar os scores
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

# Parâmetros da Logistic Regression
log_reg_params = {
    'penalty': ['l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# Randomized SearchCV
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)

# Implementação do SMOTE
# Cross-validation da maneira correta
for train, test in sss.split(original_Xtrain, original_ytrain):
    # Pipeline
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE durante a validação cruzada
    # Treinamento do modelo
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    # Melhores parâmetros
    best_est = rand_log_reg.best_estimator_
    # Predições
    prediction = best_est.predict(original_Xtrain[test])
    
rand_log_reg
y_pred_best_est_smote = best_est.predict(original_Xtest)

best_est_smote_cf = confusion_matrix(original_ytest, y_pred_log_reg)

labels = ['No Fraud', 'Fraud']

fig = plt.figure(figsize=(16,8))
fig.add_subplot(221)

plot_confusion_matrix(best_est_smote_cf, labels, title="SMOTE OverSample \n Confusion Matrix")
# Printa a "classification report"
print(classification_report(original_ytest, y_pred_best_est_smote, target_names=labels))
y_pred = log_reg.predict(original_Xtest)
undersample_score = accuracy_score(original_ytest, y_pred)
print(classification_report(original_ytest, y_pred, target_names=labels))
# Logistic Regression com "undersampling" sobre seu dado de test
y_pred = log_reg.predict(X_test)
undersample_score = accuracy_score(y_test, y_pred)

# Logistic Regression com SMOTE
y_pred_sm = best_est.predict(original_Xtest)
oversample_score = accuracy_score(original_ytest, y_pred_sm)

# Dicionário com os scores das duas técnicas (undersampling e oversampling)
d = {
    'Técnica': ['Random undersampling', 'Oversampling (SMOTE)'],
    'Score': [undersample_score, oversample_score]
}

# Cria um dataframe com o dicionário
final_df = pd.DataFrame(data=d)

# Armazena o "Score" em outra variável
score = final_df['Score']
# Remove a coluna "Score"
final_df.drop('Score', axis=1, inplace=True)
# Insere os dados armazenados anteriormente na segunda coluna
final_df.insert(1, 'Score', score)

final_df
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

# Tamanho da camada de entrada
n_inputs = X_train.shape[1]

# Criação da rede
undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
print("Training Dataset generated by RANDOM UnderSampling: Size = ", len(X_train))

undersample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose = 0)
undersample_fraud_predictions = undersample_model.predict_classes(original_Xtest, batch_size=200)

print("Neural Net KERAS with UnderSampling:\n")
print(classification_report(original_ytest, undersample_fraud_predictions, target_names=labels))
# SMOTE
sm = SMOTE('minority', random_state=42)

# Treina os dados originais utilizando SMOTE
Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)
print("Training Dataset generated by SMOTE OverSampling: Size = ", len(Xsm_train))
n_inputs = Xsm_train.shape[1]

oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=0)
oversample_fraud_predictions = oversample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)

print("Neural Net KERAS with SMOTE:\n")
print(classification_report(original_ytest, oversample_fraud_predictions, target_names=labels))
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
t0 = time.time()
sm = SMOTEENN("minority", random_state=42)
Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)
t1 = time.time()
print("SMOTEENN took {:.6} s".format(t1 - t0)) # 864.109 s = 15minutos
t0 = time.time()
print('Resampled dataset shape BEFORE %s' % Counter(original_ytrain))

print('Resampled dataset shape AFTER %s' % Counter(ysm_train))
t1 = time.time()
print("PRINT took {:.6} s".format(t1 - t0))
n_inputs = Xsm_train.shape[1]

oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=0)

# Show Result

oversample_fraud_predictions = oversample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)

print("Neural Net KERAS with SMOTEENN:\n")
print(classification_report(original_ytest, oversample_fraud_predictions, target_names=labels))
"""
Neural Net KERAS with SMOTEENN:

              precision    recall  f1-score   support

    No Fraud       1.00      1.00      1.00     56863
       Fraud       0.78      0.72      0.75        98

    accuracy                           1.00     56961
   macro avg       0.89      0.86      0.88     56961
weighted avg       1.00      1.00      1.00     56961
"""
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from collections import Counter

t0 = time.time()
sm = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
Xsm_train2, ysm_train2 = sm.fit_sample(original_Xtrain, original_ytrain)
t1 = time.time()
print("SMOTETomek took {:.6} s".format(t1 - t0)) # 543.303 s = 9 minutos
n_inputs = Xsm_train2.shape[1]

oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

oversample_model.fit(Xsm_train2, ysm_train2, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=0)

# Show Result

oversample_fraud_predictions = oversample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)

print("Neural Net KERAS with SMOTETomek:\n")
print(classification_report(original_ytest, oversample_fraud_predictions, target_names=labels))
"""
Neural Net KERAS with SMOTETomek:

              precision    recall  f1-score   support

    No Fraud       1.00      1.00      1.00     56863
       Fraud       0.90      0.67      0.77        98

    accuracy                           1.00     56961
   macro avg       0.95      0.84      0.89     56961
weighted avg       1.00      1.00      1.00     56961
"""
## SMOTEENN
# from collections import Counter
# # Class to perform over-sampling using SMOTE and cleaning using ENN.
# # Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.
# sm = SMOTEENN("minority", random_state=42)

# # Treina os dados originais utilizando SMOTE
# Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)
# print('Resampled dataset shape %s' % Counter(ysm_train))
# print("Training Dataset generated by SMOTEENN OverSampling: Size = ", len(Xsm_train))

# n_inputs = Xsm_train.shape[1]

# oversample_model = Sequential([
#     Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(2, activation='softmax')
# ])

# oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=0)

# # Show Result

# oversample_fraud_predictions = oversample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)

# print("Neural Net KERAS with SMOTEENN:\n")
# print(classification_report(original_ytest, oversample_fraud_predictions, target_names=labels))