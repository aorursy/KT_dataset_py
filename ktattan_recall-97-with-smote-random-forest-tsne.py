import pandas as pd
import numpy as np
import operator
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.over_sampling import SMOTE
import itertools
from collections import Counter
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,\
roc_auc_score,roc_curve,recall_score,precision_score,classification_report,f1_score
from sklearn.model_selection import cross_val_score
import warnings
import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('poster')
# read in data
df = pd.read_csv('../input/creditcard.csv')
# drop time and Amount - the non PCA features
df.drop(['Time','Amount'],axis=1,inplace=True)
# shuffle data
df.sample(frac=1.).reset_index(drop=True,inplace=True)
df.head()
fig, ax = plt.subplots(figsize=(8,6))
df.groupby(['Class'])['Class'].count().plot(kind='bar')
plt.xticks([0,1],['Legal (0)','Fraud (1)'], rotation=0)
plt.ylabel('Count')
plt.show()
n_fraud_records = len(df[df['Class'] == 1])
fraud_indices = np.array(df[df['Class'] == 1].index)
normal_indices = df[df.Class == 0].index
# choose a random sample of legal cases, equal in count to fraud cases
random_normal_indices = np.array(np.random.choice(normal_indices, n_fraud_records, replace=False))
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
under_sample_data = df.loc[under_sample_indices,:]
under_sample_data.reset_index(drop=True,inplace=True)

X = under_sample_data.loc[:, under_sample_data.columns != 'Class']
y = under_sample_data.loc[:, under_sample_data.columns == 'Class']

X = X.as_matrix()
y = y.as_matrix()
y = y.T[0]
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data['Class']==0])/len(under_sample_data))
print("Percentage of fraud transactions: ",len(under_sample_data[under_sample_data['Class']==1])/len(under_sample_data))
print("Total number of transactions in under sampled data: ", len(under_sample_data))
tsne = TSNE(n_components=2, random_state=np.random.randint(100))
matrix_2d = tsne.fit_transform(X)
colors = ['G' if i==0 else 'R' for i in y]
df_tsne = pd.DataFrame(matrix_2d)
df_tsne['Class'] = under_sample_data['Class']
df_tsne['color'] = colors
df_tsne.columns = ['x','y','Class','color']
cols = ['Class','color','x','y']
df_tsne = df_tsne[cols]
df_tsne.head()
# number of rows and columns
df_tsne.shape
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(df_tsne[df_tsne['Class']==1].x.values, df_tsne[df_tsne['Class']==1].y.values,
           c='red', s=10, alpha=0.5, label='Fraud')
ax.scatter(df_tsne[df_tsne['Class']==0].x.values, df_tsne[df_tsne['Class']==0].y.values,
           c='green', s=10, alpha=0.5, label='Legal')
ax.tick_params(axis='both', which='major', labelsize=15)
ax.legend()
plt.show();
# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=np.random.randint(100))
scores_dict = {num_trees:[] for num_trees in np.logspace(start=1,stop=3.478,num=20).astype(int)}
oob_err_dict = {num_trees:0 for num_trees in np.logspace(start=1,stop=3.478,num=20).astype(int)}
k = 10
for num_trees in np.logspace(start=1,stop=3.478,num=20).astype(int):
    warnings.filterwarnings("ignore")
    clf = RandomForestClassifier(n_estimators=num_trees, n_jobs=-1, max_depth=10, oob_score=True)
    scores = cross_val_score(clf, X_train, y_train, cv=k, n_jobs=-1)
    model = clf.fit(X_train, y_train)
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    scores_dict[num_trees].append([scores.mean(), precision, recall, f1])
    temp_oob = 1 - clf.oob_score_
    oob_err_dict[num_trees] = temp_oob
fig, ax = plt.subplots(figsize=(12,10))
ax.semilogx(list(scores_dict.keys()),[elem[0][0] for elem in scores_dict.values()], '-', label='Mean 10-Fold Accuracy')
ax.semilogx(list(scores_dict.keys()),[elem[0][1] for elem in scores_dict.values()], '-', label='Precision')
ax.semilogx(list(scores_dict.keys()),[elem[0][2] for elem in scores_dict.values()], '-', label='Recall')
ax.semilogx(list(scores_dict.keys()),[elem[0][3] for elem in scores_dict.values()], '-', label='F1')
ax.legend()
ax.set_xlabel('Number of Trees')
ax.set_ylabel('Measure')
# ax.set_ylim(0.89,1.)
plt.show()
fig, ax = plt.subplots(figsize=(12,10))
ax.semilogx(list(oob_err_dict.keys()), list(oob_err_dict.values()))
ax.set_xlabel('Number of trees', fontsize=20)
ax.set_ylabel('OOB error rate', fontsize=20)
plt.show()
N_trees = 200
clf = RandomForestClassifier(n_estimators=N_trees, n_jobs=-1, max_depth=10, criterion='entropy')
model = clf.fit(X_train, y_train)
preds = model.predict(X_test)
def plot_confusion_matrix(cm, classes, title='Confusion matrix',cmap=plt.cm.Blues, norm=False):
    """
    This function prints and plots the confusion matrix
    """
    fig, ax = plt.subplots(figsize=(12,8))
    
    if norm == True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round_(cm, decimals=3)
    
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.grid(False)
    plt.show()
cnf_matrix = confusion_matrix(y_test,model.predict(X_test))
class_names = [0,1]
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix', norm=True)
print("Recall:", cnf_matrix[1][1]/(cnf_matrix[1][1]+cnf_matrix[1][0]))
print("Precision:", cnf_matrix[1][1]/(cnf_matrix[1][1]+cnf_matrix[0][1]))
y_prob = clf.predict_proba(X_test)

pos_probs = [y_prob[i][1] for i in range(len(y_prob))]
    
fpr, tpr, threshold = roc_curve(y_test, pos_probs)
auc = roc_auc_score(y_test, pos_probs)

# Plot ROC
fig, ax = plt.subplots(figsize=(12,8))
ax.set_title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='Model - AUC = %0.3f'% auc)
ax.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--', label='Chance')
ax.legend()
ax.set_xlim([-0.1,1.0])
ax.set_ylim([-0.1,1.01])
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
plt.show()
y_prob = model.predict_proba(X_test)

thresholds = np.linspace(start=0.1,stop=0.9,num=9)
colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue','black']

plt.figure(figsize=(12,8))

for i,color in zip(thresholds,colors):
    # boolean values if the probability of a fraud classification is above the threshold
    y_bool = y_prob[:,1] > i
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_bool)
    
    # Plot Precision-Recall curve
    plt.plot(recall, precision, color=color,label='Threshold: %s'%i)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
plt.show()
n_fraud_records = len(df[df['Class'] == 1])
fraud_indices = np.array(df[df['Class'] == 1].index)
normal_indices = df[df.Class == 0].index
# amount to oversample by
factor = 20
random_normal_indices = np.array(np.random.choice(normal_indices, int(20*n_fraud_records), replace=False))
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
under_sample_data = df.loc[under_sample_indices,:]
under_sample_data.reset_index(drop=True,inplace=True)

X = under_sample_data.loc[:, under_sample_data.columns != 'Class']
y = under_sample_data.loc[:, under_sample_data.columns == 'Class']

X = X.as_matrix()
y = y.as_matrix()
y = y.T[0]

print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data['Class']==0])/len(under_sample_data))
print("Percentage of fraud transactions: ",len(under_sample_data[under_sample_data['Class']==1])/len(under_sample_data))
print("Total number of transactions in under sampled data: ", len(under_sample_data))
y = np.reshape(y, newshape=(-1,))
print('Original dataset shape {}'.format(Counter(y)))
smote = SMOTE(ratio='auto', random_state=np.random.randint(100), k_neighbors=5, 
              m_neighbors=10, kind='regular', n_jobs=-1)
X_res, y_res = smote.fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_res)))
matrix_2d = tsne.fit_transform(X_res)
colors = ['G' if i==0 else 'R' for i in y_res]
df_tsne = pd.DataFrame(matrix_2d)
df_tsne['Class'] = y_res
df_tsne['color'] = colors
df_tsne.columns = ['x','y','Class','color']
cols = ['Class','color','x','y']
df_tsne = df_tsne[cols]
df_tsne.head()
df_tsne.shape
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(df_tsne[df_tsne['Class']==1].x.values, df_tsne[df_tsne['Class']==1].y.values,
           c='red', s=10, alpha=0.5, label='Fraud')
ax.scatter(df_tsne[df_tsne['Class']==0].x.values, df_tsne[df_tsne['Class']==0].y.values,
           c='green', s=10, alpha=0.5, label='Legal')
ax.tick_params(axis='both', which='major', labelsize=15)
ax.legend()
plt.show();
X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.2, random_state=np.random.randint(100))
oob_err_dict = {num_trees:0 for num_trees in np.logspace(start=1,stop=3.478,num=20).astype(int)}
for num_trees in np.logspace(start=1,stop=3.478,num=20).astype(int):
    warnings.filterwarnings("ignore")
    clf = RandomForestClassifier(n_estimators=num_trees, n_jobs=-1, max_depth=10, oob_score=True)
    model = clf.fit(X_train, y_train)
    temp_oob = 1 - clf.oob_score_
    oob_err_dict[num_trees] = temp_oob
    print("trained with:",num_trees,"trees")
fig, ax = plt.subplots(figsize=(12,10))
ax.semilogx(list(oob_err_dict.keys()), list(oob_err_dict.values()))
ax.set_xlabel('Number of trees', fontsize=20)
ax.set_ylabel('OOB error rate', fontsize=20)
plt.show()
N_trees = 200
clf = RandomForestClassifier(n_estimators=N_trees, n_jobs=-1, max_depth=10)
model = clf.fit(X_train, y_train)
cnf_matrix = confusion_matrix(y_test,model.predict(X_test))
class_names = [0,1]
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix', norm=True)
print("Recall:", cnf_matrix[1][1]/(cnf_matrix[1][1]+cnf_matrix[1][0]))
print("Precision:", cnf_matrix[1][1]/(cnf_matrix[1][1]+cnf_matrix[0][1]))
y_prob = clf.predict_proba(X_test)

pos_probs = [y_prob[i][1] for i in range(len(y_prob))]
    
fpr, tpr, threshold = roc_curve(y_test, pos_probs)
auc = roc_auc_score(y_test, pos_probs)

# Plot ROC
fig, ax = plt.subplots(figsize=(12,8))
ax.set_title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='Model - AUC = %0.3f'% auc)
ax.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--', label='Chance')
ax.legend()
ax.set_xlim([-0.1,1.0])
ax.set_ylim([-0.1,1.01])
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
plt.show()
y_prob = model.predict_proba(X_test)

thresholds = np.linspace(start=0.1,stop=0.9,num=9)
colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue','black']

plt.figure(figsize=(12,8))

for i,color in zip(thresholds,colors):
    # boolean values if the probability of a fraud classification is above the threshold
    y_bool = y_prob[:,1] > i
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_bool)
    
    # Plot Precision-Recall curve
    plt.plot(recall, precision, color=color,label='Threshold: %s'%i)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
plt.show()
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(12,8))
ax.set_title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices],
       color='r', align='center')
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels([list(df.columns)[i] for i in indices], rotation=70)
ax.set_xlim([-1, X.shape[1]])
plt.show()