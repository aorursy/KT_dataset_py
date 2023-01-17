import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import collections

from collections import Counter

import itertools

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.under_sampling import NearMiss, RandomUnderSampler

from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, roc_curve, auc

import warnings

warnings.filterwarnings('ignore')
# load data

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.head()
# Check for NaN

df.isna().sum().any()
# summary stats

df.describe()
plt.figure(figsize=(8,5))

sns.distplot(df['Amount'])
# correlation between variables



plt.figure(figsize=(10,8))

plt.title('Correlation between features')

sns.heatmap(df.corr())
# Plot class distribution



fig = make_subplots(rows=1,cols=1)

trace = go.Bar(x = ['Normal cases','Fraud cases'], y = [len(df[df.Class==0]), len(df[df.Class==1])],

         name = 'Class distribution',

         text = [

             str(round(100*len(df[df.Class==0])/len(df),3)) + '%',

             str(round(100*len(df[df.Class==1])/len(df),3)) + '%'

         ],

         textposition = 'auto')

fig.append_trace(trace, 1, 1)

fig.update_layout(title='Class distribution', width=500, height=400)

fig.show()
# Scale amount



scaler = RobustScaler()

df['scaled_amount'] = scaler.fit_transform(df.Amount.values.reshape(-1,1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

sa = df['scaled_amount']

df.drop('scaled_amount', axis=1, inplace=True)

df.insert(0, 'scaled_amount', sa)
def kfold_scores(x, y):

    

    """

    This function will split data into 3 folds and 

    evaluates a Logistic Regression model over 4 values of 'C' parameter and returns best C

    

    """

    

    fold = StratifiedKFold(3, shuffle=True, random_state=10)

    

    #C paramater range

    c_param_range = [0.01, 0.1, 1, 10]

    

    results_table = pd.DataFrame(index = range(len(c_param_range)), columns = ['C_parameter', 'Mean_roc_auc'])

    results_table['C_parameter'] = c_param_range

    j = 0

    for c in c_param_range:

        print('-'*30)

        print(f'C parameter: {c}')

        print('-'*30)

        print('')

        

        recall_scores = []

        roc_scores = []

        i = 1

        for train_indices, test_indices in fold.split(x, y):

            

            lr = LogisticRegression(C=c, solver='liblinear', penalty = 'l1')            

            lr.fit(x.iloc[train_indices], y.iloc[train_indices])            

            y_pred = lr.predict(x.iloc[test_indices])

            

            recall_acc = recall_score(y.iloc[test_indices], y_pred)            

            recall_scores.append(recall_acc)  

            

            roc_scr = roc_auc_score(y.iloc[test_indices], y_pred)

            roc_scores.append(roc_scr)

            

            #print(f'Iteration:{i}', f'Recall score: {recall_acc:.4f}')

            i = i+1    

        print('Mean recall score ', round(np.mean(recall_scores),4))  

        print('Mean roc auc', round(np.mean(roc_scores),4))

        print('')

        results_table['Mean_roc_auc'].iloc[j] = round(np.mean(roc_scores),4)

        j += 1

    print('='*30)

    best_c = results_table['C_parameter'].iloc[results_table['Mean_roc_auc'].astype(float).idxmax()]

    print(f'Best model to choose from cross validation is with C-parameter:{best_c}')

    print('='*30)

    return best_c
X = df.drop('Class', axis=1)

y = df['Class']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)



print(f'size of original data {len(df)}')

print(f'No. of train examples in original data {len(X_train)}')

print(f'No. of test examples in original data {len(X_test)}')



best_c_orig_data = kfold_scores(X_train, y_train)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Reds):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(8,5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# Use best_c on all train examples and predict on test examples



lr = LogisticRegression(C = best_c_orig_data, penalty = 'l1', solver='liblinear')

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test,y_pred)

np.set_printoptions(precision=2)



print(f"Recall metric for test set: {(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])):.4f}")

print(f"ROC AUC for test set: {(roc_auc_score(y_test, y_pred)):.4f}")



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
# fit random forest classifier with default params

forest = RandomForestClassifier()

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test,y_pred)

np.set_printoptions(precision=2)



print(f"Recall for test set: {(recall_score(y_test, y_pred)):.4f}")

print(f"ROC AUC for test set: {(roc_auc_score(y_test, y_pred)):.4f}")



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
# list of sampling strategies to use

sampling_strategies = [0.05, 0.1, 1]



for sampling_strategy in sampling_strategies:

    us = RandomUnderSampler(sampling_strategy=sampling_strategy)

    X_us, y_us = us.fit_resample(X_train, y_train)

    print(f'Sampling strategy: {sampling_strategy}')

    print('='*25)

    print(f"No. of normal cases: {len(y_us[y_us==0])}")

    print(f"No. of fraud cases: {len(y_us[y_us==1])}")    

    # fit decision tree classifier with default params

    forest.fit(X_us, y_us) 

    y_pred = forest.predict(X_test)

    print(f"Recall: {(recall_score(y_test, y_pred)):.4f}")

    print(f"ROC AUC: {(roc_auc_score(y_test, y_pred)):.4f}")

    print('')
# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test,y_pred)

np.set_printoptions(precision=2)



print(f"Recall for test set: {(recall_score(y_test, y_pred)):.4f}")

print(f"ROC AUC for test set: {(roc_auc_score(y_test, y_pred)):.4f}")



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(22,8))

ax0 = plt.subplot(121)

ax0.set_title('Original data', fontsize=15)

ax0 = sns.heatmap(df.corr())

ax1 = plt.subplot(122)

ax1.set_title('Undersample data', fontsize=15)

ax1 = sns.heatmap((X_us.join(y_us)).corr())
sampling_strategies = [0.1, 0.5]

for sampling_strategy in sampling_strategies:

    os = SMOTE(sampling_strategy=sampling_strategy)

    X_os, y_os = os.fit_resample(X_train, y_train)

    print(f'Sampling strategy: {sampling_strategy}')

    print('='*25)

    print(f"No. of normal cases: {len(y_os[y_os==0])}")

    print(f"No. of fraud cases: {len(y_os[y_os==1])}")

    # fit decision tree classifier with default params

    forest.fit(X_os, y_os)

    y_pred = forest.predict(X_test)

    print(f"Recall: {(recall_score(y_test, y_pred)):.4f}")

    print(f"ROC AUC: {(roc_auc_score(y_test, y_pred)):.4f}")

    print('')
# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test,y_pred)

np.set_printoptions(precision=2)



print(f"Recall for test set: {(recall_score(y_test, y_pred)):.4f}")

print(f"ROC AUC for test set: {(roc_auc_score(y_test, y_pred)):.4f}")



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
from sklearn.ensemble import RandomForestClassifier



os = SMOTE(sampling_strategy=1)

us = RandomUnderSampler(sampling_strategy=0.5) 

# undersample train data

X_tr_1, y_tr_1 = us.fit_resample(X_train, y_train)

# oversample resampled data

X_tr_2, y_tr_2 = os.fit_resample(X_tr_1, y_tr_1)

forest = RandomForestClassifier()

forest.fit(X_tr_2,y_tr_2)

y_pred = forest.predict(X_test)



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test,y_pred)

np.set_printoptions(precision=2)



print(f"Recall for test set: {(recall_score(y_test, y_pred)):.4f}")

print(f"ROC AUC for test set: {(roc_auc_score(y_test, y_pred)):.4f}")



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
fpr, tpr, thresholds = roc_curve(y_test,y_pred)

roc_auc = auc(fpr,tpr)



plt.figure(figsize=(8,6))

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.4f'% roc_auc)

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.04,1.0])

plt.ylim([-0.04,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')

plt.show()