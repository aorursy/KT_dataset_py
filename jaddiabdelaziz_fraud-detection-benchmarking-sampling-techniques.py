# Imported Libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import interp
from sklearn import metrics
from sklearn.metrics import auc, roc_curve, average_precision_score, f1_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline

df  = pd.read_csv('../input/creditcardfraud/creditcard.csv')    
df.head()

import seaborn as sns

#sns.set_theme(style="darkgrid")
sns.countplot(x='Class', data=df,palette="Set3")
df.describe()
std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

#Split the data into x and y variables

X = df.drop('Class', axis=1)
y = df['Class']

df.head()

SK= StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in SK.split(X, y):
    Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
    ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

train_unique_label, train_counts_label = np.unique(ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(ytest, return_counts=True)
print('Label Distributions in ytrain and ytest: \n')
print(train_counts_label/len(ytrain))
print(test_counts_label/len(ytest))

LW = 2
RANDOM_STATE = 42

class DummySampler:
    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        return self.sample(X, y)
    
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
rand_log_reg = RandomizedSearchCV(LogisticRegression(solver='liblinear'), log_reg_params, n_iter=4)
samplers = [
    ['Standard', DummySampler()],
    ['ADASYN', ADASYN( )],
    ['ROS', RandomOverSampler()],
    ['SMOTE', SMOTE()],]

pipelines = [['{}'.format(sampler[0]),make_pipeline(sampler[1], rand_log_reg )] for sampler in samplers ]
report_list = pd.DataFrame( index = ['f1', 'precision', 'recall',' average_precision' ],columns =[sampler[0] for sampler in samplers])

fig, axs = plt.subplots(1, 2, figsize=(15,10))
fig1, axs1 = plt.subplots(2, 2, figsize=(22,12))
title_cm= [sampler[0] for sampler in samplers]
for (idx, (name, pipeline)), ax1 in zip(enumerate(pipelines,0), axs1.flat):
    pipeline.fit(Xtrain, ytrain)
    best_clf = rand_log_reg.best_estimator_
    yhat= best_clf.predict_proba(Xtest)
    ypred0 = best_clf.decision_function(Xtest)
    ypred = best_clf.predict(Xtest)
    fpr, tpr, thresholds = roc_curve(ytest, yhat[:,1])
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(ytest,ypred)
    pr, re, thresholds = precision_recall_curve(ytest, yhat[:,1])
    pr_auc = auc(re, pr)
    report_list.iloc[[0],[idx]] = [f1]
    report_list.iloc[[1],[idx]] = [metrics.precision_score(ytest, ypred)]
    report_list.iloc[[2],[idx]] = [metrics.recall_score(ytest, ypred)]
    report_list.iloc[[3],[idx]] = [average_precision_score(ytest, yhat[:,1])]    
    axs[1].plot(pr, re, linestyle='-', label=r'%s (area = %0.3f )' % (name, pr_auc),lw=LW)
    axs[0].plot(fpr, tpr, label='{} (area = %0.3f)'.format(name) % roc_auc, lw=LW)
# confusion_matrix
    cm_nn = confusion_matrix(ytest,ypred)
    sns.heatmap(cm_nn, ax=ax1,annot=True,robust=True,fmt='g' ,cmap="Reds", cbar=False)
    ax1.set_title(title_cm[idx], fontsize=14)

axs[0].plot([0, 1], [0, 1], linestyle='--', lw=LW, color='k', label='Luck')
# make nice plotting
xlabel= ['False Positive Rate', 'Recall']
ylabel= ['True Positive Rate', 'Precision']
title = ['Receiver operating characteristisc (ROC)', 'Precision-Recall Curve ']
for i, ax in  enumerate(axs.flat, 0):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.axis(xmin=0,xmax=1)
    ax.axis(ymin=0,ymax=1)
    ax.set_xlabel(xlabel[i])
    ax.set_ylabel(ylabel[i])
    ax.set_title(title[i])
    ax.legend(loc="lower right")

plt.show()

report_list
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
rand_log_reg = RandomizedSearchCV(LogisticRegression(solver='liblinear'), log_reg_params, n_iter=4)
samplers = [
    ['Standard', DummySampler()],
    ['Under', RandomUnderSampler()],
    ['TomekLinks', TomekLinks()],
    ['EditedNN', EditedNearestNeighbours()],]

pipelines = [['{}'.format(sampler[0]),make_pipeline(sampler[1], rand_log_reg )] for sampler in samplers ]
report_list = pd.DataFrame( index = ['f1', 'precision', 'recall',' average_precision' ],columns =[sampler[0] for sampler in samplers])

fig, axs = plt.subplots(1, 2, figsize=(15,10))
fig1, axs1 = plt.subplots(2, 2, figsize=(22,12))
title_cm= [sampler[0] for sampler in samplers]
for (idx, (name, pipeline)), ax1 in zip(enumerate(pipelines,0), axs1.flat):
    pipeline.fit(Xtrain, ytrain)
    best_clf = rand_log_reg.best_estimator_
    yhat= best_clf.predict_proba(Xtest)
    ypred0 = best_clf.decision_function(Xtest)
    ypred = best_clf.predict(Xtest)
    fpr, tpr, thresholds = roc_curve(ytest, yhat[:,1])
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(ytest,ypred)
    pr, re, thresholds = precision_recall_curve(ytest, yhat[:,1])
    pr_auc = auc(re, pr)
    report_list.iloc[[0],[idx]] = [f1]
    report_list.iloc[[1],[idx]] = [metrics.precision_score(ytest, ypred)]
    report_list.iloc[[2],[idx]] = [metrics.recall_score(ytest, ypred)]
    report_list.iloc[[3],[idx]] = [average_precision_score(ytest, yhat[:,1])]    
    axs[1].plot(pr, re, linestyle='-', label=r'%s (area = %0.3f )' % (name, pr_auc),lw=LW)
    axs[0].plot(fpr, tpr, label='{} (area = %0.3f)'.format(name) % roc_auc, lw=LW)
# confusion_matrix
    cm_nn = confusion_matrix(ytest,ypred)
    sns.heatmap(cm_nn, ax=ax1,annot=True,robust=True,fmt='g' ,cmap="Reds", cbar=False)
    ax1.set_title(title_cm[idx], fontsize=14)

axs[0].plot([0, 1], [0, 1], linestyle='--', lw=LW, color='k', label='Luck')
# make nice plotting
xlabel= ['False Positive Rate', 'Recall']
ylabel= ['True Positive Rate', 'Precision']
title = ['Receiver operating characteristisc (ROC)', 'Precision-Recall Curve ']
for i, ax in  enumerate(axs.flat, 0):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.axis(xmin=0,xmax=1)
    ax.axis(ymin=0,ymax=1)
    ax.set_xlabel(xlabel[i])
    ax.set_ylabel(ylabel[i])
    ax.set_title(title[i])
    ax.legend(loc="lower right")

plt.show()

report_list
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
rand_log_reg = RandomizedSearchCV(LogisticRegression(solver='liblinear'), log_reg_params, n_iter=4)
SamplersOverUnder = [
    ['Standard', [DummySampler(),DummySampler()]],
    ['SMOTE+RandomUnder', [ RandomUnderSampler(), SMOTE()]],
    ['SMOTE+TomekLinks', [TomekLinks(),SMOTE()  ]],
    ['SMOTE+EditedNN', [EditedNearestNeighbours(), SMOTE(),  ]],]
pipelines = [ ['{}'.format(samplerOvUn[0]), make_pipeline(samplerOvUn[1][0],samplerOvUn[1][1], rand_log_reg )] for samplerOvUn in SamplersOverUnder ]
report_list = pd.DataFrame( index = ['f1', 'precision', 'recall',' average_precision' ],columns =[sampler[0] for sampler in SamplersOverUnder])

fig, axs = plt.subplots(1, 2, figsize=(15,10))
fig1, axs1 = plt.subplots(2, 2, figsize=(22,12))
title_cm= [sampler[0] for sampler in SamplersOverUnder]
for (idx, (name, pipeline)), ax1 in zip(enumerate(pipelines,0), axs1.flat):
    pipeline.fit(Xtrain, ytrain)
    best_clf = rand_log_reg.best_estimator_
    yhat= best_clf.predict_proba(Xtest)
    ypred0 = best_clf.decision_function(Xtest)
    ypred = best_clf.predict(Xtest)
    fpr, tpr, thresholds = roc_curve(ytest, yhat[:,1])
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(ytest,ypred)
    pr, re, thresholds = precision_recall_curve(ytest, yhat[:,1])
    pr_auc = auc(re, pr)
    report_list.iloc[[0],[idx]] = [f1]
    report_list.iloc[[1],[idx]] = [metrics.precision_score(ytest, ypred)]
    report_list.iloc[[2],[idx]] = [metrics.recall_score(ytest, ypred)]
    report_list.iloc[[3],[idx]] = [average_precision_score(ytest, yhat[:,1])]    
    axs[1].plot(pr, re, linestyle='-', label=r'%s (area = %0.3f )' % (name, pr_auc),lw=LW)
    axs[0].plot(fpr, tpr, label='{} (area = %0.3f)'.format(name) % roc_auc, lw=LW)
# confusion_matrix
    cm_nn = confusion_matrix(ytest,ypred)
    sns.heatmap(cm_nn, ax=ax1,annot=True,robust=True,fmt='g' ,cmap="Reds", cbar=False)
    ax1.set_title(title_cm[idx], fontsize=14)

axs[0].plot([0, 1], [0, 1], linestyle='--', lw=LW, color='k', label='Luck')
# make nice plotting
xlabel= ['False Positive Rate', 'Recall']
ylabel= ['True Positive Rate', 'Precision']
title = ['Receiver operating characteristisc (ROC)', 'Precision-Recall Curve ']
for i, ax in  enumerate(axs.flat, 0):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.axis(xmin=0,xmax=1)
    ax.axis(ymin=0,ymax=1)
    ax.set_xlabel(xlabel[i])
    ax.set_ylabel(ylabel[i])
    ax.set_title(title[i])
    ax.legend(loc="lower right")

plt.show()

report_list
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
rand_log_reg = RandomizedSearchCV(LogisticRegression(solver='liblinear'), log_reg_params, n_iter=4)

SamplersOverUnder = [
    ['Standard', [DummySampler(),DummySampler()]],
    ['ADASYN+RandomUnder', [ADASYN(sampling_strategy=0.3), RandomUnderSampler()]],
    ['ADASYN+TomekLinks', [ADASYN(sampling_strategy=0.3),  TomekLinks()]],
    ['ADASYN+EditedNN', [ADASYN(sampling_strategy=0.3),  EditedNearestNeighbours()]],]
pipelines = [ ['{}'.format(samplerOvUn[0]), make_pipeline(samplerOvUn[1][0],samplerOvUn[1][1], rand_log_reg )] for samplerOvUn in SamplersOverUnder ]
report_list = pd.DataFrame( index = ['f1', 'precision', 'recall',' average_precision' ],columns =[sampler[0] for sampler in SamplersOverUnder])

fig, axs = plt.subplots(1, 2, figsize=(15,10))
fig1, axs1 = plt.subplots(2, 2, figsize=(22,12))
title_cm= [sampler[0] for sampler in SamplersOverUnder]
for (idx, (name, pipeline)), ax1 in zip(enumerate(pipelines,0), axs1.flat):
    pipeline.fit(Xtrain, ytrain)
    best_clf = rand_log_reg.best_estimator_
    yhat= best_clf.predict_proba(Xtest)
    ypred = best_clf.predict(Xtest)
    fpr, tpr, thresholds = roc_curve(ytest, yhat[:,1])
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(ytest,ypred)
    pr, re, thresholds = precision_recall_curve(ytest, yhat[:,1])
    pr_auc = auc(re, pr)
    report_list.iloc[[0],[idx]] = [f1]
    report_list.iloc[[1],[idx]] = [metrics.precision_score(ytest, ypred)]
    report_list.iloc[[2],[idx]] = [metrics.recall_score(ytest, ypred)]
    report_list.iloc[[3],[idx]] = [average_precision_score(ytest, yhat[:,1])]    
    axs[1].plot(pr, re, linestyle='-', label=r'%s (area = %0.3f )' % (name, pr_auc),lw=LW)
    axs[0].plot(fpr, tpr, label='{} (area = %0.3f)'.format(name) % roc_auc, lw=LW)
# confusion_matrix
    cm_nn = confusion_matrix(ytest,ypred)
    sns.heatmap(cm_nn, ax=ax1,annot=True,robust=True,fmt='g' ,cmap="Reds", cbar=False)
    ax1.set_title(title_cm[idx], fontsize=14)

axs[0].plot([0, 1], [0, 1], linestyle='--', lw=LW, color='k', label='Luck')
# make nice plotting
xlabel= ['False Positive Rate', 'Recall']
ylabel= ['True Positive Rate', 'Precision']
title = ['Receiver operating characteristisc (ROC)', 'Precision-Recall Curve ']
for i, ax in  enumerate(axs.flat, 0):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.axis(xmin=0,xmax=1)
    ax.axis(ymin=0,ymax=1)
    ax.set_xlabel(xlabel[i])
    ax.set_ylabel(ylabel[i])
    ax.set_title(title[i])
    ax.legend(loc="lower right")

plt.show()

report_list