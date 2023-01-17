%matplotlib inline

import os



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from imblearn.combine import SMOTEENN 



from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score



np.random.seed(5)
# Load csv data to data frame

file_path = '../input/creditcard.csv'

df = pd.read_csv(file_path, sep=",")
df.head()
class_ = df.Class # since class is preserved in Python, use class_ instead

df.drop('Class', axis=1, inplace=True)

df.insert(0, 'Class', class_)

df.head()
df.isnull().any()
df.shape
df.dtypes
fraud_rate = df.Class.value_counts() / df.shape[0]

fraud_rate
df.describe()
# Overview of fraud and normal transactions

fraud_summary = df.groupby('Class')

fraud_summary.mean().T
corr = df.corr()

# plot heat map

fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            ax = ax,

            cmap='YlGnBu')

plt.title('Heatmap of Correlation Matrix')
corr
amount_population = df.Amount.mean()

amount_fraud = df[df.Class == 1].Amount.mean()

print('mean amount of population: {}, mean amount of fraud transaction: {}'.format(amount_population, amount_fraud))
import scipy.stats as stats

stats.ttest_1samp(a=df[df['Class']==1]['Amount'], 

                  popmean=amount_population)
degree_freedom = len(df[df['Class']==1])

conf_level = 0.95



LQ = stats.t.ppf((1-conf_level)/2,degree_freedom)  # Left Quartile



RQ = stats.t.ppf((1+conf_level)/2,degree_freedom)  # Right Quartile



print ('The t-distribution left quartile range is: ' + str(LQ))

print ('The t-distribution right quartile range is: ' + str(RQ))
# For computational efficiency, only visualize pairwise relationships among several features, 

# including two principal components

sns.pairplot(df.loc[:, ['Class', 'Amount', 'Time', 'V1', 'V2']], hue='Class')
# Kernel Density Plot

fig = plt.figure(figsize=(16,9),)

ax=sns.kdeplot(df.loc[(df['Class'] == 0), 'Amount'] , color='b', shade=True,label='normal transaction')

ax=sns.kdeplot(df.loc[(df['Class'] == 1), 'Amount'] , color='r', shade=True, label='fraud transaction')

plt.title('Transaction amount distribution - normal V.S. fraud')
# Kernel Density Plot

fig = plt.figure(figsize=(16,9),)

ax=sns.kdeplot(df.loc[(df['Class'] == 0), 'Time'] , color='b', shade=True,label='normal transaction')

ax=sns.kdeplot(df.loc[(df['Class'] == 1), 'Time'] , color='r', shade=True, label='fraud transaction')

plt.title('Transaction time distribution - normal V.S. fraud')
sns.lmplot(x='Time', y='Amount', data=df,

           fit_reg=False, # No regression line

           hue='Class')   # Color by evolution stage
sns.jointplot(x='Time', y='Amount', data=df[df['Class']==0], color='b')

sns.jointplot(x='Time', y='Amount', data=df[df['Class']==1], color='r')
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5)

X = df.drop(['Class', 'Time'], axis=1)

X = StandardScaler().fit_transform(X.values)

y = df['Class'].values

for train_index, test_index in sss.split(X, y):

    X_train_ = X[train_index, :]

    y_train_ = y[train_index]

    X_test = X[test_index, :]

    y_test = y[test_index]
y_train_pos = y_train_[y_train_ == 1]

y_test_pos = y_test[y_test == 1]

print('# positive in train data: {}, {}%'.format(y_train_pos.shape[0], y_train_pos.shape[0]*100. / y_train_.shape[0]))

print('# positive in test data: {}, {}%'.format(y_test_pos.shape[0], y_test_pos.shape[0]*100. / y_test.shape[0]))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import recall_score

from sklearn.model_selection import StratifiedKFold



def kfold_cv(Model, X, y, n_splits=10, smote=False, verbose=False):

    """

    Args:

        model: object that has fit, predict_proba methods

        X: array

        y: array

        n_splits: number of splits

    """

    skf = StratifiedKFold(n_splits, random_state=5, shuffle=True)

    C = np.logspace(-3, 3, num=7, base=10)

    def sub_cv(model):

        kfold = skf.split(X, y)

        scores = 0

        recall = 0

        if smote:

            sme = SMOTEENN(random_state=5)

        i = 0

        for train_index, test_index in kfold:

            X_train_ = X[train_index, :]

            y_train_ = y[train_index]

            X_test = X[test_index, :]

            y_test = y[test_index]

            if smote:

                X_train, y_train = sme.fit_sample(X_train_, y_train_)

            else:

                X_train = X_train_

                y_train = y_train_

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            y_score = model.predict_proba(X_test)[:, 1]

            score = roc_auc_score(y_test, y_score, average='micro')

            if verbose:

                print('Trained {} th model, AUC score: {}'.format(i+1, score))

            scores += score

            recall += recall_score(y_test, y_pred)

            i += 1

        return scores / i, recall / i

    bestC = 0

    bestauc = 0

    bestrecall = 0

    for c in C:

        model = Model(class_weight='balanced', C=c)

        auc, recall = sub_cv(model)

        if recall > bestrecall:

            bestauc = auc

            bestC = c

            bestrecall = recall

        print('C: {}, AUC: {}, recall: {}, best C: {}'.format(c, auc, recall, bestC))

    return bestC, bestauc, bestrecall
Model = LogisticRegression

bestC, bestauc, bestrecall = kfold_cv(Model, X_train_, y_train_, n_splits=5, verbose=False)

print('Best C: {}'.format(bestC))
baseline = Model(class_weight='balanced', C=bestC)

baseline.fit(X_train_, y_train_)

y_pred = baseline.predict(X_test)

y_score = baseline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_score, average='micro')

recall = recall_score(y_test, y_pred)

print('AUC: {}, recall: {}'.format(auc, recall))
def undersample(X_train_, y_train_, n_major):

    """

    In y_train_, positive class is far fewer than negative one.

    """

    X_train_pos = X_train_[y_train_ == 1]

    X_train_neg = X_train_[y_train_ == 0]

    y_train_pos = y_train_[y_train_ == 1]

    y_train_neg = y_train_[y_train_ == 0]

    undersample_y_train_neg_index = np.random.choice(y_train_neg.shape[0], n_major, replace=False)

    undersample_y_train_ = np.concatenate((y_train_pos, y_train_neg[undersample_y_train_neg_index]), axis=0)

    undersample_X_train_ = np.concatenate((X_train_pos, X_train_neg[undersample_y_train_neg_index]), axis=0)

    indices = np.arange(undersample_X_train_.shape[0])

    np.random.shuffle(indices)

    return undersample_X_train_[indices, :], undersample_y_train_[indices]
n_majority = np.arange(1, 50, 10) * y_train_pos.shape[0]

res = pd.DataFrame(data=np.zeros((len(n_majority), 3)), columns=['best_c', 'auc', 'recall'])

for i, n_major in enumerate(n_majority):

    undersample_X_train_, undersample_y_train_ = undersample(X_train_, y_train_, n_major)

    bestC, auc, recall = kfold_cv(Model, undersample_X_train_, undersample_y_train_, n_splits=5)

    res.loc[i, 'best_c'] = bestC

    res.loc[i, 'auc'] = auc

    res.loc[i, 'recall'] = recall

    print('undersample number: {}, best C: {}'.format(n_major, bestC))
res['maj_class_num'] = n_majority

res
fig = plt.figure(figsize=(16, 9))

plt.plot(res['maj_class_num'], res['auc'], 'b', label='AUC')

plt.plot(res['maj_class_num'], res['recall'], 'r', label='Recall')

plt.xlabel('maj_class_num')

plt.legend()

plt.grid()

plt.title('AUC and recall of logistic regression with balanced class weight trained on undersampled majority class of different numbers')

plt.show()
# Applying SMOTE on the entire training data set is really time-consuming. 

# Uncomment the following lines to evaluate effect of SMOTE on entire training data.

'''

model = LogisticRegression

auc, recall = kfold_cv(model, X_train_, y_train_, n_splits=10, smote=True, verbose=True)

print('AUC score: {}, recall: {}'.format(auc, recall))

'''
n_majority = np.arange(1, 50, 10) * y_train_pos.shape[0]

res = pd.DataFrame(data=np.zeros((len(n_majority), 3)), columns=['best_c', 'auc', 'recall'])

for i, n_major in enumerate(n_majority):

    undersample_X_train_, undersample_y_train_ = undersample(X_train_, y_train_, n_major)

    bestC, auc, recall = kfold_cv(Model, undersample_X_train_, undersample_y_train_, n_splits=5, smote=True, verbose=False)

    res.loc[i, 'best_c'] = bestC

    res.loc[i, 'auc'] = auc

    res.loc[i, 'recall'] = recall

    print('undersample number: {}, best C: {}'.format(n_major, bestC))

res['maj_class_num'] = n_majority

print(res)

fig = plt.figure(figsize=(16, 9))

plt.plot(res['maj_class_num'], res['auc'], 'b', label='AUC')

plt.plot(res['maj_class_num'], res['recall'], 'r', label='Recall')

plt.xlabel('maj_class_num')

plt.legend()

plt.grid()

plt.title('AUC and recall of logistic regression with balanced class weight trained on undersampled majority class of different numbers and smoted data')

plt.show()
bestmodel = LogisticRegression(class_weight='balanced', C=0.001)

undersample_X_train_, undersample_y_train_ = undersample(X_train_, y_train_, y_train_pos.shape[0])

bestmodel.fit(undersample_X_train_, undersample_y_train_)

y_pred = bestmodel.predict(X_test)

y_score = bestmodel.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_score, average='micro')

recall = recall_score(y_test, y_pred)

print('AUC: {}, recall: {}'.format(auc, recall))
bestmodel = LogisticRegression(class_weight='balanced', C=0.001)

undersample_X_train_, undersample_y_train_ = undersample(X_train_, y_train_, y_train_pos.shape[0])

sme = SMOTEENN(random_state=5)

X_res, y_res = sme.fit_sample(undersample_X_train_, undersample_y_train_)

bestmodel.fit(X_res, y_res)

y_pred = bestmodel.predict(X_test)

y_score = bestmodel.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_score, average='micro')

recall = recall_score(y_test, y_pred)

print('AUC: {}, recall: {}'.format(auc, recall))