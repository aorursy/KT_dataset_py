import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, roc_curve
%matplotlib inline
# Class 1 for fraudulent transactions

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.describe()
nonfraud, fraud = df.Class.value_counts()

print('Percentage of fraud transactions', fraud / (nonfraud + fraud))


sns.countplot('Class', data = df)

plt.gca().set_yscale('log')

df['Class'].value_counts()
fig, ax = plt.subplots(1, 2, figsize = (18, 4))



ax[0].hist(df[df.Class == 0]['Amount'], bins = 100, density = True, log = True, label = 'nonfraud')

ax[0].set_title('nonfraud')

ax[1].hist(df[df.Class == 1]['Amount'], bins = 100, density = True, log = True, label = 'fraud')

ax[1].set_title('fraud')

plt.show()
fig, ax = plt.subplots(1, 2, figsize = (18, 4))



ax[0].hist(df[df.Class == 0]['Time'], bins = 100, density = True, label = 'nonfraud')

ax[0].set_title('nonfraud')

ax[1].hist(df[df.Class == 1]['Time'], bins = 100, density = True, label = 'fraud')

ax[1].set_title('fraud')

plt.show()
plt.figure(figsize = (12, 8))

df_corr = df.corr()

sns.heatmap(df_corr, cmap = 'coolwarm_r')

plt.show()
# Correlation with class label sorted from largest to smallest

index_sorted = df_corr.sort_values('Class', ascending = False).index



fig, ax = plt.subplots(2, 5, figsize = (20, 8))

plt.subplots_adjust(hspace = 0.3, wspace = 0.3)



for i in range(0, 5):

    sns.boxplot(x = 'Class', y = index_sorted[i + 1], data = df, ax = ax[0][i])

    ax[0][i].set_title('%s vs %s (corr = %.3f)' % (index_sorted[i + 1], 'Class', df_corr['Class'][index_sorted[i + 1]]))



for i in range(0, 5):

    var = index_sorted[- (i + 1)]

    sns.boxplot(x = 'Class', y = var, data = df, ax = ax[1][i])

    ax[1][i].set_title('%s vs %s (corr = %.3f)' % (var, 'Class', df_corr['Class'][var]))

sns.distplot(df[df.Class == 1]['V17'], bins = 100, norm_hist = True, label = 'fraud', color = 'red')

sns.distplot(df[df.Class == 0]['V17'], bins = 100, norm_hist = True, label = 'nonfraud', color = 'blue')

plt.legend()

plt.show()
# 
y = df.Class

X = df.drop('Class', axis = 1) # axis = 1 means to drop a column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 27) # random_state is the seed for rand generator
# Try dummy classification for this highly imbalanced dataset

dummy = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)

dummy_pred = dummy.predict(X_test)
# The prediction of dummy should be all the same

np.unique(dummy_pred)
# accuracy = (true positive + true negative) / (total predictions made)

accuracy_score(y_test, dummy_pred)

# Seems good but it is due to the highly skewed the dataset
# Try logistic regression

lr = LogisticRegression(solver = 'liblinear').fit(X_train, y_train) # What is liblinear solver?

lr_pred = lr.predict(X_test)

lr_test_score = lr.predict_proba(X_test)[:, 1]

lr_train_score = lr.predict_proba(X_train)[:, 1]

accuracy_score(y_test, lr_pred)
# Check logistic regression does not give the same answer as dummy classifier

np.unique(lr_pred, return_counts = True)
# sklearn gives error since there is no positive prediction in the dummy classifier

precision_score(y_test, lr_pred), precision_score(y_test, dummy_pred)
recall_score(y_test, lr_pred), recall_score(y_test, dummy_pred)
# Compute precision and recall directly from confusion matrix

conf_mat = confusion_matrix(y_test, lr_pred)

true_neg, false_pos, false_neg, true_pos = conf_mat.ravel()

# precision, recall

true_pos / (true_pos + false_pos), true_pos / (true_pos + false_neg)
# Take a look at ROC curve

def plot_roc(y_true, y_score, label = 'none'):

    fp, tp, _ = roc_curve(y_true, y_score)

    plt.plot(fp, tp, label = label + ' (%.4f)' % np.trapz(tp, fp))



#fpr, tpr, thresholds = roc_curve(y_test, lr_scores)

#plt.plot(fpr, tpr, label = 'logistic regression (%.4f)' % np.trapz(tpr, fpr))

plot_roc(y_test, lr_test_score, 'lr test')

plot_roc(y_train, lr_train_score, 'lr train')

plt.xlabel('false positive')

plt.ylabel('true postive')

plt.legend(loc = 'lower right')

plt.title('ROC')

#plt.xlim(0, 0.1)