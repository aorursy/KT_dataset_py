import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/cat-in-the-dat/train.csv', index_col='id')

test = pd.read_csv('../input/cat-in-the-dat/test.csv', index_col='id')
train['ord_5_1'] = train['ord_5'].str[0]

train['ord_5_2'] = train['ord_5'].str[1]

train = train.drop('ord_5', axis=1)



test['ord_5_1'] = test['ord_5'].str[0]

test['ord_5_2'] = test['ord_5'].str[1]

test = test.drop('ord_5', axis=1)
# columns_to_test = list(test.columns)

columns_to_test = ['nom_7', 'nom_8', 'nom_9']



replace_xor = lambda x: 'xor' if x in xor_values else x



for column in columns_to_test:

    xor_values = set(train[column].unique()) ^ set(test[column].unique())

    if xor_values:

        print('Column', column, 'has', len(xor_values), 'XOR values')

        train[column] = train[column].apply(replace_xor)

        test[column] = test[column].apply(replace_xor)

    else:

        print('Column', column, 'has no XOR values')
y_train = train['target'].copy()

x_train = train.drop('target', axis=1)

del train



x_test = test.copy()

del test
train_part = len(x_train)

traintest = pd.concat([x_train, x_test])

df = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)

x_train = df[:train_part]

x_test = df[train_part:]

del df

del traintest
x_train = x_train.sparse.to_coo().tocsr()

x_test = x_test.sparse.to_coo().tocsr()
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score

from cross_validation_framework import *
logit_param_grid = {

    'C': [0.100, 0.150, 0.120, 0.125, 0.130, 0.135, 0.140, 0.145, 0.150]

}



logit_grid = GridSearchCV(LogisticRegression(solver='lbfgs'), logit_param_grid,

                          scoring='roc_auc', cv=5, n_jobs=-1, verbose=0)

logit_grid.fit(x_train, y_train)



best_C = logit_grid.best_params_['C']

# best_C = 0.12345



print('Best C:', best_C)
logit = LogisticRegression(C=best_C, solver='lbfgs', class_weight='balanced', max_iter=10000)

cv = KFold(n_splits=10, random_state=42)

oof, trained_estimators = fit(ScikitLearnPredictProbaEstimator(logit), roc_auc_score, x_train, y_train, cv)

y = predict(trained_estimators, x_test)
# From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



import matplotlib.pyplot as plt



from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels





def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
classes = pd.Series([0,1])

plot_confusion_matrix(y_train, oof.round(), classes=classes, normalize=True, title='Confusion matrix')

plt.show()
submission = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv', index_col='id')

submission['target'] = y

submission.to_csv('logit.csv')
submission.head()