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
import pandas as pd

import numpy as np
prefix_path = '/kaggle/input/titanic/'

df_tr_titanic = pd.read_csv('{}train.csv'.format(prefix_path))

df_te_titanic = pd.read_csv('{}test.csv'.format(prefix_path))

df_meta_titanic = pd.read_csv('{}gender_submission.csv'.format(prefix_path))

df_te_titanic = df_te_titanic.set_index('PassengerId').join(df_meta_titanic.set_index('PassengerId'), on='PassengerId')
print('Training set have {} records'.format(len(df_tr_titanic)))

print('Test set have {} records'.format(len(df_te_titanic)))
print('Training Info')

display(df_tr_titanic.info())

print('Test Info')

display(df_te_titanic.info())
print('Training Value')

display(df_tr_titanic['Cabin'])

print('Test Value')

display(df_te_titanic['Cabin'])
print('Training')

display(df_tr_titanic.isnull().any())

display(df_tr_titanic.isnull().sum().sum())

print('Test')

display(df_te_titanic.isnull().any())

display(df_te_titanic.isnull().sum().sum())
df_tr_selected = df_tr_titanic.loc[:, ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived')]

df_tr_selected.head()
df_te_selected = df_te_titanic.reset_index().loc[:, ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived')]

df_te_selected.head()
df_tr_selected['Age'] = df_tr_selected['Age'].fillna(df_tr_selected['Age'].mean())

print('Training')

display(df_tr_selected.isnull().any())

df_te_selected['Age'] = df_te_selected['Age'].fillna(df_te_selected['Age'].mean())

print('Test')

display(df_te_selected.isnull().any())
def range_normalization (x, bins,

            lower_infinite=True, upper_infinite=True,

            **kwargs):

    r"""Wrapper around pandas cut() to create infinite lower/upper bounds with proper labeling.



    Takes all the same arguments as pandas cut(), plus two more.



    Args :

        lower_infinite (bool, optional) : set whether the lower bound is infinite

            Default is True. If true, and your first bin element is something like 20, the

            first bin label will be '<= 20' (depending on other cut() parameters)

        upper_infinite (bool, optional) : set whether the upper bound is infinite

            Default is True. If true, and your last bin element is something like 20, the

            first bin label will be '> 20' (depending on other cut() parameters)

        **kwargs : any standard pandas cut() labeled parameters



    Returns :

        out : same as pandas cut() return value

        bins : same as pandas cut() return value

    """



    # Quick passthru if no infinite bounds

    if not lower_infinite and not upper_infinite:

        return pd.cut(x, bins, **kwargs)



    # Setup

    num_labels      = len(bins) - 1

    include_lowest  = kwargs.get("include_lowest", False)

    right           = kwargs.get("right", True)



    # Prepend/Append infinities where indiciated

    bins_final = bins.copy()

    if upper_infinite:

        bins_final.insert(len(bins),float("inf"))

        num_labels += 1

    if lower_infinite:

        bins_final.insert(0,float("-inf"))

        num_labels += 1



    # Decide all boundary symbols based on traditional cut() parameters

    symbol_lower  = "<=" if include_lowest and right else "<"

    left_bracket  = "(" if right else "["

    right_bracket = "]" if right else ")"

    symbol_upper  = ">" if right else ">="



    # Inner function reused in multiple clauses for labeling

    def make_label(i, lb=left_bracket, rb=right_bracket):

        return "{0}{1}, {2}{3}".format(lb, bins_final[i], bins_final[i+1], rb)



    # Create custom labels

    labels=[]

    for i in range(0,num_labels):

        new_label = None



        if i == 0:

            if lower_infinite:

                new_label = "{0} {1}".format(symbol_lower, bins_final[i+1])

            elif include_lowest:

                new_label = make_label(i, lb="[")

            else:

                new_label = make_label(i)

        elif upper_infinite and i == (num_labels - 1):

            new_label = "{0} {1}".format(symbol_upper, bins_final[i])

        else:

            new_label = make_label(i)



        labels.append(new_label)



    # Pass thru to pandas cut()

    return pd.cut(x, bins_final, labels=labels, **kwargs)
df_tr_selected['Normalized_age'] = range_normalization(df_tr_selected['Age'], [1, 5, 13, 18, 40, 60])
df_tr_selected['Normalized_age'].value_counts()
df_te_selected['Normalized_age'] = range_normalization(df_te_selected['Age'], [1, 5, 13, 18, 40, 60])

df_te_selected['Normalized_age'].value_counts()
df_tr_selected['Normalized_age'] = df_tr_selected['Normalized_age'].replace({

    '< 1': 0, '(1, 5]': 1, '(5, 13]':2, '(13, 18]':3, '(18, 40]':4, '(40, 60]':5, '> 60':6

})

df_tr_selected.head()
df_te_selected['Normalized_age'] = df_te_selected['Normalized_age'].replace({

    '< 1': 0, '(1, 5]': 1, '(5, 13]':2, '(13, 18]':3, '(18, 40]':4, '(40, 60]':5, '> 60':6

})

df_te_selected.head()
df_tr_tmp = pd.get_dummies(df_tr_selected['Sex'])

df_tr_selected = pd.concat([df_tr_selected, df_tr_tmp], axis=1)

df_tr_selected.head()
df_te_tmp = pd.get_dummies(df_te_selected['Sex'])

df_te_selected = pd.concat([df_te_selected, df_te_tmp], axis=1)

df_te_selected.head()
df_tr_selected = df_tr_selected.loc[:, ('Pclass', 'female', 'male', 'Normalized_age', 'SibSp', 'Parch', 'Survived')]

df_tr_selected.head()
df_te_selected = df_te_selected.loc[:, ('Pclass', 'female', 'male', 'Normalized_age', 'SibSp', 'Parch', 'Survived')]

df_te_selected.head()
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
# Select data

from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()

X_train_ori = df_tr_selected.loc[:, ('Pclass', 'female', 'male', 'Normalized_age', 'SibSp', 'Parch')].to_numpy()

X_test_ori = df_te_selected.loc[:, ('Pclass', 'female', 'male', 'Normalized_age', 'SibSp', 'Parch')].to_numpy()

y_train_ori = df_tr_selected['Survived'].to_numpy()

y_test_ori = df_te_selected['Survived'].to_numpy()

display(X_train_ori.shape, y_train_ori.shape)

display(X_test_ori.shape, y_test_ori.shape)
# merge data for cross validate 

X = np.concatenate((X_train_ori, X_test_ori))

y = np.concatenate((y_train_ori, y_test_ori))

display(X.shape, y.shape)
import numpy as np





def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    """

    given a sklearn confusion matrix (cm), make a nice plot



    Arguments

    ---------

    cm:           confusion matrix from sklearn.metrics.confusion_matrix



    target_names: given classification classes such as [0, 1, 2]

                  the class names, for example: ['high', 'medium', 'low']



    title:        the text to display at the top of the matrix



    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm

                  see http://matplotlib.org/examples/color/colormaps_reference.html

                  plt.get_cmap('jet') or plt.cm.Blues



    normalize:    If False, plot the raw numbers

                  If True, plot the proportions



    Usage

    -----

    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by

                                                              # sklearn.metrics.confusion_matrix

                          normalize    = True,                # show proportions

                          target_names = y_labels_vals,       # list of names of the classes

                          title        = best_estimator_name) # title of graph



    Citiation

    ---------

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



    """

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
def average_metrices(df, ignore_name=[]):

    col_name = df.head()

    for col in reversed(sorted(df.head())):

        if ignore_name[0] in col or ignore_name[1] in col:

            continue

        print('Avergae {}: {}'.format(col, df[col].mean()))
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

from sklearn.metrics import make_scorer, confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm

# from sklearn.model_selection import cross_val_score, KFold

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier



# calculate each class with tp, tn, fp, fn

def confusion(y_true, y_pred): return confusion_matrix(y_true, y_pred)



def cross_val(model, X, y, folds):

    metr = {'model':[], 'each class precision':[], 'each class recall':[], 'each class F1':[], 'Average Macro precision':[], 'Average Macro recall':[], 'Average Macro F1':[], 'Confusion Matrix':[]}

    kf = KFold(n_splits=folds)

    fold_round = 1

    for train_index, test_index in tqdm(kf.split(X)):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        confu = confusion_matrix(y_test, y_pred)

        score = precision_recall_fscore_support(y_test, y_pred)

        avg_score = precision_recall_fscore_support(y_test, y_pred, average='macro')

        metr['model'].append(str(type(model).__name__)+str(fold_round))

        metr['each class precision'].append(score[0])

        metr['each class recall'].append(score[1])

        metr['each class F1'].append(score[2])

        metr['Average Macro precision'].append(avg_score[0])

        metr['Average Macro recall'].append(avg_score[1])

        metr['Average Macro F1'].append(avg_score[2])

        metr['Confusion Matrix'].append(confu)

        fold_round += 1

#         metr[]

    return metr

# kfold = KFold(n_splits=10, random_state=0)

DT = DecisionTreeClassifier(random_state=0)

NB = GaussianNB()

MLP = MLPClassifier(random_state=0, max_iter=300)

RF = RandomForestClassifier(n_estimators=50, random_state=0)

VT = VotingClassifier(estimators=[('dt', DT), ('rf', RF), ('gnb', NB), ('mlp', MLP)], voting='hard')
dt_results = cross_val(DT,X=X,y=y,folds=10)

nb_results = cross_val(NB,X=X,y=y,folds=10)

mlp_results = cross_val(MLP,X=X,y=y,folds=10)

rf_results = cross_val(RF,X=X,y=y,folds=10)

vt_results = cross_val(VT,X=X,y=y,folds=10)
# if your want only score with cross validation, you can follow below:



# scoring = {'acc': 'accuracy',

#            'prec_macro': 'precision_macro',

#            'rec_macro': 'recall_macro',

#           'f1_macro': 'f1_macro',

#           'roc_auc': 'roc_auc'}

# score2 = {'acc': 'accuracy',

#            'prec_macro': 'precision_macro',

#            'rec_macro': 'recall_macro',

#           'f1_macro': 'f1_macro'}

# dt_results = cross_validate(DT,X=X,y=y,cv=10,scoring=scoring,return_train_score=True)

# nb_results = cross_validate(NB,X=X,y=y,cv=10,scoring=scoring,return_train_score=True)

# mlp_results = cross_validate(MLP,X=X,y=y,cv=10,scoring=scoring,return_train_score=True)

# rf_results = cross_validate(RF,X=X,y=y,cv=10,scoring=scoring,return_train_score=True)

# vt_results = cross_validate(VT,X=X,y=y,cv=10,scoring=score2,return_train_score=True)
dt_df = pd.DataFrame.from_dict(dt_results)

print('Decision Tree')

display(dt_df)

print(average_metrices(dt_df, ['model','each class']))

dt_df['Confusion Matrix'].mean()

plot_confusion_matrix(cm= dt_df['Confusion Matrix'].mean(), normalize= True,

                      target_names = ['Not Survived', 'Survived'],

                      title        = "Average Confusion Matrix")
nb_df = pd.DataFrame.from_dict(nb_results)

print('Naive Bayes')

display(nb_df)

print(average_metrices(nb_df, ['model','each class']))

plot_confusion_matrix(cm= nb_df['Confusion Matrix'].mean(), normalize= True,

                      target_names = ['Not Survived', 'Survived'],

                      title        = "Average Confusion Matrix")
mlp_df = pd.DataFrame.from_dict(mlp_results)

print('Multilayer Perceptron')

display(mlp_df)

print(average_metrices(mlp_df, ['model','each class']))

plot_confusion_matrix(cm= mlp_df['Confusion Matrix'].mean(), normalize= True,

                      target_names = ['Not Survived', 'Survived'],

                      title        = "Average Confusion Matrix")
rf_df = pd.DataFrame.from_dict(rf_results)

print('Random Forest')

display(rf_df)

print(average_metrices(rf_df, ['model','each class']))

plot_confusion_matrix(cm=rf_df['Confusion Matrix'].mean(), normalize= True,

                      target_names = ['Not Survived', 'Survived'],

                      title        = "Average Confusion Matrix")
vt_df = pd.DataFrame.from_dict(vt_results)

print('Voting Classifier')

display(vt_df)

print(average_metrices(vt_df, ['model','each class']))

plot_confusion_matrix(cm= vt_df['Confusion Matrix'].mean(), normalize= True,

                      target_names = ['Not Survived', 'Survived'],

                      title        = "Average Confusion Matrix")