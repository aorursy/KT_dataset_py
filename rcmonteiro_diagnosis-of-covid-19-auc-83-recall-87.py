# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn import preprocessing 

from collections import Counter

from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc,recall_score,precision_score

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_validate 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pd.set_option("display.max_columns", 111)

sns.set_palette("Set2")
df = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")

df.head()
df.shape
df.isna().sum()
df.columns.tolist()
df['Patient ID'].nunique()
df.drop(columns='Patient ID', inplace=True)
print('Negative: {} ({}%)'.format(df['SARS-Cov-2 exam result'].value_counts()[0], round(df['SARS-Cov-2 exam result'].value_counts()[0]/len(df)*100, 2)))

print('Positive: {} ({}%)'.format(df['SARS-Cov-2 exam result'].value_counts()[1], round(df['SARS-Cov-2 exam result'].value_counts()[1]/len(df)*100, 2)))

sns.countplot('SARS-Cov-2 exam result',data=df)
df[['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)']].isna().sum()
df['Patient age quantile'].describe()
df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].map({'positive': 1, 'negative': 0})
drop_index = []

for i in range(df.shape[1]):

    if df.iloc[:,i].isna().sum() == len(df):

        drop_index.append(df.iloc[:,i].name)

        

for j in drop_index:

    df = df.drop([j],axis=1)
df = df.dropna(thresh=0.20*len(df), axis=1)
df = df.dropna(axis=0)
df.shape
df.dtypes.value_counts()
df.select_dtypes(['float64','object','int64']).apply(pd.Series.nunique, axis = 0)
categorical_variables = df.select_dtypes(['object'])

categorical_variables = categorical_variables.columns

categorical_variables.tolist()
for i in categorical_variables:

    le = preprocessing.LabelEncoder()

    le.fit(df[i].values)

    df[i] = le.transform(df[i].values)
#Correlation between features

corr = df.corr('pearson')



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
X = df.loc[:, df.columns != 'SARS-Cov-2 exam result']

y = df['SARS-Cov-2 exam result']



bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)



featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  

featureScores = featureScores.nlargest(10, 'Score')



sns.barplot(x="Score", y="Specs", data=featureScores)

featureScores
print('Negative: {} ({}%)'.format(df['SARS-Cov-2 exam result'].value_counts()[0], round(df['SARS-Cov-2 exam result'].value_counts()[0]/len(df)*100, 2)))

print('Positive: {} ({}%)'.format(df['SARS-Cov-2 exam result'].value_counts()[1], round(df['SARS-Cov-2 exam result'].value_counts()[1]/len(df)*100, 2)))

sns.countplot('SARS-Cov-2 exam result',data=df)
rus = RandomUnderSampler(random_state=42)

X_res, y_res = rus.fit_resample(X, y)

print('Resampled dataset shape %s' % Counter(y_res))
print('Negative: {} ({}%)'.format(y_res.value_counts()[0], round(y_res.value_counts()[0]/len(y_res)*100, 2)))

print('Positive: {} ({}%)'.format(y_res.value_counts()[1], round(y_res.value_counts()[1]/len(y_res)*100, 2)))

sns.countplot(y_res)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.1, random_state=42,)
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 1, 1.25, 1.5, 1.75, 2]



for learning_rate in lr_list:

    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)

    gb_clf.fit(X_train, y_train)



    print("Learning rate: ", learning_rate)

    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))

    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
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
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=10, max_depth=2, random_state=0)

gb_clf2.fit(X_train, y_train)

predictions = gb_clf2.predict(X_test)



plot_confusion_matrix(confusion_matrix(y_test, predictions), target_names=['0', '1'], normalize=False)



print(classification_report(y_test, predictions))



scores = cross_validate(gb_clf2, X_train, y_train, cv=45, scoring=['precision','recall','roc_auc'])



print("Cross Validation Scores: ")

print('Precision: ', scores.get('test_precision').mean())

print('Recall: ', scores.get('test_recall').mean())

print('ROC_ACU: ', scores.get('test_roc_auc').mean())