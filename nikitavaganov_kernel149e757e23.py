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
# Read data

data = pd.read_json('/kaggle/input/scientific-paper-russian-language/dataset3658_70.json')

data.head()
data = data.drop(data[data['isGood'].eq(0)].sample(1396).index)

data.groupby('isGood').count()['paperPath']
# Remove columns

X = data.drop(['paperPath', 'paperUrl', 'paperTitle', 'journalName', 'journalViews', 'journalDownloads', 'journalHirch', 'isGood'], axis=1)

y = data.isGood
# Visualise features distribution in dataset

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

METAFEATURES = ['keywordsLvl', 'WaterLvl', 'deviation', 'polarity', 'subjectivity', 

'formalScore', 'pronounceScore', 'length', 'lexicalDiversity']

GOOD_PAPERS = data['isGood'] == 1



fig, axes = plt.subplots(ncols=1, nrows=len(METAFEATURES), figsize=(20, 25), dpi=100)



for i, feature in enumerate(METAFEATURES):

    sns.distplot(data.loc[~GOOD_PAPERS][feature], label='Статья не удовлетворяет требованиям', ax=axes[i], color='red')

    sns.distplot(data.loc[GOOD_PAPERS][feature], label='Статья удовлетворяет требованиям', ax=axes[i], color='green')

    axes[i].set_xlabel('')

    axes[i].tick_params(axis='x', labelsize=16)

    axes[i].tick_params(axis='y', labelsize=16)

    axes[i].legend()

    

    axes[i].set_title(f'Распределение критерия {feature} в обучающей выборке', fontsize=18)



plt.show()
# D'Agostino and Pearson's Test

from scipy.stats import normaltest, shapiro, anderson

# normality test

a = data['lexicalDiversity']

stat, p = shapiro(a)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

	print('Sample looks Gaussian (fail to reject H0)')

else:

	print('Sample does not look Gaussian (reject H0)')



stat, p = normaltest(a)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

	print('Sample looks Gaussian (fail to reject H0)')

else:

	print('Sample does not look Gaussian (reject H0)')

    

result = anderson(a)

print('Statistic: %.3f' % result.statistic)

p = 0

for i in range(len(result.critical_values)):

	sl, cv = result.significance_level[i], result.critical_values[i]

	if result.statistic < result.critical_values[i]:

		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))

	else:

		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
print(X.median())

print(X.std())
# Data distribution relative to the target feature 'isGood' (the one that the model will predict later)

data.groupby('isGood').count()['paperPath']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# transform data

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train_scale = scaler.fit_transform(X_train)

X_test_scale = scaler.transform(X_test)

# split training feature and target sets into training and validation subsets

from sklearn.model_selection import train_test_split



X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train_scale, y_train, random_state=0)

# import machine learning algorithms

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
X_test
# train with Gradient Boosting algorithm

# compute the accuracy scores on train and validation sets when training with different learning rates



learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in learning_rates:

    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)

    gb.fit(X_train_sub, y_train_sub)

    print("Learning rate: ", learning_rate)

    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train_sub, y_train_sub)))

    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation_sub, y_validation_sub)))

    print()
gb.feature_importances_
# Output confusion matrix and classification report of Gradient Boosting algorithm on validation set



gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 5, random_state = 0)

gb.fit(X_train_sub, y_train_sub)

predictions = gb.predict(X_validation_sub)



print("Confusion Matrix:")

print(confusion_matrix(y_validation_sub, predictions))

print()

print("Classification Report")

print(classification_report(y_validation_sub, predictions))
# ROC curve and Area-Under-Curve (AUC)



y_scores_gb = gb.decision_function(X_validation_sub)

fpr_gb, tpr_gb, _ = roc_curve(y_validation_sub, y_scores_gb)

roc_auc_gb = auc(fpr_gb, tpr_gb)



print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))
# Export model

import pickle

pickle.dump(gb, open('model.pkl', 'wb'))
import matplotlib.pyplot as plt

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('Фактический класс')

    plt.xlabel('Спрогнозированный класс')



font = {'size' : 15}



plt.rc('font', **font)



cnf_matrix = confusion_matrix(y_validation_sub, predictions)

plt.figure(figsize=(10, 8))

plot_confusion_matrix(cnf_matrix, classes=['Не удовл. требованиям', 'Удовл. требованиям'],

                      title='Матрица ошибок')

plt.savefig("conf_matrix.png")

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1.5)

sns.set_color_codes("muted")



plt.figure(figsize=(10, 8))

fpr, tpr, thresholds = roc_curve(y_validation_sub, y_scores_gb)

lw = 2

plt.plot(fpr, tpr, lw=lw, label='ROC curve ')

plt.plot([0, 1], [0, 1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('TPR')

plt.ylabel('FPR')

plt.title('Кривая ошибок')

plt.savefig("ROC.png")

plt.show()
fpr
from sklearn.metrics import precision_recall_curve, classification_report

report = classification_report(y_test, gb.predict(X_validation_sub), target_names=['Не удовл. требованиям', 'Удовл. требованиям'])

print(report)
import matplotlib.pyplot as plt



(pd.Series(gb.feature_importances_, index=X.columns)

   .sort_values()

   .plot(kind='barh')) 
X.drop(['keywordsLvl', 'WaterLvl', 'deviation'], axis=1).corr()