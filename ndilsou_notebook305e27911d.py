# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

import statsmodels.formula as smf

import scipy



from matplotlib import pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/HR_comma_sep.csv')

print('\ninfo')

print(df.info())

print('\nN/A count')

print(df.isnull().sum())
df.head()
from sklearn.model_selection import train_test_split

frac = 0.25

train, test = train_test_split(df, test_size = frac)

train_numeric = train.loc[:,(train.dtypes != np.object)]

train_numeric.columns
train.describe()
train.describe(include=['O'])
for col in train.loc[:,train.dtypes == np.object]:

    print(col, train[col].unique())
left_frac = train.groupby('left')['left'].agg(lambda x: len(x)/len(train))



print('fraction of employees that left:')

print(left_frac)
# Let's have a look at the amplitude of the correlations.

corr_pearson = train_numeric.corr('pearson')

corr_spearman = train_numeric.corr('spearman')
def corr_heatmap(corr, kwargs={}):           

    sns.heatmap(corr, **kwargs)

plt.figure()

plt.subplot(211)

plt.title('Pearson correlation')

corr_heatmap(corr_pearson, kwargs=dict(xticklabels=[]))

plt.subplot(212)

plt.title('Spearman rank correlation')

corr_heatmap(corr_spearman, kwargs=dict(annot=False))

plt.show()
corr_heatmap((corr_spearman - corr_pearson).abs())
plt.figure()

plt.title('Salary')

sns.countplot(train['salary'])

plt.figure()

plt.title('Department')

plt.xticks(rotation='vertical')

sns.countplot(train['sales'])
plt.figure()

plt.title('Salary (Left/Stayed)')

sns.countplot(train['salary'], hue=train['left'])

plt.figure()

plt.title('Department (Left/Stayed)')

plt.xticks(rotation='vertical')

sns.countplot(train['sales'], hue=train['left'])
#TODO: Convert to countplot.

columns = train.columns.values

nb = len(columns)

row, col = 4, 2

k = 0

gb = train.groupby('left')

for col in train_numeric:

    plt.figure()

    color = ['b', 'r']

    alpha = [0.8, 0,5]

    for i, g in gb:

        if train[col].dtypes == np.object:

            g[col].plot.bar(title=col, alpha=alpha[i])

        else:

            g[col].plot.hist(title=col, alpha=alpha[i])

    #train.loc[train.left==0,col].plot.hist(title=col)
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, precision_recall_curve

from sklearn.metrics import confusion_matrix, classification_report, average_precision_score

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
def roc_plot(y_true, y_pred):

    fpr, tpr, _ = roc_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr, reorder=True)

    lw = 2

    plt.figure()

    plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    plt.show()

    return roc_auc



def precision_recall_plot(y_true, y_pred):

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    average_precision = average_precision_score(y_true, y_pred)

    lw = 2

    plt.figure()

    plt.plot(recall, precision, lw=2, label='Precision-Recall curve', color='navy')

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.ylim([0.0, 1.05])

    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall example: AUC={0:0.4f}'.format(average_precision))

    plt.legend(loc="lower left")

    plt.show()

    return average_precision_score
df_dummy = pd.get_dummies(df, drop_first=True)

train, test = train_test_split(df_dummy, test_size = frac)

Y_train = train['left']

X_train = train.drop('left', axis=1)

Y_test = test['left']

X_test = test.drop('left', axis=1)



performances = dict()
model_logit = LogisticRegressionCV().fit(X_train, Y_train)
model_dtree = DecisionTreeClassifier().fit(X_train, Y_train)
model_rforest = RandomForestClassifier().fit(X_train, Y_train)
#We standardize the data before feeding it to the neuralnet

from sklearn.preprocessing import StandardScaler, Normalizer

scaler = StandardScaler()



scaler.fit(X_train)

X_train_nn = scaler.transform(X_train)

X_test_nn = scaler.transform(X_test)
import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.regularizers import WeightRegularizer

from keras.wrappers.scikit_learn import KerasClassifier
def create_keras_model():

    model = Sequential()

    model.add(Dense(output_dim=100, input_dim=X_train_nn.shape[1]))

    model.add(Activation("relu"))

    model.add(Dropout(0.5))

    model.add(Dense(output_dim=1))

    model.add(Activation("sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'recall'])

    return model
model_mlp = create_keras_model()

h = model_mlp.fit(X_train_nn, Y_train, nb_epoch=20, batch_size=200)
def test_results(model, X_test, Y_test, is_keras=False):

    Y_pred_binary = model.predict_classes(X_test) if is_keras else model.predict(X_test)

    Y_pred = model.predict_proba(X_test) if is_keras else model.predict_proba(X_test)[:,1]



    acc = accuracy_score(Y_test, Y_pred_binary)

    cm = confusion_matrix(Y_test, Y_pred_binary)/Y_test.count()

    

    print("accuracy :", acc)

    print("confusion matrix :\n", cm)

    print(classification_report(Y_test, Y_pred_binary))

    

    roc_auc = roc_plot(Y_test, Y_pred)

    average_precision = precision_recall_plot(Y_test, Y_pred)

    return roc_auc, average_precision





    
performances['MLP'] = test_results(model_mlp, X_test_nn, Y_test, is_keras=True)
performances['LogitCV'] = test_results(model_logit, X_test, Y_test)
performances['DTree'] = test_results(model_dtree, X_test, Y_test)
importances = model_dtree.feature_importances_

importances = pd.Series(importances, index=X_test.columns)

importances.sort(inplace=True, ascending=False)

importances.plot.bar()
performances['RForest'] = test_results(model_rforest, X_test, Y_test)
importances = model_rforest.feature_importances_

importances = pd.Series(importances, index=X_test.columns)

importances.sort(inplace=True, ascending=False)

std = np.std([tree.feature_importances_ for tree in model_rforest.estimators_],

             axis=0)

std = pd.Series(std, index=X_test.columns)

print("Feature ranking:")



for i, f in enumerate(importances.index):

    print("%d. feature %s (%f)" % (i + 1, f, importances[f]))



# Plot the feature importances of the forest

importances.plot.bar(yerr=std)