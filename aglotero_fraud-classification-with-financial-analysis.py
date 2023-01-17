# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

import os

import h2o

import numpy as np

%matplotlib inline  



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = reviews = pd.read_csv('../input/creditcard.csv')

data['Class'] = data['Class'].astype('category')

cols = list(data.columns)
data.groupby('Class')['Class'].count()
h2o.init()
creditcard = h2o.upload_file(path ='../input/creditcard.csv')
creditcard['Class'] = creditcard['Class'].asfactor()
df = creditcard.as_data_frame()
cols = list(df.columns)
df.isnull().sum()
df.query('Amount == 0').groupby('Class').count()
df.query('Amount != 0').to_csv('creditcard_amount_positive.csv')

creditcard = h2o.upload_file(path ='creditcard_amount_positive.csv')

creditcard['Class'] = creditcard['Class'].asfactor()

df = creditcard.as_data_frame()
num_cols = len(cols[:-1])

plt.figure(figsize=(12,num_cols*4))

gs = gridspec.GridSpec(num_cols, 1)

for i, cn in enumerate(df[cols[:-1]]):

    ax = plt.subplot(gs[i])

    sns.distplot(df[cn][df.Class == 1], bins=50)

    sns.distplot(df[cn][df.Class == 0], bins=50)

    ax.set_xlabel('')

    ax.set_title('histogram of feature: ' + str(cn))

plt.show()
from scipy import stats

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

variables = cols[:-1]

keep = []

p_value_alpha = 0.05 #defult p-value for statistical significance



for variable in variables:

    fraud_v = df[variable][df.Class == 1]

    not_fraud_v = df[variable][df.Class == 0].sample(len(fraud_v))

    p_value = stats.ttest_ind(not_fraud_v, fraud_v).pvalue

    if p_value >= p_value_alpha:

        print("Distributions are equal. Discard {} variable".format(variable))

    else:

        print("Distributions are diferent. Keep {} variable".format(variable))

        keep.append(variable)
from h2o.estimators.random_forest import H2ORandomForestEstimator

train, valid = creditcard.split_frame(ratios=[0.7])

response_var = 'Class'

features = [col for col in cols if col != response_var]

naive_rf_model = H2ORandomForestEstimator()

naive_rf_model.train(x=features, y=response_var, training_frame=train, validation_frame=valid)

performance_train = naive_rf_model.model_performance(train=True)
# for metrics

import itertools

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
preds = naive_rf_model.predict(train)

cm = confusion_matrix(train.as_data_frame()['Class'], preds.as_data_frame()['predict'])

plot_confusion_matrix(cm, ['Não-Fraude', 'Fraude'], False)
fpr, tpr, threshold = roc_curve(train.as_data_frame()['Class'], preds.as_data_frame()['predict'])

plt.plot(fpr, tpr, color='darkorange',

         lw=2, label='ROC curve')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([-0.04, 1.0])

plt.ylim([-0.04, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Curva ROC')

plt.legend(loc="lower right")

plt.show()
predictions = naive_rf_model.predict(valid)

cm = confusion_matrix(valid.as_data_frame()['Class'], predictions.as_data_frame()['predict'])

plot_confusion_matrix(cm, ['Não-Fraude', 'Fraude'], False)
fpr, tpr, threshold = roc_curve(valid.as_data_frame()['Class'], predictions.as_data_frame()['predict'])

plt.plot(fpr, tpr, color='darkorange',

         lw=2, label='ROC curve')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([-0.04, 1.0])

plt.ylim([-0.04, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Curva ROC')

plt.legend(loc="lower right")

plt.show()
valid_class = valid['Class'].as_data_frame()

valid_amount = valid['Amount'].as_data_frame()

take_rate = 0.1
valid_data = pd.concat([predictions.as_data_frame(), valid_amount, valid_class], axis=1)
total = valid_data.groupby('Class')['Amount'].sum()

print("Fraud: {:06.2f} Gross profit: {:06.2f} Net: {:06.2f}".format(total[1], total[0] * take_rate, (total[0] * take_rate) - total[1]))
def correct_predict(row):

    if row['Class'] == row['predict'] and row['predict'] == 0:

        return row['Amount'] * take_rate

    elif row['Class'] == row['predict'] and row['predict'] == 1:

        return -row['Amount']

    return 0



def missed_profit(row):

    if row['Class'] != row['predict'] and row['predict'] == 0:

        return -row['Amount'] * take_rate

    else:

        return 0

    

def missed_loss(row):

    if row['Class'] != row['predict'] and row['predict'] == 1:

        return -row['Amount']

    return 0



valid_data['correct_predict'] = valid_data.apply(lambda row: correct_predict(row), axis=1)

valid_data['missed_profit'] = valid_data.apply(lambda row: missed_profit(row), axis=1)

valid_data['missed_loss'] = valid_data.apply(lambda row: missed_loss(row), axis=1)
avoided_loss = valid_data.query('correct_predict < 0')['correct_predict'].sum()

corrected_no_fraud = valid_data.query('correct_predict > 0')['correct_predict'].sum()

missed_profit = valid_data.query('missed_profit < 0')['missed_profit'].sum()

missed_loss = valid_data.query('missed_loss < 0')['missed_loss'].sum()
pd.DataFrame([[-avoided_loss, -missed_profit, -missed_loss, corrected_no_fraud]], 

             columns=['avoided loss', 'missed profit', 'missed loss', 'net'])
print("An increase of ${:06.2f} in the net profit".format(754372.79 - 737697.15))