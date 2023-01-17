# Global Variable

RANDOM_STATE = 42



# Library 



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from string import ascii_letters

import numpy as np

import pandas as pd

import seaborn as sns



from sklearn.metrics import roc_curve

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_score, recall_score



import matplotlib.pyplot as plt



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/pokemon/pokemon.csv')
df.head()
df.columns
df.shape
X = df.drop(columns = ['is_legendary'])

y = df['is_legendary']
X.shape, y.shape
df.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = RANDOM_STATE, stratify=y )
y_train.mean()
y_test.mean()
integer_columns = X.select_dtypes(include = ['integer']).columns.to_list()
integer_columns
baseline_model = LogisticRegression()
baseline_model.fit(X_train[integer_columns], y_train)
y_pred = baseline_model.predict_proba(X_test[integer_columns])
y_pred[:,1]
roc_auc_score( y_test, y_pred[:,1])
def fun_precision(pourcentage, predicted_proba, X, y):

    '''

    evaluate the precision on the dataset X

    Parameters

    ----------

    pourcentage : the percentage of the population to take into account when

        calculating the precision and the recall

    predicted_proba : the probability of being a fraudster for each observation

    X : the target variable

        numpy array

    y : the explaining features

        numpy array

    Return

    -------

    precision_score : the precision on the dataset

    '''



    ratio = -pourcentage * int(X.shape[0] / 100)

    ind = np.argpartition(predicted_proba, ratio)[ratio:]

    predicted = np.zeros(y.shape[0])

    predicted[ind] = 1

    return precision_score(y_true=y, y_pred=predicted)



def fun_recall(pourcentage, predicted_proba, X, y):

    '''

    evaluate the recall on the dataset X

    Parameters

    ----------

    pourcentage : the percentage of the population to take into account

        when calculating the precision and the recall

    predicted_proba : the probability of being a fraudster

        for each observation

    X : the target variable

        numpy array

    y : the explaining features

        numpy array

    Return

    -------

    recall_score : the recall on the dataset

    '''



    ratio = -pourcentage * int(X.shape[0] / 100)

    ind = np.argpartition(predicted_proba, ratio)[ratio:]

    predicted = np.zeros(y.shape[0])

    predicted[ind] = 1

    return recall_score(y_true=y, y_pred=predicted)
fun_precision(1, y_pred[:,1], X_test, y_test)
PRECISION = []

RECALL = []



for i in range(1,99):

    PRECISION.append(fun_precision(i, y_pred[:,1], X_test, y_test))

    RECALL.append(fun_recall(i, y_pred[:,1], X_test, y_test))

    
plt.plot(RECALL, PRECISION, markersize=1)

plt.xlabel('Precision')

plt.ylabel('Recall')
fpr, tpr ,_ = roc_curve(y_test, y_pred[:,1])
plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], color = 'red')

plt.xlabel('Frequency of false positive')

plt.ylabel('Frequency of true positive')
sns.set(style="white")



# Generate a large random dataset

rs = np.random.RandomState(33)

d = pd.DataFrame(data=rs.normal(size=(100, 26)),

                 columns=list(ascii_letters[26:]))



# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})