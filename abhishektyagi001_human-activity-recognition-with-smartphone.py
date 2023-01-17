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
from __future__ import print_function

import os

#Data Path has to be set as per the file location in your system

#data_path = ['..', 'data']

data_path = ['../input/human-activity-recognition-with-smartphones/test.csv']
import pandas as pd

import numpy as np

#The filepath is dependent on the data_path set in the previous cell 

filepath = os.sep.join(data_path)

data = pd.read_csv(filepath, sep=',')
data.dtypes.value_counts()
data.dtypes.tail()
#The data are all scaled from -1 (minimum) to 1.0 (maximum).

data.iloc[:, :-1].min().value_counts()
data.iloc[:, :-1].max().value_counts()
#Examine the breakdown of activities--they are relatively balanced.

data.Activity.value_counts()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

data['Activity'] = le.fit_transform(data.Activity)

data['Activity'].sample(5)
"""Question 2

Calculate the correlations between the dependent variables.

Create a histogram of the correlation values

Identify those that are most correlated (either positively or negatively)."""

# Calculate the correlation values

feature_cols = data.columns[:-1]

corr_values = data[feature_cols].corr()



# Simplify by emptying all the data below the diagonal

tril_index = np.tril_indices_from(corr_values)



# Make the unused values NaNs

for coord in zip(*tril_index):

    corr_values.iloc[coord[0], coord[1]] = np.NaN

    

# Stack the data and convert to a data frame

corr_values = (corr_values.stack().to_frame().reset_index().rename(columns={'level_0':'feature1','level_1':'feature2',0:'correlation'}))



# Get the absolute values for sorting

corr_values['abs_correlation'] = corr_values.correlation.abs()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set_context('talk')

sns.set_style('white')

sns.set_palette('dark')



ax = corr_values.abs_correlation.hist(bins=50)



ax.set(xlabel='Absolute Correlation', ylabel='Frequency');
# The most highly correlated values

corr_values.sort_values('correlation', ascending=False).query('abs_correlation>0.8')
"""Split the data into train and test data sets. This can be done using any method, but consider using Scikit-learn's StratifiedShuffleSplit to maintain the same ratio of predictor classes.

Regardless of methods used to split the data, compare the ratio of classes in both the train and test splits."""

from sklearn.model_selection import StratifiedShuffleSplit



# Get the split indexes

#strat_shuf_split = StratifiedShuffleSplit(n_splits=1,test_size=0.3, random_state=42)



#train_idx, test_idx = next(strat_shuf_split.split(data[feature_cols], data.Activity))



# Create the dataframes

X_train = data.loc[train_idx, feature_cols]

y_train = data.loc[train_idx, 'Activity']



X_test  = data.loc[test_idx, feature_cols]

y_test  = data.loc[test_idx, 'Activity']
"""Fit a logistic regression model without any regularization using all of the features. Be sure to read the documentation about fitting a multi-class model so you understand the coefficient output. Store the model.

Using cross validation to determine the hyperparameters, fit models using L1, and L2 regularization. Store each of these models as well. Note the limitations on multi-class models, solvers, and regularizations. The regularized models, in particular the L1 model, will probably take a while to fit."""

from sklearn.linear_model import LogisticRegression



# Standard logistic regression

lr = LogisticRegression().fit(X_train, y_train)
from sklearn.linear_model import LogisticRegressionCV



# L1 regularized logistic regression

lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear').fit(X_train, y_train)

#Try with different solvers like ‘newton-cg’, ‘lbfgs’, ‘sag’, ‘saga’ and give your observations
# L2 regularized logistic regression

lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2').fit(X_train, y_train)
# Combine all the coefficients into a dataframe

coefficients = list()



coeff_labels = ['lr', 'l1', 'l2']

coeff_models = [lr, lr_l1, lr_l2]



for lab,mod in zip(coeff_labels, coeff_models):

    coeffs = mod.coef_

    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]], 

                                 labels=[[0,0,0,0,0,0], [0,1,2,3,4,5]])

    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))



coefficients = pd.concat(coefficients, axis=1)



coefficients.sample(10)
fig, axList = plt.subplots(nrows=3, ncols=2)

axList = axList.flatten()

fig.set_size_inches(10,10)





for ax in enumerate(axList):

    loc = ax[0]

    ax = ax[1]

    

    data = coefficients.xs(loc, level=1, axis=1)

    data.plot(marker='o', ls='', ms=2.0, ax=ax, legend=False)

    

    if ax is axList[0]:

        ax.legend(loc=4)

        

    ax.set(title='Coefficient Set '+str(loc))



plt.tight_layout()

"""Predict and store the class for each model.

Also store the probability for the predicted class for each model."""

# Predict the class and the probability for each



y_pred = list()

y_prob = list()



coeff_labels = ['lr', 'l1', 'l2']

coeff_models = [lr, lr_l1, lr_l2]



for lab,mod in zip(coeff_labels, coeff_models):

    y_pred.append(pd.Series(mod.predict(X_test), name=lab))

    y_prob.append(pd.Series(mod.predict_proba(X_test).max(axis=1), name=lab))

    

y_pred = pd.concat(y_pred, axis=1)

y_prob = pd.concat(y_prob, axis=1)



y_pred.head()

y_prob.head()
"""For each model, calculate the following error metrics:



accuracy

precision

recall

fscore

confusion matrix

Decide how to combine the multi-class metrics into a single value for each model."""

from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

from sklearn.preprocessing import label_binarize



metrics = list()

cm = dict()



for lab in coeff_labels:



    # Preciision, recall, f-score from the multi-class support function

    precision, recall, fscore, _ = score(y_test, y_pred[lab], average='weighted')

    

    # The usual way to calculate accuracy

    accuracy = accuracy_score(y_test, y_pred[lab])

    

    # ROC-AUC scores can be calculated by binarizing the data

    auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),

              label_binarize(y_pred[lab], classes=[0,1,2,3,4,5]), 

              average='weighted')

    

    # Last, the confusion matrix

    cm[lab] = confusion_matrix(y_test, y_pred[lab])

    

    metrics.append(pd.Series({'precision':precision, 'recall':recall, 

                              'fscore':fscore, 'accuracy':accuracy,

                              'auc':auc}, 

                             name=lab))



metrics = pd.concat(metrics, axis=1)

#Run the metrics

metrics


fig, axList = plt.subplots(nrows=2, ncols=2)

axList = axList.flatten()

fig.set_size_inches(12, 10)



axList[-1].axis('off')



for ax,lab in zip(axList[:-1], coeff_labels):

    sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d');

    ax.set(title=lab);

    

plt.tight_layout()

#Display or plot the confusion matrix for each model.
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import VarianceThreshold



#threshold with .7



sel = VarianceThreshold(threshold=(.7 * (1 - .7)))



data2 = pd.concat([X_train,X_test])

data_new = pd.DataFrame(sel.fit_transform(data2))





data_y = pd.concat([y_train,y_test])



from sklearn.model_selection import train_test_split



X_new,X_test_new = train_test_split(data_new)

Y_new,Y_test_new = train_test_split(data_y)