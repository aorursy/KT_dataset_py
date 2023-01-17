# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import data



train = pd.read_csv('/kaggle/input/glass/glass.csv')

print(train.shape)

train.head(10)
train.dtypes
train.describe()
train.nunique()
#Visualize the data

# Use seaborn to conduct heatmap to identify missing data

sns.heatmap(train.isnull(), cbar=False)
#Correlation between variables of the dataset



corr = train.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='Blues', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(train.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(train.columns)

ax.set_yticklabels(train.columns)

plt.show()

#Binary Logistic Regression: The target variable has only two possible outcomes such as Window or Non Window

train['Type'] = train['Type'].apply({1:0, 2:0, 3:0, 5:1, 6:1, 7:1}.get)



count_non_window = len(train[train['Type']==1])

count_window = len(train[train['Type']==0])

pct_of_non_window = count_non_window/(count_non_window+count_window)

print("percentage of non window glass is", pct_of_non_window*100)

pct_of_window = count_window/(count_non_window+count_window)

print("percentage of window glass", pct_of_window*100)
train.groupby('Type').mean()
import matplotlib.pyplot as plt

import numpy as np

fig = plt.figure()

ax1 = fig.add_subplot(111)



ax1.scatter(train['Na'].to_numpy(),train['Type'].to_numpy(),marker="s", label='Na')

ax1.scatter(train['Al'].to_numpy(),train['Type'].to_numpy(),marker="s", label='Al')

#ax1.scatter(train['Si'].to_numpy(),train['Type'].to_numpy(),marker="s", label='si')

ax1.scatter(train['Ba'].to_numpy(),train['Type'].to_numpy(),marker="s", label='Ba')

plt.title("Glass Type")

plt.xlabel('Elements Chosen')

plt.ylabel('1:Non-Window, 0:Window)')

plt.legend(loc='center right');

ax.figure.show()
#Experimented with the scatter plot to understand the Features , taking these features as these training dataset is close.

features = ['Na', 'Al', 'Ba']



X = train[features]

y = train['Type']
# Import module to split dataset

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
#Data available for Training

X_train.shape
# import the Model class

from sklearn.linear_model import LogisticRegression



# instantiate the model 

logistic = LogisticRegression(solver='lbfgs')



# Fit the logistic regression model.

logistic.fit(X_train,y_train)



# Get predictions 

y_predict = logistic.predict(X_test)
# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_predict)

cnf_matrix
n_samples = len(y_test)

print('Accuracy:  %.2f' % ((cnf_matrix[0][0] + cnf_matrix[1][1]) / n_samples))

print('Precision: %.2f' % (cnf_matrix[1][1] / (cnf_matrix[0][1] + cnf_matrix[1][1])))

print('Recall:    %.2f' % (cnf_matrix[1][1] / (cnf_matrix[1][0] + cnf_matrix[1][1])))
sns.heatmap(cnf_matrix,annot=True,cbar=False,cmap='Blues')

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.title('Confusion Matrix')

from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logistic.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logistic.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()