# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns



import pandas as pd

import numpy as np

from scipy.stats import skew

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib

import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor

import xgboost as xgb



import warnings

warnings.simplefilter('ignore')



from scipy import stats

from scipy.stats import norm, skew #for some statistics



import os

import pandas as pd

sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")

features = [c for c in train.columns if c not in ['ID_code', 'target']]
test.head()
test.describe()
train.head()
train.describe()
train.corr()
X, y = train.iloc[:,2:], train.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 123, stratify = y)
Y_test = test.drop(columns = ['ID_code'])
obs = train.isnull().sum().sort_values(ascending = False)

percent = round(train.isnull().sum().sort_values(ascending = False)/len(train)*100, 2)

pd.concat([obs, percent], axis = 1,keys= ['Number of Observations', 'Percent'])
target = train['target']

train = train.drop(["ID_code", "target"], axis=1)

sns.set_style('whitegrid')

sns.countplot(target)
plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(train[features].mean(axis=1),color="black", kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=1),color="red", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(train[features].mean(axis=0),color="black", kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=0),color="red", kde=True,bins=120, label='test')

plt.legend();plt.show()


plt.figure(figsize=(16,6))

plt.title("Distribution of std values per rows in the train and test set")

sns.distplot(train[features].std(axis=1),color="blue",kde=True,bins=120, label='train')

sns.distplot(test[features].std(axis=1),color="green", kde=True,bins=120, label='test')

plt.legend(); plt.show()


plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(train[features].mean(axis=0),color="blue", kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=0),color="green", kde=True,bins=120, label='test')

plt.legend();plt.show()
t0 = train.loc[target == 0]

t1 = train.loc[target == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per row in the train set")

sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].mean(axis=1),color="green", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
t0 = train.loc[target == 0]

t1 = train.loc[target == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per column in the train set")

sns.distplot(t0[features].mean(axis=0),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
t0 = train.loc[target == 0]

t1 = train.loc[target == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of standard deviation values per row in the train set")

sns.distplot(t0[features].std(axis=1),color="blue", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].std(axis=1),color="red", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
t0 = train.loc[target  == 0]

t1 = train.loc[target  == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of standard deviation values per column in the train set")

sns.distplot(t0[features].std(axis=0),color="blue", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].std(axis=0),color="red", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
t0 = train.loc[target == 0]

t1 = train.loc[target == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of skew values per row in the train set")

sns.distplot(t0[features].skew(axis=1),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].skew(axis=1),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
t0 = train.loc[target == 0]

t1 = train.loc[target == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of skew values per column in the train set")

sns.distplot(t0[features].skew(axis=0),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].skew(axis=0),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
t0 = train.loc[target == 0]

t1 = train.loc[target == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of kurtosis values per row in the train set")

sns.distplot(t0[features].kurtosis(axis=1),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].kurtosis(axis=1),color="green", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
t0 = train.loc[target == 0]

t1 = train.loc[target == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of kurtosis values per column in the train set")

sns.distplot(t0[features].kurtosis(axis=0),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].kurtosis(axis=0),color="green", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

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
logist_pred_test = logreg.predict_proba(Y_test)[:,1]

submit = test[['ID_code']]

submit['target'] = logist_pred_test

submit.head()
submit.to_csv('logreg_baseline.csv', index = False)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
X.columns
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
GaussianNB(priors=None)
gnb.score(X_train,y_train)
y_pred = gnb.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import train_test_split 

from sklearn import metrics 
clf = DecisionTreeClassifier()



clf = clf.fit(X_train,y_train)



y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)



cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=[5,2])



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)
cm
# Create Decision Tree classifer object

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))