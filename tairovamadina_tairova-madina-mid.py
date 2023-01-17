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
import pandas as pd

test = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/train.csv")
train.head()
test.head()
train.shape,test.shape
train.dtypes.head()
test.dtypes.head()
test.isna().sum(), 
train.isna().sum()
train.describe()
test.describe()
train.nunique()
test.nunique()
train.target.value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(train.target)
train.std(axis = 0, skipna = True) 
train.corr()
test.corr()
features = train.columns.values[2:102]

plt.figure(figsize=(20,10))

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(train[features].mean(axis=0),color="black",kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=0),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(20,10))

plt.title("Distribution of std values per column in the train and test set")

sns.distplot(train[features].std(axis=0),color="black",kde=True,bins=120, label='train')

sns.distplot(test[features].std(axis=0),color="red", kde=True,bins=120, label='test')

plt.legend()

plt.show()
correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

correlations.head(10)
correlations.tail(10)
X = train.iloc[:, :-1].values

y = train.iloc[:, -1].values
y = train.target
#independent variables

X
#dependent variables column

y
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_classification

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from imblearn import over_sampling

from imblearn.over_sampling import SMOTE

from sklearn.metrics import f1_score

from sklearn.linear_model import LinearRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import recall_score
labelEncoderY = LabelEncoder()

y = labelEncoderY.fit_transform(y)
X = train.drop(['target', 'ID_code'],1)

y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

Y_test = test.drop(columns = ['ID_code'])



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)



print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
sm = SMOTE(random_state=42)

X_train_re, y_train_re = sm.fit_sample(X_train, y_train.ravel())
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_train_re, y_train_re, test_size = 0.2, random_state = 123, stratify = y_train_re)
from sklearn.linear_model import LogisticRegression

lrg = LogisticRegression()



lrg.fit(X_train_re, y_train_re)



y_pred_lrg = lrg.predict(X_train_re)

print(classification_report(y_train_re, y_pred_lrg))
lrg.score(X_train_re, y_train_re)
accuracy_score(y_train_re, y_pred_lrg)
logist_pred = lrg.predict_proba(X_test)[:,1]
logist_pred_test = lrg.predict_proba(Y_test)[:,1]

submit = test[['ID_code']]

submit['target'] = logist_pred_test

submit.head()
submit.to_csv('log_reg_baseline.csv', index = False)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train_re, y_train_re)
y_pred_gnb = gnb.predict(X_train_re)
recall_score(y_train_re, y_pred_gnb)
gnb_pred_test = gnb.predict_proba(Y_test)[:,1]

submit = test[['ID_code']]

submit['target'] = gnb_pred_test

submit.head()
submit.to_csv('NB_baseline.csv', index = False)
##Support Vector Machines (SVM)

##from sklearn.svm import SVC

##svm = SVC(kernel = 'linear')

##svm.fit(X_train_re, y_train_re)

##y_pred_svm = svm.predict(X_train_re)

##recall_score(y_train_re, y_pred_svm)
##RandomForestClassification

##from sklearn.ensemble import RandomForestClassifier

##rf = RandomForestClassifier(n_estimators=100)

##rf.fit(X_train, y_train)

##MuLtilayer Perceptron (MLP)

##from sklearn.neural_network import MLPClassifier

##mlp = MLPClassifier()

##mlp.fit(X_train_re, y_train_re)

##y_pred_mlp = svm.predict(X_test)