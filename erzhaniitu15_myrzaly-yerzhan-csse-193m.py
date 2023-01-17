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

        from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report



import matplotlib.pyplot as plt

from IPython.display import display

import seaborn as sns



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

sample = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")
from sklearn.model_selection import train_test_split

train.info()
test.info()
test.head()
sample.info()
sample.head()
train32 = train.drop(['ID_code', 'target'], axis = 1).astype('float32')
train32.info()
train.head(2)
train32.head(2)
X_train = train.iloc[:, 2:].values

Y_train = train.target.values

X_test = test.iloc[:, 1:].values
X_test
X_train.shape
Y_train.shape
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(Y_train,predictions))
submission_log = pd.DataFrame({'ID_code':test.ID_code.values})

submission_log['target'] = predictions

submission_log.to_csv('submission_logreg.csv', index=False)
train.head()
train.describe()
test.head()
test.describe()
X_train = train.iloc[:, train.columns != 'target'].values

Y_train = train.iloc[:, 1].values

X_test = test.values
train.describe()
train.target.value_counts() 
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()





X_train[:,0] = le.fit_transform(X_train[:,0])

X_test[:,0] = le.fit_transform(X_test[:,0])

knn = KNeighborsClassifier(11)
knn.fit(X_train, Y_train)
y_preds = knn.predict(X_test)
y_preds
pd.concat([test.ID_code, pd.Series(y_preds).rename('target')], axis = 1).to_csv('submission_knn_fix.csv', index =False)
from sklearn.naive_bayes import GaussianNB
features = [x for x in train.columns if 'var_' in x]
nv = GaussianNB()

nv.fit(train[features], train['target'])
test['target'] = nv.predict_proba(test[features])[:, 1]

test[['ID_code', 'target']].to_csv('submission_GaussianNV.csv', index=False)
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

train.isnull().values.any()
features = train.drop(['ID_code','target'], axis=1).values

targets = train.target.values
ax = sns.countplot(x = targets ,palette="Set2")

sns.set(font_scale=1.5)

ax.set_xlabel(' ')

ax.set_ylabel(' ')

fig = plt.gcf()

fig.set_size_inches(10,5)

ax.set_ylim(top=700000)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))



plt.title('Distribution of Targets')

plt.xlabel('Initiation of Customer Transaction for Next Year')

plt.ylabel('Frequency [%]')

plt.show()
sns.set(style="white")





# Compute the correlation matrix

corr = train.corr()





# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
kfold = 5

skf = StratifiedKFold(n_splits=kfold, random_state=42)

#Set params

params = {

    'min_child_weight': 10.0,

    'objective': 'binary:logistic',

    'max_depth': 7,

    'max_delta_step': 1.8,

    'colsample_bytree': 0.4,

    'subsample': 0.8,

    'eta': 0.025,

    'gamma': 0.65,

    'num_boost_round' : 700

    }
X = train.drop(['ID_code', 'target'], axis=1).values

y = train.target.values

test_id = test.ID_code.values

test = test.drop('ID_code', axis=1)
submission = pd.DataFrame()

submission['ID_code'] = test_id

submission['target'] = np.zeros_like(test_id)
submission.to_csv('submission_XGBoostSKFold.csv', index=False)
y = train['target']
train.head()
train = train.drop("ID_code", axis = 1)
x =train.drop("target", axis = 1)
x.head()
y.head()
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
SEED = 1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=SEED)
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)

adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)

adb_clf.fit(X_train, y_train)

y_pred_proba = adb_clf.predict_proba(X_test)[:,1]

adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))
test = test.drop("ID_code", axis = 1)

y_pred = adb_clf.predict_proba(test_final)[:,1]

ID_code = sample["ID_code"]

prediction = pd.DataFrame(y_pred, index= ID_code)

prediction.columns = ["target"]

prediction.index.name = "ID_code"

prediction.head()
prediction.to_csv("submission_Decision_tree.csv")