import pandas as pd

# load data
data_orig = pd.read_csv("../input/UCI_Credit_Card.csv")
data_orig.head(10)
print("data size: " + str(data_orig.shape))
print("default size: " + str(data_orig.ix[data_orig['default.payment.next.month'] == 1,:].shape))
# omit-target columns
omit_target_label = ['ID']

# categorical columns
pay_label = ['PAY_'+str(i) for i in range(0,7) if i != 1]
categorical_label = ['SEX', 'EDUCATION', 'MARRIAGE']

categorical_label.extend(pay_label)
dummied_columns = pd.get_dummies(data_orig[categorical_label].astype('category'))

# drop columns
data_orig = data_orig.drop(columns=categorical_label)
data_orig = data_orig.drop(columns=omit_target_label)

# merge one-hot-encoded columns
data = pd.concat([data_orig, dummied_columns], axis=1, join='outer')
from sklearn.model_selection import train_test_split

# explaining and explained
target = data['default.payment.next.month']
data = data.drop(columns=['default.payment.next.month'])

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.33)
from sklearn.linear_model import LogisticRegression

univar = x_train[['BILL_AMT4']]

lr = LogisticRegression()
lr.fit(univar, y_train)
from sklearn.metrics import roc_auc_score
import numpy as np

predicted_score = np.array([score[1] for score in lr.predict_proba(x_test[['BILL_AMT4']])])

roc_auc_score(y_test.values, predicted_score)
explaining_labels = x_train.columns
auc_outcomes = {}
for label in explaining_labels:
    univar = x_train[[label]]
    
    lr = LogisticRegression()
    lr.fit(univar, y_train)
    
    predicted_score = np.array([score[1] for score in lr.predict_proba(x_test[[label]])])
    
    auc_outcomes[label] = roc_auc_score(y_test.values, predicted_score)

%matplotlib inline
import matplotlib.pyplot as plt

label = []
score = []
for item in sorted(auc_outcomes.items(), key=lambda x: x[1], reverse=True):
    label.append(item[0])
    score.append(item[1])

# I wanted to show the bars with decreasing order. But it didn't work here.
plt.bar(label, score)
# using all the explaining variables
lr = LogisticRegression()
lr.fit(x_train, y_train)

predicted_score = np.array([score[1] for score in lr.predict_proba(x_test)])
    
roc_auc_score(y_test.values, predicted_score)
from sklearn.metrics import brier_score_loss

brier_score_loss(y_test.values, predicted_score)
predicted_score_train = np.array([score[1] for score in lr.predict_proba(x_train)])
predicted_score_test = np.array([score[1] for score in lr.predict_proba(x_test)])

auc_train = roc_auc_score(y_train.values, predicted_score_train)
auc_test = roc_auc_score(y_test.values, predicted_score_test)
brier_train = brier_score_loss(y_train.values, predicted_score_train)
brier_test = brier_score_loss(y_test.values, predicted_score_test)

auc = [auc_train, auc_test]
brier = [brier_train, brier_test]
pd.DataFrame({'auc': auc, 'brier': brier}, index=['train', 'test'])