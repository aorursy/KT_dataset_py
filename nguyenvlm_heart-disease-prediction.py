import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
seed_value= 30



# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

import os

os.environ['PYTHONHASHSEED']=str(seed_value)



# 2. Set the `python` built-in pseudo-random generator at a fixed value

import random

random.seed(seed_value)



# 3. Set the `numpy` pseudo-random generator at a fixed value

import numpy as np

np.random.seed(seed_value)
with open("../input/description.txt", "r") as f:

    print(*f.readlines(), sep="")
import pandas as pd

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(train.shape)

print(test.shape)
X_train = train.iloc[:, 1:-1].astype(np.float64)

y = train.iloc[:, -1]

X_test = test.iloc[:, 1:].astype(np.float64)
# Fill Inf and NaN

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())

X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.mean())
all_data = pd.concat([X_train, X_test])
categorical = [

    'cp',

    'sex',

    'fbs',

    'restecg',

    'exang',

    'slope',

    'thal'

]

print("--Categorical Features--")

for c in categorical:

    print(c, set(X_train[c]))

numerical = X_train.columns[np.logical_not(X_train.columns.isin(categorical))]

print("--Numerical Features--")

print(*numerical, sep="\n")
# Onehot encoding



all_data = pd.get_dummies(data = all_data, columns = categorical)

all_data.head()
# Create new features

all_data['age_thalach'] = all_data['thalach']*(220-all_data['age'])
# Skewness

skewed = [

    'chol', 

    'oldpeak',

    'thalach'

]

c = 1

for feature in skewed:

    if feature in numerical:

        all_data[feature] = np.log(all_data[feature]+c)
X_train = all_data[:X_train.shape[0]]

X_test = all_data[X_train.shape[0]:]
corr_data = pd.concat([X_train, y], axis=1)

print(corr_data.columns)

plt.figure(figsize=(30,10))

sns.heatmap(corr_data.corr(),cbar=True,fmt =' .2f', annot=True, cmap='coolwarm')
# Drop useless features



drop_features = [

    'fbs_0.0',

    'fbs_1.0'

]



for feature in drop_features:

    X_train.drop(columns=[feature], inplace=True)

    X_test.drop(columns=[feature], inplace=True)
X_train.head()
sns.countplot(y)

plt.show()

cnt_target = y.value_counts()

clf_thres = cnt_target[1]/(cnt_target[0]+cnt_target[1])

# clf_thres = 0.5

print(clf_thres)
def clf_result(y_proba, thres=clf_thres):

    return np.where(y_proba[:,1] > thres, 1, 0)
from sklearn.model_selection import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(X_train, y, test_size=0.2, stratify=y)
from sklearn.preprocessing import MinMaxScaler, StandardScaler



scaler = StandardScaler()

# scaler = MinMaxScaler()



x_train[:] = scaler.fit_transform(x_train[:])

x_valid[:] = scaler.transform(x_valid[:])



print(x_train.shape)

print(X_test.shape)

x_train.head()
X_test.head()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import classification_report



clf = LogisticRegression(C=0.3)

clf.fit(x_train, y_train)



print(classification_report(clf_result(clf.predict_proba(x_valid)),y_valid))



pd.DataFrame(pd.Series(sorted(clf.coef_.transpose().reshape(-1).tolist(), key=abs, reverse=True), index=X_train.columns))
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.linear_model import LogisticRegression



K = 5



models = []

scaler = StandardScaler()

# scaler = MinMaxScaler()

C = [

    0.05,

    0.3, 

    0.1, 

    0.5,

    0.1

]



accs = []

f1s = []

weights = np.zeros((1,X_train.shape[1]), dtype=np.float64)



folds = StratifiedKFold(n_splits = K, shuffle = True, random_state=seed_value)



for trainIdx, validIdx in folds.split(X_train, y):

    x_train, x_valid, y_train, y_valid = X_train.iloc[trainIdx], X_train.iloc[validIdx], y.iloc[trainIdx], y.iloc[validIdx]

    

    # Normalizing:

    x_train[:] = scaler.fit_transform(x_train[:])

    x_valid[:] = scaler.transform(x_valid[:])

    

    # Training:

    clf = LogisticRegression(C=C[len(models)])

    clf.fit(x_train, y_train)

    models.append(clf)

    print("\t\t\t--Fold %d evaluation--"%(len(models)))

    print("\t\t\t\tC =",C[len(models)-1])

    print(classification_report(clf_result(clf.predict_proba(x_valid)),y_valid))

    accs.append(accuracy_score(clf_result(clf.predict_proba(x_valid)), y_valid))

    f1s.append(f1_score(clf_result(clf.predict_proba(x_valid)), y_valid))

    weights = weights + clf.coef_

weights = (weights/K)

print("Average valid accuracy:", sum(accs)/K)

print("Average valid f1 score:", sum(f1s)/K)
pd.DataFrame(pd.Series(sorted(weights.transpose().reshape(-1).tolist(), key=abs, reverse=True), index=X_train.columns), columns=['Coefficient'])
# Normalizing test set:

scaler.fit(X_train[:])

X_test[:] = scaler.transform(X_test[:])
X_test.head()
# Final result:

final_result = sum(model.predict_proba(X_test) for model in models)/K

final_result = clf_result(final_result)

submit = pd.DataFrame()

submit['ID'] = test['ID']

submit['target'] = final_result.astype(np.int64)

sns.countplot(x='target', data=submit, palette="bwr")

plt.show()
cnt_pred_target = submit['target'].value_counts()

print(cnt_pred_target[1]/(cnt_pred_target[0]+cnt_pred_target[1]))
submit.to_csv('VuongLeMinhNguyen.csv', index=False)