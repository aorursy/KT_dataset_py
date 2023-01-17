import json
from pprint import pprint
import pandas as pd

ls = []
with open('../input/train.txt') as f:
    for line in f:
        ls.append(json.loads(line))
#ignore
ls_2 = []
with open('../input/test.txt') as f:
    for line in f:
        ls_2.append(json.loads(line))
X_test = pd.DataFrame(ls_2)
X_train = pd.DataFrame(ls)
X_train.head()
#ignore
X_test.head()
X_test = X_test.drop('patient_id', axis = 1)
X_train.info()
print("---------------------------------------")
X_train = X_train.drop('patient_id', axis = 1)
X_train['tag_dm2'].value_counts()
def convert_fill(df):
    return df.stack().apply(pd.to_numeric, errors='ignore').fillna(0).unstack()

X_train = convert_fill(X_train)
X_train['tag_dm2'] = pd.to_datetime(X_train['tag_dm2'], errors='coerce')
X_train['tag_dm2'].value_counts()
import datetime
date_0 = datetime.datetime(year=1970, month=1, day=1)
date_after = datetime.datetime(year=2017, month=12, day=31)
date_before = datetime.datetime(year=2017, month=1, day=1)
def f(row):
    if row['tag_dm2'] == date_0:
        val = 0
    elif row['tag_dm2']>date_after:
        val = 0
    elif row['tag_dm2']>=date_before and row['tag_dm2']<=date_after:
        val = 1
    else:
        val = -1
    return val
y_train = X_train.apply(f, axis=1)
y_train = y_train[y_train != -1]
y_train.value_counts()
X_train = X_train.drop('tag_dm2', axis = 1)
X_train = X_train.drop('split', axis = 1)
import seaborn as sns
sns.countplot(y_train,label="Count")
X_train.head()
X_train['bday'] = pd.to_datetime(X_train['bday'], errors='coerce')
def calculate_age(row):
    return pd.to_datetime('2016-12-31').year-row['bday'].year
X_train['age'] = X_train.apply(calculate_age, axis=1)
X_train = X_train.drop('bday', axis = 1)
X_train.head()
values = set()
for temp in list(X_train['resources'].dropna().values):
    for key, value in temp.items():
         print(values)
for v in values:
    print(v)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
()