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
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import confusion_matrix, classification_report, f1_score



import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
!python3 /kaggle/input/itis-hackathon/make_dataset_kaggle.py /kaggle/input/itis-hackathon/
train = pd.read_csv('/kaggle/working/interim_train_09.csv')

valid = pd.read_csv('/kaggle/working/interim_train_10.csv')

test = pd.read_csv('/kaggle/working/interim_test.csv')
cities = ['Еманжелинск',

'Златоуст',

'Каменск-Уральский',

'Каслин',

'Копейск',

'Коркино',

'Курган',

'Кыштым',

'Магнитогорск',

'Миасс',

'Озерск',

'Пласт',

'Троицк',

'Чебаркуль',

'Челябинск',

'Южно-Уральск']



target_related_features = ['USER_ID', 'TARGET', 'ID',

                           'SERVICE_INT_ID',

                           'ADMIN_QUESTION_INT_ID',

'FEATURE_INT_ID','CHANNEL_INT_ID']



strange_features = [ 'pattern_sms_62', 'pattern_sms_63', 'pattern_sms_66', 'pattern_sms_68',

       'pattern_sms_73', 'pattern_sms_74', 'pattern_sms_75', 'pattern_sms_76',

       'pattern_sms_90', 'pattern_sms_107', 'pattern_sms_109',

       'pattern_sms_111', 'pattern_sms_112', 'pattern_sms_113',

       'pattern_sms_115', 'pattern_sms_116', 'pattern_sms_117',

       'pattern_sms_151', 'pattern_sms_171', 'pattern_sms_181',

       'pattern_sms_183', 'pattern_sms_227', 'pattern_sms_233', 'BAL_BELOW_ZERO']



#train = pd.concat([train[train['TARGET'] == 0].sample(train[train['TARGET'] == 1].shape[0], random_state=17), train[train['TARGET'] == 1]])

train = train.drop(columns=strange_features)

train = train.drop(columns='pattern_sms_114')

train['ACTIVATE_DATE'] = (pd.to_datetime(train['ACTIVATE_DATE'])-pd.to_datetime('2019-09-01 00:00:00')).apply(lambda d: d.days*(-1))

train=train[train['ACTIVATE_DATE'] > 0]

valid = valid.drop(columns=strange_features)

train['IS_BIG_CITY'] = train.apply((lambda x: x['CITY_NAME'] in cities), axis=1)

train = train[train['IS_BIG_CITY']]

train = train.drop(columns=['IS_BIG_CITY'])

train['CITY_NAME'].dropna()

valid['IS_BIG_CITY'] = valid.apply((lambda x: x['CITY_NAME'] in cities), axis=1)

valid = valid[valid['IS_BIG_CITY']]

valid = valid.drop(columns=['IS_BIG_CITY'])

valid['ACTIVATE_DATE'] = (pd.to_datetime(valid['ACTIVATE_DATE'])-pd.to_datetime('2019-10-01 00:00:00')).apply(lambda d: d.days*(-1))

valid=valid[valid['ACTIVATE_DATE'] > 0]

X_train = train.drop(columns=target_related_features)

X_train = X_train.fillna(value=0)

X_train = pd.get_dummies(X_train,columns=['CITY_NAME'], prefix = 'city')

X_train = pd.get_dummies(X_train,columns=['PHYZ_TYPE'], prefix = 'phyztype')

X_valid = valid.drop(columns=target_related_features)

X_valid = X_valid.fillna(value=0)

X_valid['ACTIVATE_DATE'] = (pd.to_datetime(X_valid['ACTIVATE_DATE'])-pd.to_datetime('2019-09-01 00:00:00')).apply(lambda d: d.days*(-1))

X_valid = pd.get_dummies(X_valid,columns=['CITY_NAME'], prefix = 'city')

X_valid = pd.get_dummies(X_valid,columns=['PHYZ_TYPE'], prefix = 'phyztype')

X_test = test

y_train = train['TARGET']

y_valid = valid['TARGET']
X_train.head()
X_valid['ACTIVATE_DATE'] = (pd.to_datetime(X_valid['ACTIVATE_DATE'])-pd.to_datetime('2019-09-01 00:00:00')).apply(lambda d: d.days*(-1))

X_valid = pd.get_dummies(X_valid['CITY_NAME'], prefix = 'city')
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=15, class_weight={1: 10 }, min_samples_leaf=300, random_state=17)

clf.fit(X_train, y_train)
import pydot 

from sklearn.externals.six import StringIO 

dot_data = StringIO() 

tree.export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=False, impurity=False, feature_names=X_train.columns) 

graph = pydot.graph_from_dot_data(dot_data.getvalue()) 

graph[0].write_pdf("iris.pdf")
pr_y_train = clf.predict(X_train)

print(classification_report(y_true=y_train, y_pred=pr_y_train))

print(confusion_matrix(y_true=y_train, y_pred=pr_y_train))

pr_y_valid = clf.predict(X_valid)

print(classification_report(y_true=y_valid, y_pred=pr_y_valid))

print(confusion_matrix(y_true=y_valid, y_pred=pr_y_valid))
lr = LogisticRegression(class_weight="balanced", random_state=17, n_jobs=-1)

lr.fit(X_train, y_train)
pr_y_train = lr.predict(X_train)

print(classification_report(y_true=y_train, y_pred=pr_y_train))

print(confusion_matrix(y_true=y_train, y_pred=pr_y_train))

pr_y_valid = lr.predict(X_valid)

print(classification_report(y_true=y_valid, y_pred=pr_y_valid))

print(confusion_matrix(y_true=y_valid, y_pred=pr_y_valid))
test_orig = pd.read_csv('/kaggle/working/interim_test.csv')
cities = ['Еманжелинск',

'Златоуст',

'Каменск-Уральский',

'Каслин',

'Копейск',

'Коркино',

'Курган',

'Кыштым',

'Магнитогорск',

'Миасс',

'Озерск',

'Пласт',

'Троицк',

'Чебаркуль',

'Челябинск',

'Южно-Уральск']



target_related_features = ['USER_ID', 'TARGET', 'ID',

                           'SERVICE_INT_ID',

                           'ADMIN_QUESTION_INT_ID',

'FEATURE_INT_ID','CHANNEL_INT_ID', 'PHYZ_TYPE']



strange_features = [ 'pattern_sms_62', 'pattern_sms_63', 'pattern_sms_66', 'pattern_sms_68',

       'pattern_sms_73', 'pattern_sms_74', 'pattern_sms_75', 'pattern_sms_76',

       'pattern_sms_90', 'pattern_sms_107', 'pattern_sms_109',

       'pattern_sms_111', 'pattern_sms_112', 'pattern_sms_113',

       'pattern_sms_115', 'pattern_sms_116', 'pattern_sms_117',

       'pattern_sms_151', 'pattern_sms_171', 'pattern_sms_181',

       'pattern_sms_183', 'pattern_sms_227', 'pattern_sms_233', 'BAL_BELOW_ZERO']

print(test_orig.shape[0])

test = test_orig.drop(columns=strange_features)

if 'pattern_sms_114' in test.columns:

    test = test.drop(columns='pattern_sms_114')

test['ACTIVATE_DATE'] = (pd.to_datetime(test['ACTIVATE_DATE'])-pd.to_datetime('2019-09-01 00:00:00')).apply(lambda d: d.days*(-1))

result = pd.DataFrame(columns=['USER_ID', 'PREDICT'])

result['USER_ID'] = test[test['ACTIVATE_DATE'] <= 0]['USER_ID']

test = test[test['ACTIVATE_DATE'] > 0]

print(test.shape[0]+result.shape[0])

temp = pd.DataFrame(columns=['USER_ID', 'PREDICT'])

test['IS_BIG_CITY'] = test.apply((lambda x: x['CITY_NAME'] in cities), axis=1)

temp['USER_ID'] = test[test['IS_BIG_CITY'] == False]['USER_ID']

print(test[test['IS_BIG_CITY'] == True].shape[0]+test[test['IS_BIG_CITY'] == False].shape[0]+result.shape[0])

result = pd.concat([result, temp])

result['PREDICT'] = 0

test = test[test['IS_BIG_CITY'] == True]

print(test.shape[0]+result.shape[0])

test = test.drop(columns=['IS_BIG_CITY'])

test = test.fillna(value=0)

test = pd.get_dummies(test,columns=['CITY_NAME'], prefix = 'city')

test = pd.get_dummies(test,columns=['PHYZ_TYPE'], prefix = 'phyztype')

temp = pd.DataFrame(columns=['USER_ID', 'PREDICT'])

temp['USER_ID'] = test['USER_ID']

temp['PREDICT'] = clf.predict(test.drop(columns=['USER_ID']))

result = pd.concat([result, temp])

print(test.shape[0])

print(result.shape[0])

result.to_csv("submit2.csv", index=False)
result.to_csv("submit1.csv", index=False)
test_orig[['USER_ID']].describe()