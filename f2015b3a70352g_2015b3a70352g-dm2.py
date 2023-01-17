import pandas as pd

import numpy as np

import matplotlib as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.svm import LinearSVC

from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from sklearn.tree.tree import DecisionTreeClassifier

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
train = pd.read_csv("../input/train.csv", sep=",", na_values='?')

test = pd.read_csv("../input/test.csv", sep=",", na_values='?')
train.head()
for column in test.columns:

    if(train[column].isnull().sum()>5000):

        test.drop(column,axis=1,inplace=True)

        train.drop(column,axis=1,inplace=True)

for column in ['COB MOTHER', 'COB FATHER', 'COB SELF', 'Hispanic']:

    test[column].fillna(value = test[column].mode()[0], inplace = True)

    train[column].fillna(value = train[column].mode()[0], inplace = True)
train.head()
y = train['Class']

X = train.drop(['Class','ID'],axis=1)
X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state=1,shuffle=True)
column = X_train.columns

num_column = X_train._get_numeric_data().columns

cat_column = list(set(column) - set(num_column))
print(cat_column)
print(X_train.columns)
##Removed columns are WorkerClass, Enrolled, MIC, MOC, MLU, Reason, Area, State, MSA, REG, MOVE, Live, PREV, NOP, Teen, Fill  
train = pd.concat([X_train,y_train],axis=1)

val = pd.concat([X_val,y_val],axis=1)
le = LabelEncoder()

for ft in cat_column:

    train[ft] = le.fit_transform(train[ft])

    val[ft] = le.fit_transform(val[ft])

    test[ft] = le.fit_transform(test[ft])
target = 'Class'

X_train = train.drop(labels=target,axis = 1)

y_train = train[target]

X_val = val.drop(labels=target,axis = 1)

y_val = val[target]
def performance_metrics(y_true,y_pred):

    acc = accuracy_score(y_true, y_pred)

    rec = recall_score(y_true, y_pred)

    pre = precision_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    roc = roc_auc_score(y_true, y_pred)

    return acc,rec,pre,f1,roc
test_ids = test['ID']

test.drop('ID', axis = 1, inplace = True)
sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)

X_val_sc = sc.transform(X_val)

test_sc = sc.transform(test)
model = Pipeline([('sampling', RandomOverSampler()), ('classification', LinearSVC())])

model.fit(X_train_sc, y_train)

y_pred=model.predict(X_val_sc)

y_pred=list(y_pred)

acc,rec,pre,f1,roc = performance_metrics(y_val,y_pred)

print(roc)
model = Pipeline([('sampling', RandomOverSampler()), ('classification', AdaBoostClassifier())])

model.fit(X_train_sc, y_train)

y_pred=model.predict(X_val_sc)

y_pred=list(y_pred)

acc,rec,pre,f1,roc = performance_metrics(y_val,y_pred)

print(roc)
model = Pipeline([('sampling', RandomOverSampler()), ('classification', DecisionTreeClassifier())])

model.fit(X_train_sc, y_train)

y_pred=model.predict(X_val_sc)

y_pred=list(y_pred)

acc,rec,pre,f1,roc = performance_metrics(y_val,y_pred)

print(roc)
model = Pipeline([('sampling', RandomOverSampler()), ('classification', MLPClassifier())])

model.fit(X_train_sc, y_train)

y_pred=model.predict(X_val_sc)

y_pred=list(y_pred)

acc,rec,pre,f1,roc = performance_metrics(y_val,y_pred)

print(roc)
model = Pipeline([('sampling', RandomOverSampler()), ('classification', AdaBoostClassifier())])

model.fit(X_train_sc, y_train)

y_pred=model.predict(X_val_sc)

y_pred=list(y_pred)

acc,rec,pre,f1,roc = performance_metrics(y_val,y_pred)

print(roc)
y_pred_tst = model.predict(test_sc)

y_pred_tst = list(y_pred_tst)

test['Class'] = y_pred_tst

pd.concat([test_ids,test['Class']],axis = 1).to_csv(r'Submissions/'+'sub06.csv',index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(pd.concat([test_ids,test['Class']],axis = 1))