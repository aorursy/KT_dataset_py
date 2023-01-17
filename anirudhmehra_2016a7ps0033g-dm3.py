import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
data_benign_orig = pd.read_csv('../input/opcode_frequency_benign.csv', sep = ',', na_values = '?')

data_malware_orig = pd.read_csv('../input/opcode_frequency_malware.csv', sep = ',', na_values = '?')
data_benign_orig['Class'] = 0

data_malware_orig['Class'] = 1
data = pd.concat([data_benign_orig, data_malware_orig])
data.drop(columns = 'FileName', axis = 1 , inplace = True)

data = data.drop_duplicates()

# data = data.loc[:, (data != data.iloc[0]).any()]
X = data.drop('Class', axis = 1)

y = data['Class']
X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size = 0.2, random_state = 42)
best_dt = DecisionTreeClassifier(max_depth = 10, random_state = 42) #0.1 0.2 

best_dt.fit(X_train,y_train)
y_pred = list(best_dt.predict(X_train))

roc = roc_auc_score(y_train,y_pred)

roc
y_pred = list(best_dt.predict(X_test))

roc = roc_auc_score(y_test,y_pred)

roc
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
dt_ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 2), n_estimators = 18)

dt_ada.fit(X_train,y_train)
y_pred = list(dt_ada.predict(X_train))

roc = roc_auc_score(y_train,y_pred)

roc
y_pred = list(dt_ada.predict(X_test))

roc = roc_auc_score(y_test,y_pred)

roc
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# best_rf = RandomForestClassifier(max_depth = 7, random_state = 42, n_estimators = 5, n_jobs = -1) #worked 0.1 0.2

best_rf = RandomForestClassifier(max_depth = 14, random_state = 42, n_estimators = 15, n_jobs = -1) #worked 0.1 0.2

best_rf.fit(X_train,y_train)
y_pred = list(best_rf.predict(X_train))

roc = roc_auc_score(y_train,y_pred)

roc
y_pred = list(best_rf.predict(X_test))

roc = roc_auc_score(y_test,y_pred)

roc
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
evc = VotingClassifier(estimators= [('best_dt',best_dt),('best_rf',best_rf),('dt_ada',dt_ada)], voting = 'hard')

evc.fit(X_train,y_train)
y_pred = list(evc.predict(X_train))

roc = roc_auc_score(y_train,y_pred)

roc
y_pred = list(evc.predict(X_test))

roc = roc_auc_score(y_test,y_pred)

roc
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
test_orig = pd.read_csv("../input/Test_data.csv", sep=',', na_values = '?')

test = test_orig.drop(columns = ['FileName', 'Unnamed: 1809'], axis = 1)
evc = VotingClassifier(estimators= [('best_dt',best_dt),('best_rf',best_rf),('dt_ada',dt_ada)], voting = 'hard')

data = pd.concat([data_benign_orig, data_malware_orig])

data.drop(columns = 'FileName', axis = 1 , inplace = True)

data = data.drop_duplicates()

data = data.sample(frac = 1,random_state = 42)

X = data.drop('Class', axis = 1)

y = data['Class']

evc.fit(X,y)

result = list(evc.predict(test))
final = pd.DataFrame({'FileName': test_orig['FileName'],'Class': result})
print(sum(final['Class'] == 0))

print(sum(final['Class'] == 1))

print(sum(final['Class'] == 0)/(sum(final['Class'] == 1) + sum(final['Class'] == 0)))

print(sum(final['Class'] == 1)/(sum(final['Class'] == 1) + sum(final['Class'] == 0)))
from IPython.display import HTML

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html) 

create_download_link(final)