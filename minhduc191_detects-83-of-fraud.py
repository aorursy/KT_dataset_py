import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/creditcard.csv")

# print(df.describe())

print(df['Class'].value_counts())
y = df['Class']

df = df.drop("Class",axis=1)



print(df.shape)

print(df.columns)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report



base_clf = DecisionTreeClassifier()
from sklearn.model_selection import GridSearchCV

tree_params = {'max_depth': np.arange(1, 11, 1)}

gs_base = GridSearchCV(base_clf, tree_params, n_jobs=-1, verbose=1)
gs_base = gs_base.fit(X_train, y_train)

base_pred = gs_base.predict(X_test)

print(classification_report(base_pred, y_test))



print('best depth of decision tree: %d' %gs_base.best_params_['max_depth'] )
rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test) 



print(classification_report(y_pred,y_test))


rf_params = {'criterion': ('gini', 'entropy'), 

             'n_estimators': np.arange(5, 25, 5) } # 'max_depth': np.arange(1, 11, 1)

gs_rfc = GridSearchCV(rfc, rf_params, n_jobs=-1, verbose=1)



gs_rfc = gs_rfc.fit(X_train, y_train)

gs_rfc_pred = gs_rfc.predict(X_test)

print(classification_report(y_test, y_pred=gs_rfc_pred))



for param_name in rf_params.keys():

    print('%s %r' %(param_name, gs_rfc.best_params_[param_name]))