import warnings

warnings.filterwarnings("ignore")



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from scipy import stats



import jb_helper_functions_prep

from jb_helper_functions_prep import create_enc



import prep_telco

from prep_telco import prep_telco_df



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import tree
df = prep_telco_df()

df.head()
train, test = train_test_split(df, test_size=.3, random_state=123, stratify=df[['churn_enc']])
print('Percent of non-churn: ' + str(train.churn_enc.value_counts()[0]/train.churn_enc.count()))

print('Percent of churn: ' + str(train.churn_enc.value_counts()[1]/train.churn_enc.count()))
y_train = train[['churn_enc']]

y_test = test[['churn_enc']]
df.columns
X_train = train[['tenure', 'monthlycharges', 'internetservice_enc', 'techsupport_enc', 'contract_enc', 'phoneservice_enc']]

X_test = test[['tenure', 'monthlycharges', 'internetservice_enc', 'techsupport_enc', 'contract_enc', 'phoneservice_enc']]
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=123).fit(X_train, y_train)

y_pred = clf.predict(X_train)

y_pred_proba = clf.predict_proba(X_train)

print('Accuracy of Decision Tree classifier on training set: {:.6f}'

     .format(clf.score(X_train, y_train)))
print(classification_report(y_train, y_pred))
cm = pd.DataFrame(confusion_matrix(y_train, y_pred),

             columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +'])



cm
log_reg = LogisticRegression(random_state=123, solver='saga').fit(X_train, y_train)

y_pred = log_reg.predict(X_train)

print('Accuracy of Logistic Regression classifier on training set: {:.6f}'

     .format(log_reg.score(X_train, y_train)))
print(classification_report(y_train, y_pred))
cm = pd.DataFrame(confusion_matrix(y_train, y_pred),

             columns=['Pred -', 'Pred +'], index=['Actual -', 'Actual +'])



cm
from sklearn.preprocessing import StandardScaler



import keras

from keras.models import Sequential

from keras.layers import Dense



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
cf = Sequential()

cf.add(Dense(output_dim=4, init='uniform', activation='relu', input_dim=6))

cf.add(Dense(output_dim=4, init='uniform', activation='relu'))

cf.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))



cf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

cf.fit(X_train, y_train, nb_epoch=100, batch_size=30)



scores = cf.evaluate(X_train, y_train)

print('%s: %.2f%%' % (cf.metrics_names[1], scores[1]*100))
print('Accuracy of Decision Tree classifier on test set: {:.6f}'

     .format(log_reg.score(X_test, y_test)))