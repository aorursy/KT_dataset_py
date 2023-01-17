import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

plt.style.use('bmh')
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.head()
df.describe()
df.isnull().sum()
df.dtypes
sns.countplot(x='Class', data=df, palette='CMRmap')

print('Non-fraud transactions: {}%'.format(round(df.Class.value_counts()[0]/len(df)*100.0,2)))

print('Fraud transactions: {}%'.format(round(df.Class.value_counts()[1]/len(df)*100.0,2)))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1 = sns.distplot(df['Time'], ax=ax1, color='y')

ax2 = sns.distplot(df['Amount'], ax=ax2, color='r')

ax1.set_title('Distribution of Time', fontsize=13)

ax2.set_title('Distribution of Amount', fontsize=13)
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()

df['scaled_amount'] = rs.fit_transform(df['Amount'].values.reshape(-1,1))

df['scaled_time'] = rs.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
scaled_amount = df['scaled_amount']

scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

df.insert(0, 'scaled_amount', scaled_amount)

df.insert(0, 'scaled_time', scaled_time)

df.head()
from sklearn.model_selection import train_test_split as holdout

x = np.array(df.iloc[:, df.columns != 'Class'])

y = np.array(df.iloc[:, df.columns == 'Class'])

x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, precision_score, recall_score, accuracy_score

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

cnf_matrix = confusion_matrix(y_test, y_pred)



sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')



labels = ['Non-fraud', 'Fraud']

print(classification_report(y_test, y_pred, target_names=labels))
from imblearn.over_sampling import SMOTE



print("Transaction Number x_train dataset: ", x_train.shape)

print("Transaction Number y_train dataset: ", y_train.shape)

print("Transaction Number x_test dataset: ", x_test.shape)

print("Transaction Number y_test dataset: ", y_test.shape)



print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))



sm = SMOTE(random_state=2)

x_train_s, y_train_s = sm.fit_sample(x_train, y_train.ravel())



print('After OverSampling, the shape of train_x: {}'.format(x_train_s.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_s.shape))



print("After OverSampling, counts of label '1', %: {}".format(sum(y_train_s==1)/len(y_train_s)*100.0,2))

print("After OverSampling, counts of label '0', %: {}".format(sum(y_train_s==0)/len(y_train_s)*100.0,2))



sns.countplot(x=y_train_s, data=df, palette='CMRmap')
logreg = LogisticRegression()

logreg.fit(x_train_s, y_train_s)

y_pred = logreg.predict(x_test)

cnf_matrix = confusion_matrix(y_test, y_pred)



sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')



print(classification_report(y_test, y_pred))
y_pred_prob = logreg.predict_proba(x_test)[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.plot(precision, recall)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision Recall Curve')
from sklearn.ensemble import RandomForestClassifier as rfc

rand_f = rfc(n_estimators=1000, min_samples_split=10, min_samples_leaf=1,

           max_features='auto', max_leaf_nodes=None,

           oob_score=True, n_jobs=-1, random_state=1)

rand_f.fit(x_train_s, y_train_s)

y_pred = rand_f.predict(x_test)



cnf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')



print(classification_report(y_test, y_pred))
y_pred_prob = rand_f.predict_proba(x_test)[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.plot(precision, recall)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision Recall Curve')
#Plotting Feature Importances

print('Feature importance ranking\n\n')

importances = rand_f.feature_importances_

std = np.std([tree.feature_importances_ for tree in rand_f.estimators_],axis=0)

indices = np.argsort(importances)[::-1]

variables = df.columns

importance_list = []

for f in range(x.shape[1]):

    variable = variables[indices[f]]

    importance_list.append(variable)

    print("%d.%s(%f)" % (f + 1, variable, importances[indices[f]]))

plt.figure(figsize=(20, 8))

plt.title("Feature importances")

plt.bar(importance_list, importances[indices],

       color="purple", yerr=std[indices], align='center')
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators = 5000, max_depth = 30, learning_rate = 0.01)

model.fit(x_train_s, y_train_s)

y_pred = model.predict(x_test)



cnf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')



print(classification_report(y_test, y_pred))
from keras.models import Sequential

from keras.layers import Dense, Dropout

model = Sequential([Dense(input_dim=30, units=16, activation='relu'),

                   Dense(units=24, activation='relu'),

                   Dropout(0.5),

                   Dense(units=20, activation='relu'),

                   Dense(units=24, activation='relu'),

                   Dense(units=1, activation='sigmoid')])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train_s, y_train_s, batch_size=15, epochs=15)
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
score = model.evaluate(x_test, y_test)

print(score)
y_pred = model.predict_classes(x_test)



cnf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')



print(classification_report(y_test, y_pred))