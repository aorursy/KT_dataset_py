import pandas as pd



PATH_TO_DATA = '../input/bank.csv'

df = pd.read_csv(PATH_TO_DATA)

df.head()
df.isna().sum()
df.dtypes
import matplotlib.pyplot as plt

%matplotlib inline



df.hist(bins=20, figsize=(15,10))

plt.show()
df.drop(columns=['duration'], inplace=True)



from sklearn.preprocessing import minmax_scale



for col in ['age', 'balance', 'campaign', 'pdays', 'previous']:

    df[col] = minmax_scale(df[col].values.astype(float).reshape(-1, 1))
import seaborn as sns

import numpy as np



tmp = pd.concat([df.select_dtypes(exclude=np.number), df['y']], axis=1)

n = len(tmp.columns)



plt.figure(figsize=(15,30))

for i, col in enumerate(tmp.columns):

    ax = plt.subplot(int(np.ceil(n/2)),2,i+1)

    if tmp[col].nunique() > 6:

        sns.countplot(y=col, data=tmp, hue='y', ax=ax)

    else:

        sns.countplot(x=col, data=tmp, hue='y', ax=ax)



plt.show()
df = pd.get_dummies(df, drop_first=True, 

                    columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome'])

df.head()
df['date'] = pd.to_datetime(df['day'].astype(str) + ' ' + df['month'] + ' ' + '2014')

df.drop(columns=['day', 'month'], inplace=True)

df[df['y']==1].set_index('date').resample('D').size().plot(figsize=(15,5), label='Subsribed deposit')

df[df['y']==0].set_index('date').resample('D').size().plot(label='Didn\' subscribe deposit')

plt.legend()

plt.show()
tmp = df.groupby(['y', (df['date'].dt.weekday>4)]).size().reset_index()

tmp['date'] = tmp['date'].replace(to_replace={False: 'Weekday', True: 'Weekend'})

tmp
df['month'] = df['date'].dt.month

df['weekend'] = df['date'].dt.weekday >= 5

df.drop(columns='date', inplace=True)

df = pd.get_dummies(df, columns=['month', 'weekend'], drop_first=True)

df.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['y']).values, 

                                                    df['y'].values,

                                                    test_size=0.2,

                                                    random_state=17

                                                   )

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LogisticRegressionCV



lrCVroc_auc = LogisticRegressionCV(Cs = 100, random_state=17, cv=5, 

                                   scoring='roc_auc', max_iter=10000000)

lrCVroc_auc.fit(X_train, y_train)



lrCVprec_rec = LogisticRegressionCV(Cs = 100, random_state=17, cv=5, 

                                    scoring='average_precision', max_iter=10000000)

lrCVprec_rec.fit(X_train, y_train)



from sklearn.metrics import average_precision_score, roc_auc_score



print('TRAIN RESULTS:')

y_pred = lrCVroc_auc.predict_proba(X_train)[:,1]

print('ROC-AUC score for ROC-AUC trained model: {0:.5f}'.format(roc_auc_score(y_train, y_pred)))

print('PR AUC  score for ROC-AUC trained model: {0:.5f}'.format(average_precision_score(y_train, y_pred)))



y_pred = lrCVprec_rec.predict_proba(X_train)[:,1]

print('ROC-AUC score for PR-AUC  trained model: {0:.5f}'.format(roc_auc_score(y_train, y_pred)))

print('PR AUC  score for PR-AUC  trained model: {0:.5f}'.format(average_precision_score(y_train, y_pred)))



print('TEST RESULTS:')

y_pred = lrCVroc_auc.predict_proba(X_test)[:,1]

print('ROC-AUC score for ROC-AUC trained model: {0:.5f}'.format(roc_auc_score(y_test, y_pred)))

print('PR AUC  score for ROC-AUC trained model: {0:.5f}'.format(average_precision_score(y_test, y_pred)))



y_pred = lrCVprec_rec.predict_proba(X_test)[:,1]

print('ROC-AUC score for PR-AUC  trained model: {0:.5f}'.format(roc_auc_score(y_test, y_pred)))

print('PR AUC  score for PR-AUC  trained model: {0:.5f}'.format(average_precision_score(y_test, y_pred)))