import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score



from imblearn.over_sampling import SMOTE



import tensorflow as tf
df = pd.read_csv('../input/loan_final313.csv')
df.head()
df.info()
plt.figure(figsize=(16,8))

sns.heatmap(df.corr(),

           cmap = 'coolwarm')
def defaulted(x):

    if x == 'Good Loan':

        return 0

    else:

        return 1
df['default'] = df['loan_condition'].apply(lambda x: defaulted(x))
df.drop('id', axis=1, inplace=True)
df.drop('year', axis=1, inplace=True)
df.drop('issue_d', axis=1, inplace=True)
df.drop('final_d', axis=1, inplace=True)
scaler = MinMaxScaler()
df['emp_length_int'] = scaler.fit_transform(df['emp_length_int'].values.reshape(-1,1))
plt.figure(figsize=(12,6))

sns.countplot(x='home_ownership',data=df, hue='default')
df = pd.concat([df, pd.get_dummies(df['home_ownership'])],axis=1).drop(['home_ownership', 'home_ownership_cat'],axis=1)
df.drop(['OTHER', 'NONE', 'ANY'],axis=1,inplace=True)
df = pd.concat([df, pd.get_dummies(df['income_category'])],axis=1).drop(['income_category', 'income_cat'],axis=1)
plt.figure(figsize=(12,6))

sns.boxplot(x=df['annual_inc'])
outliers = df[df['annual_inc'] > df['annual_inc'].quantile(0.99)].index
df.loc[outliers,'annual_inc'] = df['annual_inc'].quantile(0.99)
plt.figure(figsize=(12,6))

sns.boxplot(x=df['annual_inc'])
scaler = MinMaxScaler()

df['annual_inc'] = scaler.fit_transform(df['annual_inc'].values.reshape(-1,1))
plt.figure(figsize=(12,6))

sns.boxplot(x=df['loan_amount'])
scaler = MinMaxScaler()

df['loan_amount'] = scaler.fit_transform(df['loan_amount'].values.reshape(-1,1))
df['term'].unique()
plt.figure(figsize=(12,6))

sns.countplot(x='term',data=df, hue='default')
df = pd.concat([df, pd.get_dummies(df['term_cat'],prefix='term')],axis=1).drop(['term', 'term_cat'],axis=1)
plt.figure(figsize=(12,6))

sns.countplot(x='application_type_cat',data=df, hue='default')
df.drop(['application_type','application_type_cat'],axis=1,inplace=True)
df['purpose'].unique()
plt.figure(figsize=(12,6))

sns.countplot(x='purpose',data=df, hue='default')
df = pd.concat([df, pd.get_dummies(df['purpose'])],axis=1).drop(['purpose', 'purpose_cat'],axis=1)
df.drop(['car', 'small_business', 'other', 'wedding', 'home_improvement', 'major_purchase',

       'medical', 'moving', 'vacation', 'house', 'renewable_energy',

       'educational'],axis=1, inplace=True)
plt.figure(figsize=(12,6))

sns.countplot(x='interest_payments',data=df, hue='default')
df = pd.concat([df, pd.get_dummies(df['interest_payments'],prefix='int')],axis=1).drop(['interest_payments', 'interest_payment_cat'],axis=1)
df.drop('int_High',axis=1,inplace=True)
df.drop(['loan_condition', 'loan_condition_cat'],axis=1,inplace=True)
plt.figure(figsize=(12,6))

plt.hist(df[df['default']==0]['interest_rate'],color='orange',alpha=0.5,label='Good')

plt.hist(df[df['default']==1]['interest_rate'],color='blue',alpha=0.5,label='Bad')



plt.legend()
plt.figure(figsize=(12,6))

sns.boxplot(x=df['interest_rate'])
outliers = df[df['interest_rate'] > df['interest_rate'].quantile(.99)].index
df.loc[outliers,'interest_rate'] = df['interest_rate'].quantile(.99)
scaler = MinMaxScaler()

df['interest_rate'] = scaler.fit_transform(df['interest_rate'].values.reshape(-1,1))
df.info()
plt.figure(figsize=(12,6))

sns.countplot(x='grade',data=df, hue='default')
df.drop(['grade', 'grade_cat'],axis=1,inplace=True)
plt.figure(figsize=(12,6))

sns.boxplot(x=df['dti'])
outliers = df[df['dti'] > df['dti'].quantile(.99)].index

df.loc[outliers,'dti'] = df['dti'].quantile(.99)
scaler = MinMaxScaler()

df['dti'] = scaler.fit_transform(df['dti'].values.reshape(-1,1))
df.drop('total_pymnt', axis=1, inplace=True)
df.drop('total_rec_prncp', axis=1, inplace=True)
df.drop('recoveries', axis=1, inplace=True)
plt.figure(figsize=(12,6))

plt.hist(df[df['default']==0]['installment'],color='orange',alpha=0.5,label='Good')

plt.hist(df[df['default']==1]['installment'],color='blue',alpha=0.5,label='Bad')



plt.legend()
df.drop('installment', axis=1, inplace=True)
df.drop('region', axis=1, inplace=True)
sum(df['default']) / len(df)
X = df.drop(['default'],axis=1)

y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
rf_base = RandomForestClassifier(n_estimators=100)

ada_base = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100)
rf_base.fit(X_train,y_train)

ada_base.fit(X_train,y_train)
rf_base_pred = rf_base.predict(X_test)

ada_base_pred = ada_base.predict(X_test)
print(confusion_matrix(y_test, rf_base_pred))

print('\n')

print(confusion_matrix(y_test, ada_base_pred))
print(accuracy_score(y_test,rf_base_pred))

print(accuracy_score(y_test,ada_base_pred))
print(recall_score(y_test,rf_base_pred))

print(recall_score(y_test,ada_base_pred))
ratios = np.array([0.2,0.4,0.6,0.8,1.0])

rf_acc_us = []

ada_acc_us = []

rf_recall_us = []

ada_recall_us = []

default_count = y_train.value_counts()[1]

X_train.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)

bad_indices = y_train[y_train==1].index
for i in ratios:

    majority_count = int(np.floor(default_count / i))

    good_indices = np.random.choice(y_train[y_train==0].index,size=majority_count)

    indices = np.concatenate((bad_indices,good_indices))

    X_train_us = X_train.iloc[indices]

    y_train_us = y_train.iloc[indices]

    

    rf_ = RandomForestClassifier(n_estimators=100)

    ada_ = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100)

    

    rf_.fit(X_train_us,y_train_us)

    ada_.fit(X_train_us,y_train_us)

    

    rf_pred = rf_.predict(X_test)

    ada_pred = ada_.predict(X_test)

    

    rf_acc_us.append(accuracy_score(y_test,rf_pred))

    ada_acc_us.append(accuracy_score(y_test,ada_pred))

    

    rf_recall_us.append(recall_score(y_test,rf_pred))

    ada_recall_us.append(recall_score(y_test,ada_pred))

    
ada_acc_us
rf_acc_us
ada_recall_us
rf_recall_us
plt.figure(figsize=(12,6))

plt.plot(ratios,ada_acc_us,linestyle='-',marker='o',color='red',label='ada_acc')

plt.plot(ratios,ada_recall_us,linestyle='-',marker='o',color='red',label='ada_rec',alpha=0.5)

plt.plot(ratios,rf_acc_us,linestyle='-',marker='o',color='blue',label='rf_acc')

plt.plot(ratios,rf_recall_us,linestyle='-',marker='o',color='blue',label='rf_rec', alpha=0.5)



plt.legend()
ratios = np.array([0.2,0.4,0.6,0.8,1.0])

rf_acc_os = []

ada_acc_os = []

rf_recall_os = []

ada_recall_os = []

X_train.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
for j in ratios:

    sm = SMOTE(random_state=101, ratio = j)

    X_train_os, y_train_os = sm.fit_sample(X_train, y_train)

    

    X_train_os[:,5:] = np.round(X_train_os[:,5:])

    

    rf_ = RandomForestClassifier(n_estimators=100)

    ada_ = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100)

    

    rf_.fit(X_train_os,y_train_os)

    ada_.fit(X_train_os,y_train_os)

    

    rf_pred = rf_.predict(X_test)

    ada_pred = ada_.predict(X_test)

    

    rf_acc_os.append(accuracy_score(y_test,rf_pred))

    ada_acc_os.append(accuracy_score(y_test,ada_pred))

    

    rf_recall_os.append(recall_score(y_test,rf_pred))

    ada_recall_os.append(recall_score(y_test,ada_pred))
rf_acc_os
ada_acc_os
rf_recall_os
ada_recall_os
plt.figure(figsize=(12,6))

plt.plot(ratios,ada_acc_os,linestyle='-',marker='o',color='red',label='ada_acc')

plt.plot(ratios,ada_recall_os,linestyle='-',marker='o',color='red',label='ada_rec',alpha=0.5)

plt.plot(ratios,rf_acc_os,linestyle='-',marker='o',color='blue',label='rf_acc')

plt.plot(ratios,rf_recall_os,linestyle='-',marker='o',color='blue',label='rf_rec', alpha=0.5)



plt.legend()
default_count = y_train.value_counts()[1]

X_train.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)

bad_indices = y_train[y_train==1].index

majority_count = int(np.floor(default_count / 0.6))

good_indices = np.random.choice(y_train[y_train==0].index,size=majority_count) 

indices = np.concatenate((bad_indices,good_indices))

X_train_tf = X_train.iloc[indices]

y_train_tf = y_train.iloc[indices]
emp_length = tf.feature_column.numeric_column('emp_length_int')

ann_inc = tf.feature_column.numeric_column('annual_inc')

loan_amt = tf.feature_column.numeric_column('loan_amount')

int_rate = tf.feature_column.numeric_column('interest_rate')

dti = tf.feature_column.numeric_column('dti')
mortgage = tf.feature_column.numeric_column('MORTGAGE')

own = tf.feature_column.numeric_column('OWN')

rent = tf.feature_column.numeric_column('RENT')

high = tf.feature_column.numeric_column('High')

low = tf.feature_column.numeric_column('Low')

medium = tf.feature_column.numeric_column('Medium')

short = tf.feature_column.numeric_column('term_1')

long = tf.feature_column.numeric_column('term_2')

credit = tf.feature_column.numeric_column('credit_card')

debt = tf.feature_column.numeric_column('debt_consolidation')

low_int = tf.feature_column.numeric_column('int_Low')
feat_cols = [emp_length,ann_inc,loan_amt,int_rate,dti,mortgage,own,rent,high,low,medium,short,long,credit,debt,low_int]
input_func = tf.estimator.inputs.pandas_input_fn(X_train_tf,y_train_tf,

                                                batch_size=10000,

                                                num_epochs=1000,

                                                shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10,10,10],

                                      feature_columns=feat_cols,

                                      n_classes=2)
dnn_model.train(input_fn=input_func,steps=5000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,

                                                     y=y_test,

                                                     batch_size=10000,

                                                     num_epochs=1,

                                                     shuffle=False)
dnn_model.evaluate(eval_input_func)