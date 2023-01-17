import numpy as np

import pandas as pd

from pandas import DataFrame,Series



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/shopee-code-league-20/_DA_Marketing_Analytics/train.csv",index_col='row_id')
df.head()
df.info()
df['last_open_day'] = df['last_open_day'].replace('Never open',0).astype(int)

df['last_login_day'] = df['last_login_day'].replace('Never login',0).astype(int)

df['last_checkout_day'] = df['last_checkout_day'].replace('Never checkout',0).astype(int)
df = df.drop(['grass_date','subject_line_length','user_id'],axis=1)
df.describe()
sns.countplot('open_flag',data=df)
sns.countplot('country_code',data=df)
fig,(ax1,ax2) = plt.subplots(1,2)



xmax = df['last_open_day'].max()

xmin = df['last_open_day'].min()



plt.xlim((xmin,xmax))

sns.kdeplot(df['last_open_day'],ax=ax1)



sns.kdeplot(df['last_open_day'],ax=ax2)

ax2.set_xlim(0,100)



ax1.set_title('Last open day')

ax2.set_title('Last open within 100days')

fig.tight_layout()

fig.set_size_inches(10,5)
fig,(ax1,ax2,ax3) = plt.subplots(1,3)



sns.kdeplot(df['open_count_last_10_days'],ax=ax1)

ax1.set_xlim(0,10)



sns.kdeplot(df['open_count_last_30_days'],ax=ax2)

ax2.set_xlim(0,30)



sns.kdeplot(df['open_count_last_60_days'],ax=ax3)

ax3.set_xlim(0,60)



ax1.set_title('open_count_last_10_days')

ax2.set_title('open_count_last_30_days')

ax3.set_title('open_count_last_60_days')

fig.set_size_inches(20,5)
fig,(ax1,ax2) = plt.subplots(1,2)



xmax = df['last_login_day'].max()

xmin = df['last_login_day'].min()



plt.xlim((xmin,xmax))

sns.kdeplot(df['last_login_day'],ax=ax1)



sns.kdeplot(df['last_login_day'],ax=ax2)

ax2.set_xlim(0,1500)



ax1.set_title('Last login day')

ax2.set_title('Last login within 1500days')

fig.tight_layout()

fig.set_size_inches(10,5)
fig,(ax1,ax2,ax3) = plt.subplots(1,3)



sns.kdeplot(df['login_count_last_10_days'],ax=ax1)

ax1.set_xlim(0,10)



sns.kdeplot(df['login_count_last_30_days'],ax=ax2)

ax2.set_xlim(0,30)



sns.kdeplot(df['login_count_last_60_days'],ax=ax3)

ax3.set_xlim(0,60)



ax1.set_title('login_count_last_10_days')

ax2.set_title('login_count_last_30_days')

ax3.set_title('login_count_last_60_days')

fig.set_size_inches(20,5)
fig,(ax1,ax2) = plt.subplots(1,2)



xmax = df['last_checkout_day'].max()

xmin = df['last_checkout_day'].min()



plt.xlim((xmin,xmax))

sns.kdeplot(df['last_checkout_day'],ax=ax1)



sns.kdeplot(df['last_checkout_day'],ax=ax2)

ax2.set_xlim(0,500)



ax1.set_title('Last checkout day')

ax2.set_title('Last checkout within 500days')

fig.tight_layout()

fig.set_size_inches(10,5)
fig,(ax1,ax2,ax3) = plt.subplots(1,3)



sns.kdeplot(df['checkout_count_last_10_days'],ax=ax1)

ax1.set_xlim(0,10)



sns.kdeplot(df['checkout_count_last_30_days'],ax=ax2)

ax2.set_xlim(0,30)



sns.kdeplot(df['checkout_count_last_60_days'],ax=ax3)

ax3.set_xlim(0,60)



ax1.set_title('checkout_count_last_10_days')

ax2.set_title('checkout_count_last_30_days')

ax3.set_title('checkout_count_last_60_days')

fig.set_size_inches(20,5)
sns.heatmap(df.corr(),annot=True)

fig = plt.gcf()

fig.set_size_inches(15,5)
Y = df['open_flag']

#Y = Y.values.reshape(-1,1)

X = df.drop(['open_count_last_10_days', 'open_count_last_30_days',

       'open_count_last_60_days', 'login_count_last_10_days',

       'login_count_last_30_days', 'login_count_last_60_days',

       'checkout_count_last_10_days', 'checkout_count_last_30_days',

       'checkout_count_last_60_days', 'open_flag'],axis=1)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X,Y)
log_model.score(X,Y)
coeff_df = DataFrame(zip(X.columns,np.transpose(log_model.coef_)))

coeff_df
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(X,Y)



print(f'Splitting happens as {x_train.shape},{x_test.shape},{y_train.shape},{y_test.shape}')
log_model2 = LogisticRegression(class_weight='balance')



log_model2.fit(x_train,y_train)



y_pred = log_model2.predict(x_test)

y_pred
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
from sklearn.svm import SVC
model = SVC()
clf = model.fit(x_train,y_train)
model.score(X,Y)
y_pred = clf.predict(x_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)