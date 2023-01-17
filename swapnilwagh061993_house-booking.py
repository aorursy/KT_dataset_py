import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

df.head(10)
df.tail(10)
#df['arriavl_date']=pd.to_datetime(df['arrival_date_year']+df['arrival_date_month']+df['arrival_date_day_of_month'])
print('shape of dataset',df.shape)

print('\n')

print('size of dataset',df.size)
df.info()
df.describe().T
df.describe(include='object').T
df.isna().sum()
cat=df.select_dtypes(include='object').columns

cat
df=df.drop(['agent','company','reservation_status_date'],axis=1)
df['country'].mode()
df['country']=df['country'].replace(np.nan,'PRT')
df.isnull().sum()
sns.countplot(df['hotel'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.countplot(df['arrival_date_month'])

plt.show()


sns.countplot(df['is_canceled'])

plt.show()
df.is_canceled.value_counts()
plt.figure(figsize=(15 ,10 ))

sns.countplot(df['meal'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.countplot(df['market_segment'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.countplot(df['distribution_channel'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.countplot(df['reserved_room_type'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.countplot(df['assigned_room_type'])

plt.show()


sns.countplot(df['deposit_type'])

plt.show()


sns.countplot(df['customer_type'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.countplot(df['reservation_status'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.barplot(df['reservation_status'],df['arrival_date_year'],)

plt.show()
df.corr()
plt.figure(figsize=(15 ,10 ))

sns.barplot(df['arrival_date_year'],df['previous_cancellations'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.barplot(df['arrival_date_year'],df['previous_bookings_not_canceled'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.barplot(df['arrival_date_month'],df['previous_cancellations'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.barplot(df['arrival_date_month'],df['previous_bookings_not_canceled'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.barplot(df['arrival_date_month'],df['is_canceled'])

plt.show()
plt.figure(figsize=(15 ,10 ))

sns.barplot(df['arrival_date_year'],df['is_canceled'])

plt.show()
cat
df=pd.get_dummies(df,prefix=['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',

       'distribution_channel', 'reserved_room_type', 'assigned_room_type',

       'deposit_type', 'customer_type', 'reservation_status'])
df.head()
print('shape of dataset',df.shape)

print('\n')

print('size of dataset',df.size)
for i in df.columns:

    if (df[i].isnull().sum())!=0:

        print("{} {}".format(i, df[i].isnull().sum()))
df.children.mode()
df['children']=df['children'].replace(np.nan,'0')
df['children']=df['children'].astype('int')
df.corr()
plt.figure(figsize=(25 ,20 ))

sns.heatmap(df.corr())
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
X=df.drop('is_canceled',axis=1 )

y=df['is_canceled']
LR=LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=2)
LR.fit(X_train,y_train)
y_pred = LR.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

confusion_matrix
from sklearn.metrics import accuracy_score



accuracy=accuracy_score(y_test, y_pred)

accuracy 
import statsmodels.api as sm

X=df.drop('is_canceled',axis=1 )

y=df['is_canceled']
Xc=sm.add_constant(X)

model=sm.OLS(y,X).fit()

model.summary()
cols = X.columns.tolist()



while len(cols)>0:

    

    x_1 = X[cols]

    model = sm.OLS(y, x_1).fit()

    p = pd.Series(model.pvalues.values, index = cols)

    pmax = max(p)

    feature_max_p = p.idxmax()

    

    if(pmax > 0.05):

        cols.remove(feature_max_p)

    else:

        break
print(len(cols))

print(cols)
X=df[cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=2)
LR.fit(X_train,y_train)
y_pred1=LR.predict(X_test)

y_pred1
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred1)

confusion_matrix
from sklearn.metrics import accuracy_score



accuracy=accuracy_score(y_test, y_pred1)

accuracy 