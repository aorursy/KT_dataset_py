# Finding fraudulent transactions in Data
# Importing libraries



import pandas as pd

import seaborn as sb

import numpy as np

from datetime import datetime

from matplotlib import pyplot as plt

from datetime import date
# Importing data



df = pd.read_csv('../input/fraud-detection/Fraud_Data.csv')

df.head(10)
df.describe()
df.dtypes
# checking for missing values



df.isnull().sum()
# So there are no missing values
# Descriptive analysis



# Fraud cases out of total population



fraud_val = df.is_fraud.value_counts()

fig1, ax1 = plt.subplots()

ax1.pie(fraud_val.values, labels=fraud_val.index, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
# So less than 10 % are values are fraud
# Creating new column



df['signup_time'] = pd.to_datetime(df['signup_time'])

df['purchase_time'] = pd.to_datetime(df['purchase_time'])

df['timediff']=(df['purchase_time']-df['signup_time']).astype('timedelta64[m]')

df['timediff'].head(10)
sb.distplot(df[df['is_fraud']==1]['timediff'],bins=50, kde = False)

sb.distplot(df[df['is_fraud']==0]['timediff'],bins=50, kde = False)
# Clearly, transactions with a small difference between signup time and transaction time are fraud transactions
# Lets look at purchase value now



sb.distplot(df[df['is_fraud']==1]['purchase_value'],bins=50)

sb.distplot(df[df['is_fraud']==0]['purchase_value'],bins=50)
# Purchase value seems to be very similar for both fraud and non-fraud
# Lets look at age now



sb.distplot(df[df['is_fraud']==1]['age'],bins=50)

sb.distplot(df[df['is_fraud']==0]['age'],bins=50)
# Age distribution appears to be similar as well with outliers in non-fraud but not in fraud which is weird
# Looking at the data again



df.head(10)
# Intializing the modeling process



# Dropping unnecessary columns



df2=df.drop(['user_id', 'signup_time','purchase_time','device_id','ip_address'], axis=1)

df2.head(10)
# Encoding categorical variables



def encoding(df2):

    for column in df2.columns[df2.columns.isin(['source','browser','sex'])]:

        df2[column]=df2[column].factorize()[0]

    return df2



df3 = encoding(df2)
df3.head(10)
# Since target variable 'is_fraud' is highly biased, smoting is required to make the dataset more balanced



from imblearn.over_sampling import SMOTE
# Splitting the dataa into training and testing



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
x = df3.iloc[:,[0,1,2,3,4,6]]

y=df3.iloc[:,5]



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2, random_state=123)

x_train.head(10)
# Applying SMOTE



y_train.value_counts()
smt = SMOTE()

x_train, y_train = smt.fit_sample(x_train, y_train)
y_train.value_counts()
# Now building a logistic regression model



lr = LogisticRegression()

lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
# Looking at metrics



accuracy_score(y_test, y_pred)
# Accuracy is not terrible



# Lets look at confusion metrics



confusion_matrix(y_test, y_pred)
# Looking at recall



recall_score(y_test, y_pred)
# Our true metric should be recall because we can afford to classify a non-fraudulent transaction as fraudulent but not the other way round
# Also checking f-1 score



f1_score(y_test, y_pred)
# Lets also try out decision tree to get another perspective



dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)
y_pred2 = dt.predict(x_test)
accuracy_score(y_test, y_pred2)
# Better accuracy score, but what about recall and f1 ?



recall_score(y_test, y_pred2)
f1_score(y_test, y_pred2)
confusion_matrix(y_test, y_pred2)
# Much better accuracy with slightly lower recall, I'll go with this model !
# Lets try with Random Forest



from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(x_train, y_train)
y_pred3 = rf.predict(x_test)

accuracy_score(y_test, y_pred3)
# So accuracy is better than previous 2 models



recall_score(y_test, y_pred3)
confusion_matrix(y_test, y_pred3)
y_test.value_counts()
# I would finally go with logistic regression because false negatives are minimum in that case