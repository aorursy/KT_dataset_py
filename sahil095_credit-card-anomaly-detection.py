import matplotlib.pyplot as plt # visualization

import seaborn as sns # visualization

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline







df = pd.read_csv('../input/creditcard.csv')

df.head()
df.columns
df.dtypes
df.describe()
df.info()
df.isnull().any()
c = df['Class']

ax = sns.countplot(c, label='Frequency')

V, F = c.value_counts()



print('Number of Fraudulent: ', F)

print('Number of Valid: ', V)
df.hist(figsize=(20,20), color = "green")

plt.show()
# Correlation matrix



corr = df.corr()



fig = plt.figure(figsize=(12, 9))



sns.heatmap(corr, vmax=.8, cmap=sns.diverging_palette(180,10,as_cmap=True),square=True)

plt.show()
corr = df.corrwith(df['Class']).reset_index()

corr.columns = ['Index','Correlations']

corr = corr.set_index('Index')

corr = corr.sort_values(by=['Correlations'], ascending = False)

plt.figure(figsize=(4,15))

fig = sns.heatmap(corr, annot=True, fmt="g", cmap='RdYlGn')

plt.title("Correlation of Variables with Class")

plt.show()
plt.figure(figsize=(8,4))

fig = plt.scatter(x=df[df['Class'] == 1]['Time'], y=df[df['Class'] == 1]['Amount'], color="red")

plt.title("Time vs Transaction Amount in Fraud Cases")

plt.show()
plt.figure(figsize=(8,4))

fig = plt.scatter(x=df[df['Class'] == 0]['Time'], y=df[df['Class'] == 0]['Amount'], color="green")

plt.title("Time vs Transaction Amount in Legit Cases")

plt.show()
# importing the model

from sklearn.ensemble import IsolationForest

from sklearn.model_selection import train_test_split
valid = df[df.Class==0]

valid = valid.drop(['Class'], axis=1)

fraud = df[df.Class==1]

fraud = fraud.drop(['Class'], axis=1)

valid_train, valid_test = train_test_split(valid, test_size=0.30, random_state=42)
valid_train.head()
valid_test.head()
model = IsolationForest()

model.fit(df.drop(['Class'], axis=1))

valid_pred_test = model.predict(valid_test)

fraud_pred = model.predict(fraud)
print("Accuracy in Detecting Valid Cases:", list(valid_pred_test).count(1)/valid_pred_test.shape[0])

print("Accuracy in Detecting Fraud Cases:", list(fraud_pred).count(-1)/fraud_pred.shape[0])