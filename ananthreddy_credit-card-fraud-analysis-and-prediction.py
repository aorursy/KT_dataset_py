%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

df = pd.read_csv('../input/creditcard.csv')
print (df.shape)
df.columns
X = df[['Class']]
y = df.drop('Class',1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
df.describe()
print ("Fraud")
print (df.Amount[df.Class == 1].describe())
print ()
print ("Normal")
print (df.Amount[df.Class == 0].describe())
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,6))

bins = 100

ax1.hist(df.Time[df.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(df.Time[df.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Number of Transactions')
plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,6))

ax1.scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1])
ax1.set_title('Fraud')

ax2.scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0])
ax2.set_title('Normal')

plt.ylabel('Amount ($)')
plt.xlabel('Time')
plt.show()
rf = RandomForestClassifier()
rf.fit(y.values, X.values.ravel())

importance = rf.feature_importances_
importance = pd.DataFrame(importance, index = y.columns, columns = ['Importance'])

feats = {}
for feature, importance in zip(y.columns,rf.feature_importances_):
    feats[feature] = importance
    
print (feats)
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)
plt.subplots(figsize=(15,10))
y_cols = y.columns.tolist()
corr = df[y_cols].corr()

sns.heatmap(corr)
rf = RandomForestClassifier(n_estimators = 20)
rf.fit(y_train, X_train)
predicted_data = rf.predict(y_test)
print (rf.score(y_test, X_test))
confusion_matrix(predicted_data, X_test)