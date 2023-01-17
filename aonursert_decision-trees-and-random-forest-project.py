import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
loans = pd.read_csv("../input/lending-club-data/loan_data.csv")
loans.info()
loans.describe()
loans.head()
sns.set_style("whitegrid")
fig = plt.figure(figsize=(15,4))
sns.distplot(loans[loans["credit.policy"] == 1]["fico"], kde=False, label="Credit Policy = 1")
sns.distplot(loans[loans["credit.policy"] == 0]["fico"], kde=False, label="Credit Policy = 0")
plt.legend(loc=0)
sns.set_style("whitegrid")
fig = plt.figure(figsize=(15,4))
sns.distplot(loans[loans["not.fully.paid"] == 0]["fico"], kde=False, label="Not Fully Paid = 0")
sns.distplot(loans[loans["not.fully.paid"] == 1]["fico"], kde=False, label="Not Fully Paid = 1")
plt.legend(loc=0)
fig = plt.figure(figsize=(10,6))
sns.countplot(x="purpose", data=loans, hue="not.fully.paid")
sns.jointplot(x="fico", y="int.rate", data=loans)
sns.lmplot(x="fico", y="int.rate", col="not.fully.paid", data=loans, hue="credit.policy")
loans.info()
loans.head()
cat_feats = ["purpose"]
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)
loans.head()
final_data.head()
final_data.columns
from sklearn.model_selection import train_test_split
X = final_data.drop("not.fully.paid", axis=1)
y = final_data["not.fully.paid"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_predictions = rfc.predict(X_test)
print(classification_report(y_test, rfc_predictions))
print(confusion_matrix(y_test, rfc_predictions))
# Random Forest