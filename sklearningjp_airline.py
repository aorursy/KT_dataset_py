import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df_tr = pd.read_csv('../input/airline-passenger-satisfaction/train.csv')
df_te = pd.read_csv('../input/airline-passenger-satisfaction/test.csv')
print(df_tr.shape)
print(df_te.shape)
df_tr.head()
df_tr.columns
df_te.columns
df_tr.isnull().sum()
df_te.isnull().sum()
df_tr['satisfaction'].unique()
df_tr['satisfaction'].value_counts()
sns.countplot(x='satisfaction',data=df_tr,palette='Set2')
plt.show()
sns.countplot(x='Type of Travel',data=df_tr,palette='Set2')
plt.show()
sns.countplot(x='Customer Type',data=df_tr,palette='Set2')
plt.show()
sns.countplot(x='Gender',data=df_tr,palette='Set2')
plt.show()
sns.countplot(x='Class',data=df_tr,palette='Set2')
plt.show()
df_tr.groupby(['Class', 'Gender'])['satisfaction'].count()
sns.distplot(df_tr['Age'])
sns.distplot(df_tr['Flight Distance'])
corr_matrix = df_tr.corr()
corr_matrix
df_tr.dtypes
df_tr = pd.get_dummies(df_tr, columns=["Gender", "Customer Type", "Type of Travel", "Class", "satisfaction"], sparse=True)
df_tr.head()
df_tr.drop(columns=['satisfaction_neutral or dissatisfied', 'Class_Eco Plus', 'Class_Eco',
                   'Type of Travel_Personal Travel', 'Customer Type_disloyal Customer',
                   'Gender_Male'], inplace=True)
df_tr.drop(columns=['Unnamed: 0', 'id'], inplace=True)
df_tr.drop(columns=['Arrival Delay in Minutes'], inplace=True)
df_tr.head()
df_corr = df_tr.corr()
df_corr
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(df_corr, annot=True, linewidth=".5", fmt=".2f")
plt.show()
df_tr.columns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_list = ['Age', 'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Gender_Female',
       'Customer Type_Loyal Customer', 'Type of Travel_Business travel',
       'Class_Business']
y_list = ['satisfaction_satisfied']

X = df_tr[X_list]
y = df_tr[y_list]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
logit = LogisticRegression()
logit.fit(X_train, y_train)
logit.coef_
logit.intercept_
y_pred = logit.predict(X_test)
y_pred
accuracy_score(y_test, y_pred)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=12)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
from sklearn.ensemble import RandomForestClassifier
clf_r = RandomForestClassifier(n_estimators=10, max_depth=12)
clf_r.fit(X_train, y_train)
y_pred = clf_r.predict(X_test)
accuracy_score(y_test, y_pred)
importances = clf_r.feature_importances_
importances
features = np.array(df_tr.columns)
factor = np.argsort(importances)
plt.figure(figsize=(6, 6))
plt.barh(range(len(factor)), importances[factor], color='b', align='center')
plt.yticks(range(len(factor)), features[factor])
plt.show()