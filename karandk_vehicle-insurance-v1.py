import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
df.head(5)
df.shape
df.describe()
df.info()
df.isnull().sum()
corr = df.corr()
sns.heatmap(corr,square=True,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values,
           cmap= 'coolwarm')
corr
sns.distplot(df['Annual_Premium'], bins=50)
sns.boxplot(df['Annual_Premium'])
fig = plt.figure()
plt.hist(df['Annual_Premium'], bins=50)
plt.show()
sns.distplot(df['Policy_Sales_Channel'], bins=50)
sns.countplot(df['Previously_Insured'])
sns.countplot(df['Driving_License'])
sns.distplot(df['Age'], bins=50)
plt.figure(figsize = (15, 6))
sns.distplot(df.loc[(df['Gender'] == 'Male'), 'Age'], kde_kws = {"color": "b", "lw": 1, "label": "Male"})
sns.distplot(df.loc[(df['Gender'] == 'Female'), 'Age'], kde_kws = {"color": "g", "lw": 1, "label": "Female"})
plt.title('Age distribution by Gender', fontsize = 15)
plt.show()
plt.figure(figsize = (15, 6))
sns.distplot(df.loc[(df['Gender'] == 'Male'), 'Annual_Premium'], kde_kws = {"color": "b", "lw": 1, "label": "Male"})
sns.distplot(df.loc[(df['Gender'] == 'Female'), 'Annual_Premium'], kde_kws = {"color": "r", "lw": 1, "label": "Female"})
plt.title('Annual Premium distribution by Gender', fontsize = 15)
plt.show()
plt.figure(figsize = (15, 6))
sns.distplot(df.loc[(df['Driving_License'] == 0), 'Age'], kde_kws = {"color": "b", "lw": 1, "label": "Not Licensed"})
sns.distplot(df.loc[(df['Driving_License'] == 1), 'Age'], kde_kws = {"color": "r", "lw": 1, "label": "Licensed"})
plt.title('Age distribution by Driving License', fontsize = 15)
plt.show()
plt.figure(figsize = (15, 6))
sns.distplot(df.loc[(df['Driving_License'] == 0), 'Annual_Premium'], kde_kws = {"color": "b", "lw": 1, "label": "Not Licensed"})
sns.distplot(df.loc[(df['Driving_License'] == 1), 'Annual_Premium'], kde_kws = {"color": "r", "lw": 1, "label": "Licensed"})
plt.title('Annual_Premium distribution by Driving License', fontsize = 15)
plt.show()
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Vehicle_Damage'] = le.fit_transform(df['Vehicle_Damage'])
df['Vehicle_Age'] = le.fit_transform(df['Vehicle_Age'])
df.head(1)
X = df.drop(columns=['Response'])
y = df['Response']
lr = LogisticRegression()
lr.fit(X,y)
lr_pred = lr.predict(X)
print(roc_auc_score(y,lr_pred))
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X,y)
rf_pred = rf.predict(X)
print (roc_auc_score(y,rf_pred))
gs = GaussianNB()
gs.fit(X,y)
gs_pred = gs.predict(X)
print (roc_auc_score(y,gs_pred))
test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
test['Gender'] = le.fit_transform(test['Gender'])
test['Vehicle_Damage'] = le.fit_transform(test['Vehicle_Damage'])
test['Vehicle_Age'] = le.fit_transform(test['Vehicle_Age'])
test.head(1)
testIDs = test['id']
testIDs[:5]
Final_preds = [predClass[1] for predClass in rf.predict_proba(test)]
submission = pd.DataFrame(data = {'id': testIDs, 'Response': Final_preds})
submission.to_csv('Health_Insurance_v1.csv', index = False)
submission.head()