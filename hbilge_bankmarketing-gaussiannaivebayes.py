import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv('../input/bank-marketing/bank-additional-full.csv', sep = ';')
print(dataset.shape)
print(dataset.head())
print(dataset.info())
dataset = dataset.rename(columns={'y': 'subscribed'})
print(dataset.duplicated().sum())
print(dataset[dataset.duplicated(keep=False)].iloc[:,:7])
dataset = dataset.drop_duplicates()
print(dataset.shape)
print('\033[1mNULL VALUES\033[0m\n'+ str(dataset.isnull().values.any()))
Subscribed = pd.DataFrame(dataset['subscribed'].value_counts())
print(Subscribed.T)
pd.DataFrame(dataset['subscribed'].value_counts()).plot(kind='bar', color='lightgreen')
plt.show()
plt.figure(figsize=(16,4))

plt.subplot(1,4,1)
sns.distplot(dataset['age'])
plt.title('Age Distribution')

plt.subplot(1,4,2)
sns.countplot(dataset['job'])
plt.title('Job Distribution')
plt.xticks(rotation=90)

plt.subplot(1,4,3)
sns.countplot(dataset['marital'], color='pink')
plt.title('Marital Status')

plt.subplot(1,4,4)
sns.countplot(dataset['education'], color='lightgreen')
plt.xticks(rotation=90)
plt.title('Education Level')

plt.show()
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
sns.countplot(dataset['default'], palette="Set3")
plt.title('Default Credit')

plt.subplot(1,3,2)
sns.countplot(dataset['housing'], palette="Set3")
plt.title('Housing Loan')

plt.subplot(1,3,3)
sns.countplot(dataset['loan'], palette="Set3")
plt.title('Loan')

plt.show()
plt.figure(figsize=(18,5))

plt.subplot(1,4,1)
sns.countplot(dataset['contact'], palette="vlag")
plt.title('Contact Type')

plt.subplot(1,4,2)
sns.countplot(dataset['month'], palette="vlag",order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
plt.title('Month')
plt.xticks(rotation=90)

plt.subplot(1,4,3)
sns.countplot(dataset['day_of_week'], palette="vlag")
plt.title('Day Of Week')

plt.subplot(1,4,4)
sns.distplot(dataset['duration'])
plt.xticks(rotation=90)
plt.title('Duration of Calls')

plt.show()
dataset.drop('day_of_week', axis=1, inplace=True)
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.violinplot("contact", "campaign", data=dataset, kind='reg')
plt.title('Number of Contacts vs Contact Type')

plt.subplot(1,2,2)
sns.distplot(dataset['campaign'])
plt.title('Number of Contacts with Customers')

plt.show()
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
sns.countplot(dataset['pdays'])
plt.xticks(rotation=90)
plt.title('Number of Days Passed Since Previous Campaign')

plt.subplot(1,3,2)
sns.countplot(dataset['previous'])
plt.title('Number of Previous Contacts')

plt.subplot(1,3,3)
sns.countplot(dataset['poutcome'])
plt.title('Previous Campaign Result')

plt.show()
dataset.loc[dataset['pdays'] < 999, 'pdays'] = 1
dataset.loc[dataset['pdays'] == 999, 'pdays'] = 0
dataset = dataset.rename(columns={'pdays': 'previouslycontacted', 'previous':'previouscontacts'})
bins= [0,10,20,30,40,50,60,70,80,90,100]
labels = [0,1,2,3,4,5,6,7,8,9]
dataset.insert(1, 'agegroup', pd.cut(dataset['age'], bins=bins, labels=labels, right=False))
dataset = dataset.drop('age', axis=1)
categorical_columns = dataset.select_dtypes(include='object').columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in categorical_columns:
    dataset[i] = le.fit_transform(dataset[i]) 
print(dataset.head())
print(dataset.shape)
X_train = dataset.iloc[:, :-1].values.astype('float')
y_train = dataset['subscribed'].values
pd.DataFrame(X_train[y_train == 1]).plot(kind='density', ind=100, legend=False)
plt.title('Subscribed Likelihood Plots')

plt.show()
pd.DataFrame(X_train[y_train == 0]).plot(kind='density', ind=100, legend=False)
plt.title('Not Subscribed Likelihood Plots')

plt.show()
from sklearn.preprocessing import StandardScaler
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train))
X_train[y_train == 1].plot(kind='density', ind=100, legend=False)
plt.title('Subscribed Likelihood Plot after Standardization')
plt.show()
X_train[y_train == 0].plot(kind='density', ind=100, legend=False)
plt.title('Not Subscribed Likelihood Plot after Standardization')
plt.show()
plt.figure(figsize=(12,10))
sns.heatmap(dataset.corr(method='spearman'), cbar=True, cmap="RdBu_r")
plt.title("Correlation Matrix", fontsize=16)
plt.show()
correlation = X_train.corr(method='spearman').abs()
upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.40)]
X_train.drop(X_train[to_drop], axis=1, inplace=True)
print(X_train.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25, random_state=42)
from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(X_train, y_train)
pred = gb.predict(pd.DataFrame(X_test))
from sklearn.metrics import roc_curve, auc
gbprob = gb.predict_proba(X_train)[:,1]
fpr, tpr, thr = roc_curve(y_train, gbprob)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Plot')
print(auc(fpr, tpr))
from sklearn.metrics import confusion_matrix, accuracy_score
print('Accuracy score of Gaussian Naive Bayes:' + str(accuracy_score(y_test,pred)))
print('Confusion Matrix\n' + str(confusion_matrix(y_test, pred)))