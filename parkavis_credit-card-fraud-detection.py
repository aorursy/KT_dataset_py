# Imported Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Classifier Libraries
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
credit_card = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
credit_card.head()
credit_card.describe()
credit_card.shape
credit_card.info()
#checking for null values in features
credit_card.isnull().sum()
print('No Frauds', round(credit_card['Class'].value_counts()[0]/len(credit_card) * 100,2), '% of the dataset')
print('Frauds', round(credit_card['Class'].value_counts()[1]/len(credit_card) * 100,2), '% of the dataset')
sns.countplot('Class', data=credit_card)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
fig, ax = plt.subplots(1, 2, figsize=(18,4))
amount_val = credit_card['Amount'].values
time_val = credit_card['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()
X = credit_card.iloc[:,:-1]
X.shape
corr_mtx = credit_card.corr()
f, ax = plt.subplots(figsize=(16, 14))
ax = sns.heatmap(corr_mtx,annot=False,cmap="YlGnBu")
X.hist(figsize=(20,21))
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(credit_card[credit_card['Class']==1].Time, credit_card[credit_card['Class']==1].Amount)
ax1.set_title('Fraud')
ax2.scatter(credit_card[credit_card['Class']==0].Time, credit_card[credit_card['Class']==0].Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(credit_card[credit_card['Class']==1].Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(credit_card[credit_card['Class']==0].Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show()
#modifying the dataset
credit_card_new = credit_card.drop(columns=['Time', 'V1', 'V2', 'V3','V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18','V20','V22', 'V23','V24', 'V25', 'V26', 'V27', 'V28', 'Amount'],axis=1)
credit_card_new.sample(10)
credit_card_new["Class"].value_counts()
y = credit_card_new.iloc[:,-1]
X = credit_card_new.iloc[:,:-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

kval = range(1, 6)
inertias = []
for k in kval:
    model = KMeans(n_clusters=k)
    model.fit(X_train)
    inertias.append(model.inertia_)
    
plt.plot(kval, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(kval)
plt.show()
model = KNeighborsClassifier(n_neighbors=4)
k_labels = model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test,y_test)