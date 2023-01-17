import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('max.columns', None)
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

%matplotlib inline
df = pd.read_csv('../input/Pokemon.csv', low_memory=False)
df.info()
df.head()
## Percentage of Legendaries in the dataset
print('Legendary:',str(len(df[df['Legendary'] == True]) / len(df) * 100) + '%')
plt.title('Count Plot')
plt.xticks(rotation = 45)
sns.countplot(df['Type 1'])

# Expected Fire type to be the highest
plt.title('Count Plot')
plt.xticks(rotation = 45)
sns.countplot(df['Type 2'])
sns.distplot(df['Total'])
## Break down of the Generations
df['Generation'].value_counts()
sns.pairplot(df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']])
corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', annot=True)
df.describe()
df[df['Name'].duplicated()] # no dupliactes
pd.crosstab(df['Type 1'] , df['Legendary'])
for i in df.columns:
    print(i, len(df[i].unique()))
df['Legendary'] = df['Legendary'].apply(lambda x: 1 if x == True else 0)
dataset = df.iloc[:, 2:]
dataset.head()
dataset = pd.get_dummies(dataset, dummy_na=True,drop_first=True)
dataset['Target'] = dataset['Legendary']
dataset.drop(['Legendary', 'Total'], inplace=True, axis=1)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
y.head(2)
X.head()
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
clr = LogisticRegression()
clr.fit(X_train, y_train)
y_pred = clr.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
cm
probs = clr.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
## SVC
svc = SVC(probability=True)
svc.fit(X_train, y_train)
svc_probs = svc.predict_proba(X_test)
svc_preds = svc_probs[:,1]
svc_fpr, svc_tpr, svc_threshold = metrics.roc_curve(y_test, svc_preds)
svc_roc_auc = metrics.auc(svc_fpr, svc_tpr)
svc_y_pred = svc.predict(X_test)
accuracy_score(y_test, svc_y_pred)
tpr
svc_tpr
cm
svc_cm = confusion_matrix(y_test, svc_y_pred)
svc_cm
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'LR AUC = %0.2f' % roc_auc)
plt.plot(svc_fpr, svc_tpr, 'g', label = 'SVC AUC = %0.2f' % svc_roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
