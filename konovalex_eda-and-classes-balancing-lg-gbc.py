import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



%matplotlib inline
import os

print(os.listdir("../input/bank-customer-churn-modeling"))
data = pd.read_csv('../input/bank-customer-churn-modeling/Churn_Modelling.csv')
data.info()
data.head(15)
data.duplicated().sum()
print(data.Gender.value_counts())

print(data.Gender.value_counts(normalize=True))

sns.catplot(kind='count', data=data, x='Gender', height=6, aspect=1.3, hue="Exited");
plt.figure(figsize=(18, 8))

sns.countplot(x=data.Age, data=data);
plt.figure(figsize=(14, 8))

sns.countplot(x='Tenure', data=data);
plt.figure(figsize=(15, 8))

sns.distplot(data.Balance);
plt.figure(figsize=(15, 8))

sns.boxplot(x='Exited', y='Balance', data=data);
data[data['Exited'] == 0]['Balance'].describe()
data[data['Exited'] == 1]['Balance'].describe()
print(data.NumOfProducts.value_counts())

print(data.NumOfProducts.value_counts(normalize=True))

sns.catplot(kind='count', data=data, x='NumOfProducts', hue="Exited", height=6, aspect=1.3);
print(data.HasCrCard.value_counts())

print(data.HasCrCard.value_counts(normalize=True))

sns.catplot(kind='count', data=data, x='HasCrCard', height=6, aspect=1.3, hue="Exited");
print(data.IsActiveMember.value_counts())

print(data.IsActiveMember.value_counts(normalize=True))

sns.catplot(kind='count', data=data, x='IsActiveMember', height=6, aspect=1.3, hue="Exited");
data.EstimatedSalary.describe()
sns.catplot(x='Exited', y='EstimatedSalary', data=data, height=6, aspect=2);
plt.figure(figsize=(18, 8))

sns.boxplot(x='EstimatedSalary', data=data);
data[data['EstimatedSalary'] < 50000]['EstimatedSalary'].describe()
data[data['EstimatedSalary'] < 2000]['EstimatedSalary'].describe()
print(data.Exited.value_counts())

print(data.Exited.value_counts(normalize=True))

sns.catplot(kind='count', data=data, x='Exited', height=6, aspect=1.3);
target = data['Exited']

data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1, inplace=True)
l = list(data.columns)



l.remove('Geography')

l.remove('Gender')

l.remove('NumOfProducts')

l.remove('HasCrCard')

l.remove('IsActiveMember')

numeric = l
numeric
data[numeric].head()
data = pd.get_dummies(data, drop_first=True)
data.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(data[numeric])

data[numeric] = scaler.transform(data[numeric])
data.head()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
data, features_test, target, target_test = train_test_split(data, target, test_size=0.2, random_state=42)
features_train, features_valid, target_train, target_valid = train_test_split(data, target, test_size=0.25, random_state=42)
features_train.shape
features_valid.shape
model = LogisticRegression(solver='liblinear', random_state=42)

model.fit(features_train, target_train)

prediction = model.predict(features_valid)
f1_score(target_valid, prediction)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve
proba = model.predict_proba(features_valid)

proba = proba[:, 1]

proba

auc_roc = roc_auc_score(target_valid, proba)

print(auc_roc)
fpr, tpr, thresholds = roc_curve(target_valid, proba)



plt.figure(figsize=(12, 12))

plt.plot([0, 1], [0, 1])

plt.plot(fpr, tpr, linestyle='--')



plt.xlim([0, 1])

plt.ylim([0, 1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')



plt.show()
from sklearn.ensemble import GradientBoostingClassifier
for i in range(1, 502, 50):

    model = GradientBoostingClassifier(n_estimators=i, random_state=42)

    model.fit(features_train, target_train)

    print(i, model.score(features_valid, target_valid))
%%time

model_gbc = GradientBoostingClassifier(n_estimators=200, random_state=42)

model_gbc.fit(features_train, target_train)

print(model_gbc.score(features_train, target_train))

print(model_gbc.score(features_valid, target_valid))
prediction = model_gbc.predict(features_valid)

prediction
f1_score(target_valid, prediction)
proba_gbc = model_gbc.predict_proba(features_valid)

proba_gbc = proba_gbc[:, 1]

proba_gbc
auc_roc = roc_auc_score(target_valid, proba_gbc)

print(auc_roc)
fpr, tpr, thresholds = roc_curve(target_valid, proba_gbc)
plt.figure(figsize=(12, 12))

plt.plot([0, 1], [0, 1])

plt.plot(fpr, tpr, linestyle='--')



plt.xlim([0, 1])

plt.ylim([0, 1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')



plt.show()
from sklearn.utils import shuffle
def upsample(features, target, repeat):

    features_zeros = features[target == 0]

    features_ones = features[target == 1]

    target_zeros = target[target == 0]

    target_ones = target[target == 1]



    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)

    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    

    features_upsampled, target_upsampled = shuffle(

        features_upsampled, target_upsampled, random_state=42)

    

    return features_upsampled, target_upsampled
features_upsampled, target_upsampled = upsample(features_train, target_train, 4)
print(features_train[target_train == 0].shape)

print(features_train[target_train == 1].shape)
print(features_upsampled[target_upsampled == 0].shape)

print(features_upsampled[target_upsampled == 1].shape)
model_lr = LogisticRegression(solver='liblinear', random_state=42)

model_lr.fit(features_upsampled, target_upsampled)

predicted_valid = model_lr.predict(features_valid)



print("F1:", f1_score(target_valid, predicted_valid))
proba = model_lr.predict_proba(features_valid)

proba = proba[:, 1]

proba

auc_roc = roc_auc_score(target_valid, proba)

print(auc_roc)
fpr, tpr, thresholds = roc_curve(target_valid, proba)



plt.figure(figsize=(12, 12))

plt.plot([0, 1], [0, 1])

plt.plot(fpr, tpr, linestyle='--')



plt.xlim([0, 1])

plt.ylim([0, 1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')



plt.show()
%%time

model_gbc = GradientBoostingClassifier(n_estimators=300, random_state=42)

model_gbc.fit(features_upsampled, target_upsampled)

print(model_gbc.score(features_upsampled, target_upsampled))

print(model_gbc.score(features_valid, target_valid))
prediction = model_gbc.predict(features_valid)

prediction
f1_score(target_valid, prediction)
proba_gbc = model_gbc.predict_proba(features_valid)

proba_gbc = proba_gbc[:, 1]

proba_gbc
auc_roc = roc_auc_score(target_valid, proba_gbc)

print(auc_roc)
fpr, tpr, thresholds = roc_curve(target_valid, proba_gbc)



plt.figure(figsize=(12, 12))

plt.plot([0, 1], [0, 1])

plt.plot(fpr, tpr, linestyle='--')



plt.xlim([0, 1])

plt.ylim([0, 1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')



plt.show()
model_final = LogisticRegression(solver='liblinear', random_state=42)

model_final.fit(features_upsampled, target_upsampled)

prediction = model_final.predict(features_test)
f1_score(target_test, prediction)
proba_final = model_final.predict_proba(features_test)

proba_final = proba_final[:, 1]

proba_final

auc_roc = roc_auc_score(target_test, proba_final)

print(auc_roc)
fpr, tpr, thresholds = roc_curve(target_test, proba_final)



plt.figure(figsize=(12, 12))

plt.plot([0, 1], [0, 1])

plt.plot(fpr, tpr, linestyle='--')



plt.xlim([0, 1])

plt.ylim([0, 1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')



plt.show()
model_gbc_final = GradientBoostingClassifier(n_estimators=300, random_state=42)

model_gbc_final.fit(features_upsampled, target_upsampled)

print(model_gbc_final.score(features_upsampled, target_upsampled))

print(model_gbc_final.score(features_valid, target_valid))

print(model_gbc_final.score(features_test, target_test))
prediction = model_gbc_final.predict(features_test)

prediction
f1_score(target_test, prediction)
proba_gbc_final = model_gbc_final.predict_proba(features_test)

proba_gbc_final = proba_gbc_final[:, 1]

proba_gbc_final
auc_roc = roc_auc_score(target_test, proba_gbc_final)

print(auc_roc)
fpr, tpr, thresholds = roc_curve(target_test, proba_gbc_final)



plt.figure(figsize=(12, 12))

plt.plot([0, 1], [0, 1])

plt.plot(fpr, tpr, linestyle='--')



plt.xlim([0, 1])

plt.ylim([0, 1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')



plt.show()
for alpha in np.linspace(0.1, 1.0, 10):

    print(alpha)

    kombo=alpha*proba_final+(1-alpha)*proba_gbc_final

    auc_roc = roc_auc_score(target_test, kombo)

    print(auc_roc)

    kombo_final = np.rint(kombo)

    print(f1_score(target_test, kombo_final))

    print()
kombo=0.3*proba_final+(1-0.3)*proba_gbc_final



fpr, tpr, thresholds = roc_curve(target_test, kombo)



plt.figure(figsize=(12, 12))

plt.plot([0, 1], [0, 1])

plt.plot(fpr, tpr, linestyle='--')



plt.xlim([0, 1])

plt.ylim([0, 1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')



plt.show()