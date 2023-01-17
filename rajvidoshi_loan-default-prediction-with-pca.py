import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/loan-default-prediction/train_v2.csv.zip")

t = pd.read_csv("/kaggle/input/loan-default-prediction/test_v2.csv.zip")

data.shape
data.head()
data.info()
data.select_dtypes(include=['object']).head()
invalid = data.select_dtypes(include=['object']).columns

data.drop(invalid, axis=1, inplace=True)

t.drop(invalid, axis=1, inplace=True)

t_id = t['id'].copy

t.drop('id', axis=1, inplace = True)
data.describe()
t.describe()
missing = data.isnull().sum()

missing = pd.DataFrame(missing[missing!=0])

missing.columns = ['No. of missing values']

missing['Percentage'] = 100*missing['No. of missing values']/data.id.count()

missing.sort_values(by="Percentage", ascending=False)
correlations = data.iloc[:,1:752].corr()

correlations.head()
x = data.iloc[:,1:751].copy()

y = data.iloc[:,751].copy()

y.value_counts()
y[y>0] = 1

y.value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, stratify = y, random_state=0)
[X_train.shape, X_test.shape, y_train.shape, y_test.shape]
X_train = X_train.fillna(X_train.mean())

X_test = X_test.fillna(X_train.mean())

t = t.fillna(X_train.mean())

[X_train.isnull().sum().sum(), X_test.isnull().sum().sum(), t.isnull().sum().sum()]
from sklearn.preprocessing import StandardScaler

scalar= StandardScaler()

scalar.fit(X_train)

X_train = scalar.transform(X_train)

X_test = scalar.transform(X_test)

X_t = scalar.transform(t)
from sklearn.decomposition import PCA

pca = PCA()

pca.fit(X_train)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');
np.cumsum(pca.explained_variance_ratio_)[200]
final_pca = PCA(n_components=200)

final_pca.fit(X_train)

X_train = final_pca.transform(X_train)

X_train = pd.DataFrame(data = X_train)

X_test = final_pca.transform(X_test)

X_test = pd.DataFrame(data = X_test)

X_t = final_pca.transform(X_t)

X_t = pd.DataFrame(data = X_t)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver= 'saga', class_weight='balanced',max_iter=500, random_state=1).fit(X_train, y_train)

model.coef_[0]
y_pred = model.predict(X_test)

y_pred
import sklearn.metrics as sm

c = pd.DataFrame(sm.confusion_matrix(y_test, y_pred), index=['Actual non defaulter','Actual defaulter'])

c.columns = ['Predicted non defaulter','Predicted defaulter']

c['Actual Total'] = c.sum(axis=1)

c.loc['Predicted Total',:] = c.sum(axis = 0)

c
print(["The accuracy on the validation data is " + str(round(sm.accuracy_score(y_test, y_pred)*100,ndigits = 2)) + "%"])
print("The sensitivity (true positive rate) is " + str(round(100*c.iloc[1,1]/c.iloc[1,2], ndigits=2)) + "%")
ns_fpr, ns_tpr, _ = sm.roc_curve(y_test, np.zeros(len(y_test)))

lr_probs = model.predict_proba(X_test)

# keep probabilities for the positive outcome only

lr_probs = lr_probs[:, 1]

lr_fpr, lr_tpr, _ = sm.roc_curve(y_test, lr_probs)

# plot the roc curve for the model

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
print("The Area under ROC curve is " + str(round(100 * sm.roc_auc_score(y_test, y_pred), ndigits=2)) + "%")
print(sm.classification_report(y_test, y_pred))
pred = model.predict(X_t)

sns.countplot(pred);
submission = pd.read_csv("../input/loan-default-prediction/sampleSubmission.csv")

submission['loss'] = pred
submission.head()
submission.to_csv("submit.csv", index=False)