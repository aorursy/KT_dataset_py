import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split, cross_val_score



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.metrics import roc_curve, auc, roc_auc_score
import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
df = train_df.copy()
df.head()
df.dropna(axis=1, inplace=True)
Vectorize = TfidfVectorizer()



X = Vectorize.fit_transform(df["comment_text"])



test_X = Vectorize.transform(test_df["comment_text"])
y = np.where(train_df['target'] >= 0.5, 1, 0)
X.shape, y.shape, test_X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
lr = LogisticRegression(C=5, random_state=42, solver='sag', max_iter=1000, n_jobs=-1)
lr.fit(X_train, y_train)
cv_accuracy = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')

print(cv_accuracy)

print(cv_accuracy.mean())
y_pred = lr.predict(X_test)
plt.figure(figsize=(8, 6))

mat = confusion_matrix(y_test, y_pred)

sns.heatmap(mat, square=True, annot=True, cbar=False, cmap='Reds')

plt.xlabel('predicted value')

plt.ylabel('true value');
print(classification_report(y_test, y_pred))
fpr, tpr, thr = roc_curve(y_test, lr.predict_proba(X_test)[:,1])

#auc = auc(fpr, tpr)

auc = roc_auc_score(y_test, y_pred)

lw = 2

plt.figure(figsize=(10, 8))

plt.plot(fpr, tpr, color='darkorange', lw=lw, label="Curve Area = %0.3f" % auc)

plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic Plot')

plt.legend(loc="lower right")

plt.show()
predictions = lr.predict_proba(test_X)[:,1]
sub['prediction'] = predictions

sub.to_csv('submission.csv', index=False)
sub.head(15)