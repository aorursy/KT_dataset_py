import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix , roc_curve, roc_auc_score
heart_df = pd.read_csv('../input/heart-disease-uci/heart.csv')

heart_df
heart_df.info()
heart_df.describe(percentiles=[0.25, 0.50, 0.75, 0.85, 0.95, 1])
sns.distplot(heart_df['oldpeak'], bins=25)

plt.show()
X = heart_df.drop('target', axis=1)

y = heart_df['target']
sc = StandardScaler()

X = pd.DataFrame(data=sc.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)



y_pred = log_reg.predict(X_test)
confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

(tn, fp, fn, tp)
def draw_roc(actual , probs):

  fpr,tpr,thresholds = roc_curve(actual, probs, drop_intermediate=False)



  auc_score = roc_auc_score(actual, probs)

  plt.figure(figsize=(6,4))

  plt.plot(fpr,tpr, label='ROC curve ( area = %0.2f)'% auc_score)

  plt.plot([0,1],[0,1],'k--')



  plt.xlim([0.0,1.0])

  plt.ylim([0.0,1.05])

  plt.xlabel('False Positive Rate or [1- True Negative Rate]')

  plt.ylabel('True Positive Rate')

  plt.title('Receiver operating Characterstics example')

  plt.legend(loc='lower right')

  plt.show()



  return fpr ,tpr , thresholds



draw_roc(y_test, y_pred)  