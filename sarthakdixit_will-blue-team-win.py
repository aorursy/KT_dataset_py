import pandas as pd

df = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
df.head()
df.describe(include="all")
df = df.drop("gameId", axis=1)
# Cleaning Data

for i in range(1, 19):

    df[df.columns[i]].fillna(df[df.columns[i]].mean(), inplace = True)

df.head()
X = df.iloc[:, 1:19].values

y = df.iloc[:, 0].values
# Splitting dataset into training and testing set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 10)
# Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {}'.format(logreg.score(X_test, y_test)))
# Confusion Matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
# Classification Report

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
# ROC Curve

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

y_pred_proba = logreg.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Logistic Regression')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC curve')

plt.show()
from sklearn.metrics import roc_auc_score

print("ROC Accuracy: {}".format(roc_auc_score(y_test,y_pred_proba)))