import numpy as np

import pandas as pd

df = pd.read_csv("../input/seattleWeather_1948-2017.csv")

df.head()
df = df.dropna()

X = df.loc[:, ['PRCP', 'TMAX', 'TMIN']].shift(-1).iloc[:-1].values

y = df.iloc[:-1, -1:].values.astype('int')



from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()

clf.fit(X, y)

y_hat = clf.predict(X)
from sklearn.metrics import accuracy_score



accuracy_score(y, y_hat)
from sklearn.metrics import confusion_matrix



confusion_matrix(y, y_hat)
import seaborn as sns

sns.heatmap(confusion_matrix(y, y_hat) / len(y), cmap='Blues', annot=True)
from sklearn.metrics import hamming_loss



hamming_loss(y, y_hat)
from sklearn.metrics import precision_score, recall_score, precision_recall_curve



print(precision_score(y, y_hat))

print(recall_score(y, y_hat))
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



precision, recall, _ = precision_recall_curve(y, y_hat)



fig, ax = plt.subplots(1, figsize=(12, 6))

ax.step(recall, precision, color='steelblue',

         where='post')

ax.fill_between(recall, precision, step='post', color='lightgray')

plt.suptitle('Precision-Recall Tradeoff for Seattle Rain Prediction')

plt.xlabel('Recall')

plt.ylabel('Precision')
from sklearn.metrics import fbeta_score



fbeta_score(y, y_hat, beta=1)
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y, y_hat)



fig, ax = plt.subplots(1, figsize=(12, 6))

plt.plot(fpr, tpr, color='darkorange', label='Model Performace')

plt.plot([0, 1], [0, 1], color='gray', label='Random Performace')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Seattle Rain ROC Curve')

plt.legend(loc="lower right")
from sklearn.metrics import roc_auc_score



roc_auc_score(y, y_hat)