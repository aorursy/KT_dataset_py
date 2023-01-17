import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import roc_curve, auc

from sklearn import metrics

import numpy as np

data = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

data
# Check for null values

data.isna().sum()
# Check count of classified mushrooms

data['class'].value_counts()
plt.figure()

sns.countplot(x='class', data=data).set_title('Count of each class')
# Converting the data to numerical

df = data.apply(lambda col: pd.factorize(col)[0])

# Deleting because redundant, all values in this column are same.

df.drop(df.columns[[16]], axis=1, inplace=True)
plt.figure(figsize=(14, 8))

sns.heatmap(df.corr(), annot=True).set_title("Correlation")
fs = SelectKBest(score_func=chi2, k='all')

features = fs.fit(df.iloc[:, 1:], df['class'])

print('Chi-Square statistic of features')

for i in range(len(fs.scores_)):

    print(f'Feature {i}: {fs.scores_[i]}\nFeature {i}: {fs.pvalues_[i]}')
plt.figure()

plt.bar([i for i in range(len(fs.scores_))], fs.scores_)

plt.title('Chi-square values of features')

x = df.iloc[:, 1:]

y = df.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)



model = GaussianNB()

model.fit(x_train, y_train)

predict = model.predict(x_test)

predict
fpr, tpr, thr = roc_curve(y_train, model.predict_proba(x_train)[:, 1])



plt.figure()

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic Plot')
plot_confusion_matrix(model, x_test, y_test,values_format='d',cmap='Blues')
auc_score = auc(fpr, tpr)

print("\nAccuracy:", metrics.accuracy_score(y_test, predict))

print(f'AUC Score: {auc_score}')