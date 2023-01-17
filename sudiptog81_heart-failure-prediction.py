%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head()
df.info()
df.isna().sum()
df.duplicated().sum()
sns.countplot(df['DEATH_EVENT'])

plt.show()
for c in df[['age', 'ejection_fraction', 'creatinine_phosphokinase', 'platelets',

            'serum_creatinine', 'serum_sodium', 'time']].columns:

    sns.distplot(df[c])

    plt.show()
for c in df[['age', 'ejection_fraction', 'creatinine_phosphokinase', 'platelets',

            'serum_creatinine', 'serum_sodium', 'time']].columns:

    sns.boxplot(df[c])

    plt.show()
df = df[df['ejection_fraction'] < 70]
x_train, x_test, y_train, y_test = train_test_split(

    df[['time','ejection_fraction','serum_creatinine','age']], df['DEATH_EVENT'],

    test_size=0.2, random_state=42

)
sns.heatmap(x_train.corr())

plt.show()
x_train.describe()
# scaler = StandardScaler()



# for d in [x_train, x_test]:

#     for c in d.columns:

#         d[c] = scaler.fit_transform(d[c].values.reshape(-1,1))
from sklearn.neighbors import KNeighborsClassifier

kMax = 20

kVals = list(range(1, kMax + 1))

mean_acc = np.zeros(len(kVals))

std_acc = np.zeros(len(kVals))

for i in kVals:

    knnModel = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)

    yHat = knnModel.predict(x_test)

    mean_acc[i - 1] = np.mean(yHat == y_test);

bestK = pd.DataFrame({'k':kVals, 'mean_acc':mean_acc}).set_index('k')['mean_acc'].idxmax()

print('best k = ', bestK)

knnModel = KNeighborsClassifier(n_neighbors=bestK).fit(x_train, y_train)

knnModel
from sklearn.tree import DecisionTreeClassifier

dTreeModel = DecisionTreeClassifier(criterion='entropy', max_depth = 4)

dTreeModel.fit(x_train, y_train)

dTreeModel
from sklearn.ensemble import RandomForestClassifier

rfModel = RandomForestClassifier(n_estimators=11, criterion='entropy', random_state=42)

rfModel.fit(x_train, y_train)
knnPred = knnModel.predict(x_test)

print('KNN F1-score: %.2f' % f1_score(y_test, knnPred, average='weighted'))

print('KNN Accuracy score: %.2f' % accuracy_score(y_test, knnPred))

print('Confusion Matrix:')

print(confusion_matrix(y_test, knnPred))
dTreePred = dTreeModel.predict(x_test)

print('DecisionTree F1-score: %.2f' % f1_score(y_test, dTreePred, average='weighted'))

print('DecisionTree Accuracy score: %.2f' % accuracy_score(y_test, dTreePred))

print('Confusion Matrix:')

print(confusion_matrix(y_test, dTreePred))
rfPred = rfModel.predict(x_test)

print('RandomForest F1-score: %.2f' % f1_score(y_test, rfPred, average='weighted'))

print('RandomForest Accuracy score: %.2f' % accuracy_score(y_test, rfPred))

print('Confusion Matrix:')

print(confusion_matrix(y_test, rfPred))