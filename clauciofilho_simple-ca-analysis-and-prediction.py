import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv').drop("Unnamed: 32", axis=1)

df.head()
df.info()
df['diagnosis'].unique()
df['diagnosis'].value_counts()
sns.set(style="darkgrid", palette='Set2')

sns.countplot(df['diagnosis']) 
df_mean = df.iloc[: , 1:12]

df_mean.head()
plt.figure(figsize=(8, 6))

corr = df_mean.corr()

sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
df_mean['diagnosis']=df_mean['diagnosis'].map({'M':0, 'B':1})

df_mean.head()
df_mean['diagnosis'].value_counts()
from sklearn.model_selection import train_test_split

X = df_mean.iloc[: , 1:] #features

y = df_mean['diagnosis'] #target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
clf = KNeighborsClassifier()

clf.fit(X_train, y_train)



clf.score(X_train, y_train)
prediction = clf.predict(X_test)

prediction
print('KNN Acurácia:', metrics.accuracy_score(prediction, y_test))

 

print( metrics.classification_report(y_test, prediction))

cm = metrics.confusion_matrix(y_test, prediction)

sns.heatmap(cm ,annot=True,fmt="d")
from sklearn.svm import SVC



clf = SVC()

clf.fit(X_train, y_train)

clf.score(X_train, y_train)
prediction = clf.predict(X_test)

prediction
print('SVM Acurácia:', metrics.accuracy_score(prediction, y_test))



print( metrics.classification_report(y_test, prediction))

cm = metrics.confusion_matrix(y_test, prediction)

sns.heatmap(cm ,annot=True,fmt="d")
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train, y_train)

clf.score(X_train, y_train)
prediction = clf.predict(X_test)

prediction
print('Random Forest Acurácia:', metrics.accuracy_score(prediction, y_test))



print( metrics.classification_report(y_test, prediction))

cm = metrics.confusion_matrix(y_test, prediction)

sns.heatmap(cm ,annot=True,fmt="d")