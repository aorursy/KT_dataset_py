import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np





plt.style.use('ggplot')

sns.__version__
df = pd.read_csv('../input/heart.csv')
df.head(5)
print('total number of features: ', len(df.columns)-1)

print("Feature names: ", list(df.columns.drop('target')))
columns = df.columns

unique = df.nunique()

plt.figure(figsize=(8, 8))

plt.barh(columns, unique)

plt.xlabel('Unique values')

plt.ylabel('Feature names')

plt.show()
print('Unique values in Target: ',  df['target'].nunique())
df.describe()
df.isnull().values.any()
df.info()
pd.DataFrame(df.corr(method='spearman')['target'])
vars = np.array(['cp', 'oldpeak', 'thalach', 'slope'])

plt.figure(figsize=(10, 10))

for i in vars:

    plt.subplot(2,2,np.where(i == vars)[0][0]+1)

    sns.stripplot(x="target", y=i, data=df)

    

plt.suptitle('Figure - Strip plot', x=0.5, y=0.9, verticalalignment='center', fontsize= 18)

plt.show()
vars = np.array(['cp', 'oldpeak', 'thalach', 'slope'])

plt.figure(figsize=(10, 10))

for i in vars:

    plt.subplot(2,2,np.where(i == vars)[0][0]+1)

    sns.kdeplot(df['target'], df[i], shade=True, cut=4)

    

plt.suptitle('Figure - Kernel density estimation plot', x=0.5, y=0.9, verticalalignment='center', fontsize= 18)

plt.show()
sns.pairplot(df, vars=['cp', 'restecg', 'thalach', 'slope'], height=4, hue='target', 

             diag_kind='kde', markers=["D", "s"], diag_kws=dict(shade=True))

plt.suptitle('Figure - Scatter plot of features ', x=0.5, y=1.01, verticalalignment='center', fontsize= 20)

plt.show()
vars = np.array(['age', 'oldpeak', 'thalach', 'chol', 'trestbps'])

plt.figure(1 , figsize=(20, 10))

for i in vars:

    plt.subplot(2,3,np.where(i == vars)[0][0]+1)

    sns.distplot(a = df[i], rug=True, color = 'blue')



plt.suptitle('Figure - Histograms', x=0.5, y=0.9, verticalalignment='center', fontsize= 18)

plt.show()
plt.figure(figsize=(24,12))

plt.subplot(1,2,1)

sns.countplot(x="sex", hue='target', data=df)

plt.subplot(1,2,2)

sns.countplot(x="age", hue='target', data=df)

plt.suptitle('Figure - Count plot of sex and age with target grouping variable', 

             x=0.5, y=0.9, verticalalignment='center', fontsize= 18)
plt.figure(figsize=(14,7))

plt.subplot(1,2,1)

sns.kdeplot(df['sex'], df['target'], shade=True, cut=3)

plt.subplot(1,2,2)

sns.kdeplot(df['age'], df['target'], shade=True, cut=3)

plt.suptitle('Figure - KDE plot of sex and age wrt target', x=0.5, y=1, verticalalignment='center', fontsize= 18)

plt.show()
vars = np.array(['cp', 'oldpeak', 'thalach', 'slope'])

plt.figure(figsize=(15, 15))

for i in vars:

    plt.subplot(2,2,np.where(i == vars)[0][0]+1)

    sns.distplot(df[i][df['sex'] == 1], color='blue', label='male')

    sns.distplot(df[i][df['sex'] == 0], label='female')

    plt.legend()

    

plt.suptitle('Figure - Histogram of features wrt sex', x=0.5, y=0.9, verticalalignment='center', fontsize= 18)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

import xgboost as xgb
y = df['target']

x = df.drop(columns=['target'])
scaler = StandardScaler().fit(x)

rescaledX = scaler.transform(x)
X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, random_state=5)



print("X train: ", X_train.shape)

print("X test: ", X_test.shape)

print("y train: ", y_train.shape)

print("y test: ", y_test.shape)
lr = svm.SVC(kernel='linear')

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

lr.score(X_test, y_test)
confusion_matrix(y_test, y_pred)
precision_recall_fscore_support(y_test, y_pred, average='binary')
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=5)



print("X train: ", X_train.shape)

print("X test: ", X_test.shape)

print("y train: ", y_train.shape)

print("y test: ", y_test.shape)
lr = svm.SVC(kernel='linear')

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

lr.score(X_test, y_test)
confusion_matrix(y_test, y_pred)
precision_recall_fscore_support(y_test, y_pred, average='binary')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)



print("X train: ", X_train.shape)

print("X test: ", X_test.shape)

print("y train: ", y_train.shape)

print("y test: ", y_test.shape)
lr = LogisticRegression(C=0.1, solver='liblinear')

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

lr.score(X_test, y_test)
confusion_matrix(y_test, y_pred)
precision_recall_fscore_support(y_test, y_pred, average='binary')
scaler = StandardScaler().fit(x)

rescaledX = scaler.transform(x)
X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, test_size=0.15, random_state=5)



print("X train: ", X_train.shape)

print("X test: ", X_test.shape)

print("y train: ", y_train.shape)

print("y test: ", y_test.shape)
sgd = SGDClassifier(max_iter=50, random_state=5)

sgd.fit(X_train, y_train)
y_pred = lr.predict(X_test)

sgd.score(X_test, y_test)
confusion_matrix(y_test, y_pred)
precision_recall_fscore_support(y_test, y_pred, average='binary')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5)



print("X train: ", X_train.shape)

print("X test: ", X_test.shape)

print("y train: ", y_train.shape)

print("y test: ", y_test.shape)
accuracy = []



max_dep = range(1,10)



for i in max_dep:

    xg = xgb.XGBClassifier(max_depth=i, min_samples_leaf=2)

    xg.fit(X_train, y_train)

    accuracy.append(xg.score(X_test, y_test))

    

print('List of accuracy: ', accuracy)    
plt.plot(max_dep, accuracy, label='Accuracy of validation set')

plt.ylabel('Accuracy')

plt.xlabel('Max Depth')

plt.legend()

plt.show()
xg =  xgb.XGBClassifier(max_depth=3, min_samples_leaf=2)

xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

xg.score(X_test, y_test)
confusion_matrix(y_test, y_pred)
precision_recall_fscore_support(y_test, y_pred, average='binary')