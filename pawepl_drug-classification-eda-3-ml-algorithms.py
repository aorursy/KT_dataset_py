!pip install pydotplus
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# linear algebra

import numpy as np 

# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 



#Visualization libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



#decision tree visualization

import pydotplus

import matplotlib.image as mpimg

from sklearn import tree



#Data split

from sklearn.model_selection import train_test_split



#ML Algorithms 

from sklearn import tree

import pydotplus

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm







#Model evaluation metrics

from sklearn import metrics



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/drug-classification/drug200.csv')
df.head()
df.shape
df.describe(include = 'all')
df.info()
sns.catplot('Drug', 'Age', data = df)
plt.figure(figsize = (8,6))

ax = sns.boxplot('Sex', 'Age', data = df).set(ylim = (0, 80))
df.Sex.value_counts()
sex_drug = df.groupby('Sex').Drug.value_counts()

sex_drug
sex_drug.unstack(level=0).plot(kind='bar', subplots=False)
df.BP.value_counts()
tab = pd.crosstab(df['BP'], df['Drug'])

print (tab)



tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=False)

plt.xlabel('BP')

plt.ylabel('Percentage')
df.Cholesterol.value_counts()
tab = pd.crosstab(df['Cholesterol'], df['Drug'])



tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=False)

plt.xlabel('Cholesterol')

plt.ylabel('Percentage')
sns.catplot('Drug', 'Na_to_K', data=df)
for col in df:

    print(col)

    print(df[col].unique())

    print()
df["Sex"] = df["Sex"].map({"M": 0, "F":1})

df["BP"] = df["BP"].map({"HIGH" : 3, "NORMAL" : 2, "LOW": 1})

df["Cholesterol"] = df["Cholesterol"].map({"HIGH": 1, "NORMAL" : 0})

df["Drug"] = df["Drug"].map({"DrugY": 0, "drugC": 1, "drugX": 2, "drugA":3, "drugB":4})
df.head()
df.dtypes
sns.boxplot(x=df['Age'])
sns.boxplot(x=df['Na_to_K'])
df.drop(df[df.Na_to_K > 30].index, inplace=True)
sns.boxplot(x=df['Na_to_K'])
plt.figure(figsize=(15,6))

sns.heatmap(df.corr(), vmax=0.6, square=True, annot=True)
df.drop('Sex', axis=1, inplace=True)
df.head()
values=df.values

X, y = values[:, :-1], values[:, -1]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3, random_state = 2)
print("X_train shape:",X_train.shape)

print("X_test shape:",X_test.shape)

print("y_train shape:",y_train.shape)

print("y_test shape:",y_test.shape)
rfc = RandomForestClassifier(n_estimators = 9, criterion = 'entropy', random_state=22)



rfc.fit(X_train,y_train)
rf_pred = rfc.predict(X_test)
print("Accuracy score : ", metrics.accuracy_score(y_test, rf_pred))



print("F1 score: ", metrics.f1_score(y_test, rf_pred, average='weighted') )



print("Jaccard score: ", metrics.jaccard_score(y_test, rf_pred, average='weighted'))



print("recall score: ", metrics.recall_score(y_test, rf_pred, average='weighted'))



print("precision score: ", metrics.precision_score(y_test, rf_pred, average='weighted'))
rfc_score = {

            'accuracy': metrics.accuracy_score(y_test, rf_pred),

            'f1': metrics.f1_score(y_test, rf_pred, average='weighted'),

            'jaccard': metrics.jaccard_score(y_test, rf_pred, average='weighted'),

            'recall': metrics.recall_score(y_test, rf_pred, average='weighted'),

            'precision': metrics.precision_score(y_test, rf_pred, average='weighted')

        }
dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)
dt_pred = dtc.predict(X_test)
print("Accuracy score : ", metrics.accuracy_score(y_test, dt_pred))



print("F1 score: ", metrics.f1_score(y_test, dt_pred, average='weighted') )



print("Jaccard score: ", metrics.jaccard_score(y_test, dt_pred, average='weighted'))



print("recall score: ", metrics.recall_score(y_test, dt_pred, average='weighted'))



print("precision score: ", metrics.precision_score(y_test, dt_pred, average='weighted'))
dt_score = {

            'accuracy': metrics.accuracy_score(y_test, dt_pred),

            'f1': metrics.f1_score(y_test, dt_pred, average='weighted'),

            'jaccard': metrics.jaccard_score(y_test, dt_pred, average='weighted'),

            'recall': metrics.recall_score(y_test, dt_pred, average='weighted'),

            'precision': metrics.precision_score(y_test, dt_pred, average='weighted')

        }
featureNames = ['Age', 'BP', 'Cholesterol','Na_to_K']
dot_data = tree.export_graphviz(dtc,

                                feature_names=featureNames,

                                out_file=None,

                                special_characters=True,

                                filled=True,

                                rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data)



filename = "drugTree.png"

graph.write_png(filename)

img = mpimg.imread(filename)

plt.figure(figsize=(50,100))

plt.imshow(img,interpolation = 'nearest')

plt.show()
Ks = 15

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1,Ks):

    

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    yhat=neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)



    

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])



mean_acc
plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Nabors (K)')

plt.tight_layout()

plt.show()
k = 4

#Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = k)

neigh.fit(X_train,y_train)
predKNN = neigh.predict(X_test)
print("Accuracy score : ", metrics.accuracy_score(y_test, predKNN))



print("F1 score: ", metrics.f1_score(y_test, predKNN, average='weighted') )



print("Jaccard score: ", metrics.jaccard_score(y_test, predKNN, average='weighted'))



print("recall score: ", metrics.recall_score(y_test, predKNN, average='weighted'))



print("precision score: ", metrics.precision_score(y_test, predKNN, average='weighted'))
knn_score = {

            'accuracy': metrics.accuracy_score(y_test, predKNN),

            'f1': metrics.f1_score(y_test, predKNN, average='weighted'),

            'jaccard': metrics.jaccard_score(y_test, predKNN, average='weighted'),

            'recall': metrics.recall_score(y_test, predKNN, average='weighted'),

            'precision': metrics.precision_score(y_test, predKNN, average='weighted')

        }
svc = svm.SVC(kernel='rbf', random_state = 22)

svc.fit(X_train, y_train)
predSVC = svc.predict(X_test)
print("Accuracy score : ", metrics.accuracy_score(y_test, predSVC))



print("F1 score: ", metrics.f1_score(y_test, predSVC, average='weighted') )



print("Jaccard score: ", metrics.jaccard_score(y_test, predSVC, average='weighted'))



print("recall score: ", metrics.recall_score(y_test, predSVC, average='weighted'))



print("precision score: ", metrics.precision_score(y_test, predSVC, average='weighted', zero_division=1))
svm_score = {

            'accuracy': metrics.accuracy_score(y_test, predSVC),

            'f1': metrics.f1_score(y_test, predSVC, average='weighted'),

            'jaccard': metrics.jaccard_score(y_test, predSVC, average='weighted'),

            'recall': metrics.recall_score(y_test, predSVC, average='weighted'),

            'precision': metrics.precision_score(y_test, predSVC, average='weighted', zero_division=1)

        }
print(pd.DataFrame.from_dict(rfc_score, orient = "index",columns=["Score"]))
print(pd.DataFrame.from_dict(dt_score, orient = "index",columns=["Score"]))
print(pd.DataFrame.from_dict(knn_score, orient = "index",columns=["Score"]))
print(pd.DataFrame.from_dict(svm_score, orient = "index",columns=["Score"]))