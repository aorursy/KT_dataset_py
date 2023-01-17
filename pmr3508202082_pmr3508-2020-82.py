import pandas as pd
test = pd.read_csv("../input/adult-pmr3508/test_data.csv")

train = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values="?")

train.columns.values
train.shape
train.head()
e_train = train.dropna()

e_train.shape
import matplotlib as plt

import seaborn as sns
train['age'].hist()

plt.pyplot.xlabel("Age")

plt.pyplot.ylabel("Number of people")
train['workclass'].value_counts()
train['education'].value_counts()
train['marital.status'].value_counts().sort_values().plot(kind = 'barh')

plt.pyplot.xlabel("Number of people")
train['occupation'].value_counts()
train['relationship'].value_counts()
train['race'].value_counts()
train['sex'].value_counts()
train['capital.gain'].value_counts()
train['capital.loss'].value_counts()
print(train['hours.per.week'].value_counts())
train['native.country'].value_counts()
from sklearn import preprocessing
print(e_train['income'])

e_train = e_train.apply(preprocessing.LabelEncoder().fit_transform)

e_train.corr()[['income']]
target = e_train.loc[:, 'income']

print(target)

features = e_train.loc[:, ['age', 'education.num', 'marital.status', 'relationship', 'occupation', 'sex', 'capital.gain', 

                           'capital.loss']]
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
results = []

for k in range(3,19,5):

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, features, target, cv=10)

    results.append([k,scores.mean()])

print(results)
# e_test = test.apply(preprocessing.LabelEncoder().fit_transform)

# e_test = e_test.loc[:, ['age', 'education.num', 'marital.status', 'relationship', 'occupation', 'sex', 'capital.gain', 

#                            'capital.loss']]

# knn.fit(features, target)

# y_pred = knn.predict(e_test)

# submi = []

# for i in range(0,len(y_pred)):

#     if y_pred[i] == 0:

#         submi.append("<=50K")

#     else:

#         submi.append(">50K")



# submi = pd.DataFrame(submi)

# submi.columns = ['income']

# submi.index.name = 'Id'

# submi.to_csv(' PMR3508-2020-82')