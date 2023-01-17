# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/HR_comma_sep.csv")

print(df.shape)

print(df.describe())

print(df[pd.isnull(df).any(axis=1)])

print(df.dtypes)
df.pivot_table(['promotion_last_5years','satisfaction_level','average_montly_hours','number_project'], index=['salary','sales'], columns='left')
based_onsalary = pd.crosstab([df.sales, df.left], df.salary)

print(based_onsalary)
based_onsalary.plot.barh(stacked=True)

plt.show()
based_onpromotion = df.groupby('sales')[['promotion_last_5years']].count()

print(based_onpromotion)
based_onpromotion.plot.pie(y='promotion_last_5years', autopct='%.2f')

plt.show()

# sales is the industry sector with maximum promotions and management is the one with minimum promotions
based_onsatisfaction = df.groupby(['sales','left'] ,as_index=False)[['satisfaction_level']].mean()

print(based_onsatisfaction)
sns.barplot(x='satisfaction_level', data=based_onsatisfaction, y='sales', hue='left')

plt.show()

# employee satisfaction is very important
based_on_montlyhours = df.groupby(['sales','left'], as_index=False)[['average_montly_hours']].mean()

print(based_on_montlyhours)
sns.barplot(x='average_montly_hours', y='sales', data=based_on_montlyhours, hue='left')

plt.show()
y = np.ravel(df.loc[:,['left']])

# converting the object values of salary and sales

ind = list(enumerate(np.unique(df['sales'])))

ind_dict = {name:i for i, name in ind}

df.sales = df.sales.map(lambda x: ind_dict[x]).astype(int)

# same for salary

sal = list(enumerate(np.unique(df['salary'])))

sal_dict = {name:i for i, name in sal}

df.salary = df.salary.map(lambda x: sal_dict[x]).astype(int)

df.drop(['left'], axis=1 , inplace=True)



print(df.dtypes)
from sklearn.preprocessing import StandardScaler

X = df.values

X_scaled = StandardScaler().fit_transform(X)

print(X_scaled.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit_transform(X_scaled)

print(pca.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7, test_size=0.3)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

data_train, data_test, Y_train, Y_test = train_test_split(pca, y, random_state=7, test_size=0.3)

print(data_train.shape, data_test.shape, Y_train.shape, Y_test.shape)
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

svc = SVC(C=1, kernel='rbf', gamma=0.15).fit(data_train,Y_train)

Y_predict = svc.predict(data_test)

print(accuracy_score(Y_test, Y_predict ))

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1).fit(X_train, y_train)

y_predict = clf.predict(X_test)

print(clf.feature_importances_)

print(accuracy_score(y_test, y_predict))

important = clf.feature_importances_

indices = np.argsort(important)[::-1]

labels = df.columns

for i in range(X_train.shape[1]):

    print(i+1,important[indices[i]], labels[i])    

plt.title('feature importances')

plt.bar(range(X_train.shape[1]), important[indices], color='blue')

plt.xticks(range(X_train.shape[1]), labels, rotation=90)

plt.xlim([-1, X_train.shape[1]])

plt.tight_layout()

plt.show()
from matplotlib.colors import ListedColormap

def plotboundary(model, X, y):

    resolution = 0.02

    markers = ('s', 'x', 'o', '^', 'v')

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))

    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)

        

plotboundary(svc, data_test, Y_test)