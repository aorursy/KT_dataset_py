import numpy as np

import pandas as pd



import seaborn as sns



from sklearn.pipeline import Pipeline



import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.shape
train.head(5)
train['Survived'].value_counts()    # train.groupby('Survived').size()
train.info()



train.Cabin.dtypes

train.Cabin.value_counts()    #train['Cabin'].value_counts()
train.describe(include = 'all').T
train.isnull().sum(axis = 0)



train = train.dropna(axis = 1, thresh = 500)



mean_age = train.Age.mean()

train.Age.fillna(mean_age, inplace = True)



most_freq = train.Embarked.value_counts(dropna = True).idxmax()

train.Embarked.fillna(most_freq, inplace = True)



print(train.isnull().sum())
train.Survived = train.Survived.astype('category')

train.Pclass = train.Pclass.astype('category')

train.Embarked = train.Embarked.astype('category')

train.info()
count, bin_dividers = np.histogram(train.Age, bins = 8)

bin_dividers



bin_names = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80']

train['Age_bins'] = pd.cut(x = train['Age'],

                          bins = bin_dividers,

                          labels = bin_names,

                          include_lowest = False)



train[['Age', 'Age_bins']].head(10)
train.Fare = train.Fare/abs(train.Fare.max())
train.info()
#train['Survived'].value_counts().plot(kind = 'bar')



#train['Survived'].value_counts().index.astype(object)

#plt.bar(train['Survived'].value_counts().index.astype(object), train['Survived'].value_counts().values)



plt.figure(figsize = (8,5))

plt.style.use('seaborn')

print(plt.style.available)

sns.countplot(x = 'Survived', data = train)

plt.title('Count of Survived', fontsize = 20)
fig = plt.figure(figsize = (15, 30))

ax1 = fig.add_subplot(4, 2, 1)

ax2 = fig.add_subplot(4, 2, 2)

ax3 = fig.add_subplot(4, 2, 3)

ax4 = fig.add_subplot(4, 2, 4)

ax5 = fig.add_subplot(4, 2, 5)

ax6 = fig.add_subplot(4, 2, 6)



ax1 = sns.countplot(x = 'Survived', hue = 'Pclass', data = train, ax = ax1)

ax1.set_title('Pclass')



ax2 = sns.countplot(x = 'Survived', hue = 'Sex', data = train, ax = ax2)

ax2.set_title('Sex')



ax3 = sns.distplot(train[train['Survived'] == 0]['Age'].dropna(), color = 'red', label = '0', ax = ax3)

ax3 = sns.distplot(train[train.Survived == 1]['Age'].dropna(), color = 'blue', label = '1', ax = ax3)

ax3.set_title('Age')

ax3.legend(title = 'Age', loc = 'best')



ax4 = sns.countplot(x = 'Survived', hue = 'SibSp', data = train, ax = ax4)

ax4.set_title('Sibling')

ax4.legend(title = 'Sibling', loc = 'upper right')



ax5 = sns.countplot(x = 'Survived', hue = 'Parch', data = train, ax = ax5)

ax5.set_title('Parch')

ax5.legend(title = 'Parch', loc = 'upper right')



ax6 = sns.distplot(train[train.Survived == 0]['Fare'], color = 'red', label = '0', ax = ax6)

ax6 = sns.distplot(train[train.Survived == 1]['Fare'], color = 'blue', label = '1', ax = ax6)

ax6.set_title('Fare')

ax6.legend(title = 'Fare', loc = 'upper right')
sns.regplot(x = 'Age', y = 'Fare', data = train)
fig = plt.figure(figsize = (10, 10))

ax1 = fig.add_subplot(2, 1, 1)

ax2 = fig.add_subplot(2, 1, 2)



table = train.pivot_table(index = 'Sex', columns = 'Pclass', aggfunc = 'size')

sns.heatmap(table, annot = True, fmt = 'd', cmap = 'YlGnBu', linewidth = 5, ax = ax1)



table2 = table = train.pivot_table(index = 'Embarked', columns = 'Pclass', aggfunc = 'size')

sns.heatmap(table2, annot = True, fmt = 'd', linewidth = 5, ax = ax2)
sns.jointplot(x = 'Fare', y = 'Age', kind = 'reg', data = train)

sns.jointplot(x = 'Fare', y = 'SibSp', kind = 'kde', data = train)

sns.jointplot(x = 'Fare', y = 'Pclass', data = train)
g = sns.FacetGrid(data = train, col = 'Sex', row = 'Survived')

g.map(plt.hist, 'Age')
train_pair = train[['Age', 'Pclass', 'Fare']]

sns.pairplot(train_pair)
train.head()

X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

y = train['Survived']



onehot_Pclass = pd.get_dummies(train.Pclass)

onehot_Sex = pd.get_dummies(train.Sex)

onehot_Embarked = pd.get_dummies(train.Embarked)



X = pd.concat([X, onehot_Pclass, onehot_Sex, onehot_Embarked], axis = 1)

X.drop(['Pclass', 'Sex', 'Embarked'], axis = 1, inplace = True)

X.head()
#from sklearn import preprocessing

#X = preprocessing.StandardScaler().fit(X).transform(X)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)

knn.fit(X_train, y_train)

y_hat = knn.predict(X_test)



from sklearn import metrics

knn_matrix = metrics.confusion_matrix(y_test, y_hat)

print(knn_matrix)



knn_report = metrics.classification_report(y_test, y_hat)

print(knn_report)