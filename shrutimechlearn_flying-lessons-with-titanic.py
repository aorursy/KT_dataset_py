import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt
train_data = pd.read_csv('../input/train.csv')
train_data.info()
train_data.head()
train_data.Sex.value_counts()
train_data['is_male'] = np.where(train_data.Sex == 'male', 1,0)
train_data.Embarked.value_counts()
train_data['Embarked_S'] = np.where(train_data.Embarked == 'S', 1,0)

train_data['Embarked_C'] = np.where(train_data.Embarked == 'C', 1,0)
train_data.isnull().sum()
train_data.isnull().mean()
train_data.isnull().mean().plot.bar(figsize=(12,6))

plt.ylabel('Percentage of missing values')

plt.xlabel('Variables')

plt.title('Quantifying missing data')
train_data.nunique()
train_data.nunique().plot.bar(figsize=(12,6))

plt.ylabel('Number of unique categories')

plt.xlabel('Variables')

plt.title('Cardinality')
label_freq = train_data['Pclass'].value_counts() / len(train_data)

fig = label_freq.sort_values(ascending=False).plot.bar()

fig.axhline(y=0.30, color='red')

fig.set_ylabel('percentage within each category')

fig.set_xlabel('Variable: Pclass')

fig.set_title('Identifying Rare Categories')

plt.show()
label_freq = train_data['Sex'].value_counts() / len(train_data)

fig = label_freq.sort_values(ascending=False).plot.bar()

fig.axhline(y=0.35, color='red')

fig.set_ylabel('percentage within each category')

fig.set_xlabel('Variable: Sex')

fig.set_title('Identifying Rare Categories')

plt.show()
label_freq = train_data['SibSp'].value_counts() / len(train_data)

fig = label_freq.sort_values(ascending=False).plot.bar()

fig.axhline(y=0.10, color='red')

fig.set_ylabel('percentage within each category')

fig.set_xlabel('Variable: SibSp')

fig.set_title('Identifying Rare Categories')

plt.show()
label_freq = train_data['Embarked'].value_counts() / len(train_data)

fig = label_freq.sort_values(ascending=False).plot.bar()

fig.axhline(y=0.20, color='red')

fig.set_ylabel('percentage within each category')

fig.set_xlabel('Variable: Embarked')

fig.set_title('Identifying Rare Categories')

plt.show()
label_freq = train_data['Parch'].value_counts() / len(train_data)

fig = label_freq.sort_values(ascending=False).plot.bar()

fig.axhline(y=0.15, color='red')

fig.set_ylabel('percentage within each category')

fig.set_xlabel('Variable: Parch')

fig.set_title('Identifying Rare Categories')

plt.show()
train_data['Fare'].hist(bins=50)



plt.title('Histogram Column Fare')
import scipy.stats as stats

stats.probplot(train_data['Fare'], dist="norm", plot=plt)

plt.title('Q-Q Plot for column Fare')

plt.show()
train_data['Age'].hist(bins=50)

plt.title('Histogram Column Age')
stats.probplot(train_data['Age'], dist="norm", plot=plt)

plt.title('Q-Q Plot for column Age')

plt.show()
_ = train_data.hist(bins=30, figsize=(12,12), density=True)
import seaborn as sns

sns.lmplot(x="Age", y="Survived", data=train_data, order=1)

plt.ylabel('Target')

plt.xlabel('Independent variable')
import seaborn as sns

sns.lmplot(x="Fare", y="Survived", data=train_data, order=1)

plt.ylabel('Target')

plt.xlabel('Independent variable')
sns.boxplot(y=train_data['Fare'])

plt.title('Boxplot')
sns.boxplot(y=train_data['Age'])

plt.title('Boxplot')
def find_boundaries(df, variable, distance):

    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)

    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary
upper_boundary, lower_boundary = find_boundaries(train_data, 'Age', 1.5)

upper_boundary, lower_boundary
train_data.describe()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import BayesianRidge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import BayesianRidge

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
import sklearn

sklearn.__version__
X_train, X_test, y_train, y_test = train_test_split(

train_data.drop(['Name','Sex', 'Embarked', 'Ticket', 'PassengerId', 'Cabin'], axis=1), train_data['Survived'], test_size=0.3,

random_state=0)
imputer_bayes = IterativeImputer(

estimator=BayesianRidge(),

max_iter=10,

random_state=0)



imputer_knn = IterativeImputer(

estimator=KNeighborsRegressor(n_neighbors=5),

max_iter=10,

random_state=0)



imputer_nonLin = IterativeImputer(

estimator=DecisionTreeRegressor(

max_features='sqrt', random_state=0),

max_iter=10,

random_state=0)



imputer_missForest = IterativeImputer(

estimator=ExtraTreesRegressor(

n_estimators=10, random_state=0),

max_iter=10,

random_state=0)
imputer_bayes.fit(X_train)

imputer_knn.fit(X_train)

imputer_nonLin.fit(X_train)

imputer_missForest.fit(X_train)
X_train_bayes = imputer_bayes.transform(X_train)

X_train_knn = imputer_knn.transform(X_train)

X_train_nonLin = imputer_nonLin.transform(X_train)

X_train_missForest = imputer_missForest.transform(X_train)
variables = train_data.columns

predictors = [var for var in variables if var not in ['Name','Sex', 'Embarked', 'Ticket', 'PassengerId', 'Cabin']]

X_train_bayes = pd.DataFrame(X_train_bayes, columns = predictors)

X_train_knn = pd.DataFrame(X_train_knn, columns = predictors)

X_train_nonLin = pd.DataFrame(X_train_nonLin, columns = predictors)

X_train_missForest = pd.DataFrame(X_train_missForest, columns = predictors)
vis_column = 'Fare'

fig = plt.figure()

ax = fig.add_subplot(111)

X_train[vis_column].plot(kind='kde', ax=ax, color='blue')

X_train_bayes[vis_column].plot(kind='kde', ax=ax, color='green')

X_train_knn[vis_column].plot(kind='kde', ax=ax, color='red')

X_train_nonLin[vis_column].plot(kind='kde', ax=ax, color='black')

X_train_missForest[vis_column].plot(kind='kde', ax=ax, color='orange')

# add legends

lines, labels = ax.get_legend_handles_labels()

labels = [vis_column+' original', vis_column+' bayes', vis_column+' knn', vis_column+' Trees', vis_column+' missForest']

ax.legend(lines, labels, loc='best')

plt.show()
vis_column = 'Age'

fig = plt.figure()

ax = fig.add_subplot(111)

X_train[vis_column].plot(kind='kde', ax=ax, color='blue')

X_train_bayes[vis_column].plot(kind='kde', ax=ax, color='green')

X_train_knn[vis_column].plot(kind='kde', ax=ax, color='red')

X_train_nonLin[vis_column].plot(kind='kde', ax=ax, color='black')

X_train_missForest[vis_column].plot(kind='kde', ax=ax, color='orange')

# add legends

lines, labels = ax.get_legend_handles_labels()

labels = [vis_column+' original', vis_column+' bayes', vis_column+' knn', vis_column+' Trees', vis_column+' missForest']

ax.legend(lines, labels, loc='best')

plt.show()