import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree

from sklearn.metrics import accuracy_score



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import datasets

df_train = pd.read_csv("../input/train.csv") 

df_test = pd.read_csv("../input/test.csv") 



# view first five lines of training data

df_train.head()
df_test.head()
df_train.info()
df_train.describe()
# plot of count(Survived)

sns.countplot(x="Survived", data=df_train)

plt.show()
no_survived = pd.Series([0] * df_test.shape[0])
out = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': no_survived})
out.to_csv('no_survival.csv', index=False)
# plot count of male and female on titanic

sns.countplot(x="Sex", data=df_train);
sns.countplot(x="Survived", hue='Sex', data=df_train);
sns.catplot(x="Survived", col="Sex", kind="count", data=df_train);
df_train.groupby(['Sex']).Survived.sum()
df_train.groupby(["Sex"]).Survived.value_counts()
print(df_train[df_train["Sex"]=="female"].Survived.sum() / df_train[df_train["Sex"]=="female"].shape[0]) 

# print(df_train[df_train["Sex"]=="female"].Survived.sum() / df_train[df_train["Sex"]=="female"].count()) 

print(df_train[df_train["Sex"]=="male"].Survived.sum() / df_train[df_train["Sex"]=="male"].shape[0]) 
women_survived_series = pd.Series(list(map(int, df_test["Sex"]=="female")))
out = pd.DataFrame({"PassengerId": df_test.PassengerId, "Survived": women_survived_series})

out.to_csv('all_women_survived.csv', index=False)
sns.catplot(x="Survived", col="Pclass", kind="count", data=df_train);
sns.catplot(x="Survived", col="Pclass", kind="count", hue="Sex", data=df_train);
print(df_train.groupby("Pclass").Survived.sum() / df_train.groupby("Pclass").Survived.count())
sns.catplot(x="Survived", col="Embarked", kind="count", data=df_train);
#sns.catplot(x="Embarked", col="Survived", kind="count", data=df_train);
print(df_train.groupby("Embarked").Survived.sum() / df_train.groupby("Embarked").Survived.count())
sns.catplot(x="Survived", col="Embarked", hue="Pclass", kind="count", data=df_train);
# df_train.groupby("Embarked").Survived.sum()  # shows number of people survived from each embarked point

# df_train.groupby("Embarked").Survived.count() # shows number of poeple embarked form each port
plt.figure(figsize=(18, 8))

sns.distplot(a=df_train.Fare, kde=False);
# Use a pandas plotting method to plot the column 'Fare' for each value of 'Survived' on the same plot.

df_train.groupby('Survived').Fare.hist(alpha=0.5);
df_train_drop = df_train.Age.dropna()

plt.figure(figsize=(18, 8))

sns.distplot(a=df_train_drop, kde=False);
sns.stripplot(x="Survived", y="Fare", data=df_train);
sns.swarmplot(x="Survived", y="Fare", data=df_train);
df_train.Fare.describe()
# Use the DataFrame method .describe() to check out summary statistics of 'Fare' as a function of survival.

df_train.groupby('Survived').Fare.describe()
sns.scatterplot(x="Age", y="Fare", hue="Survived", data=df_train, alpha=0.5);
sns.pairplot(data=df_train, hue="Survived");
# Import modules

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import numpy as np

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



# Figures inline and set visualization style

%matplotlib inline

sns.set()



# Import data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# target variable

survived_train = df_train.Survived

# concatenate train and test set (to perform same data manipulation on both datasets)

data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])
data.info()
data['Age'] = data.Age.fillna(data.Age.median())

data['Fare'] = data.Fare.fillna(data.Fare.median())

data.info()
data = pd.get_dummies(data, columns=["Sex"], drop_first=True)

data.head()
data = data[["Pclass", "Age", "SibSp", "Fare", "Sex_male"]]

data.head()
data_train = data.iloc[:891]

data_test = data.iloc[891:]
X = data_train.values

test = data_test.values

y = survived_train.values
clf = tree.DecisionTreeClassifier(max_depth=3)

clf.fit(X, y)
Y_pred = clf.predict(test)

out = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': Y_pred})
out.to_csv('DecisionTree3.csv', index=False)
# plt.figure(figsize=(10, 10))

# tree.plot_tree(clf.fit(X, y));
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42, stratify=y)
# Setup arrays to store train and test accuracies

dep = np.arange(1, 9)

train_accuracy = np.empty(len(dep))

test_accuracy = np.empty(len(dep))



# Loop over different values of k

for i, k in enumerate(dep):

    # Setup a k-NN Classifier with k neighbors: knn

    clf = tree.DecisionTreeClassifier(max_depth=k)



    # Fit the classifier to the training data

    clf.fit(X_train, y_train)

    

    #Compute accuracy on the training set

    train_accuracy[i] = clf.score(X_train, y_train)



    #Compute accuracy on the testing set

    test_accuracy[i] = clf.score(X_test, y_test)



# Generate plot

plt.title('clf: Varying depth of tree')

plt.plot(dep, test_accuracy, label = 'Testing Accuracy')

plt.plot(dep, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Depth of tree')

plt.ylabel('Accuracy')

plt.show()
# Imports

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import numpy as np

from sklearn import tree

from sklearn.model_selection import GridSearchCV



# Figures inline and set visualization style

%matplotlib inline

sns.set()



# Import data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



# Store target variable of training data in a safe place

survived_train = df_train.Survived



# Concatenate training and test sets

data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])



# View head

data.head()
data.Name.head()
data.Name.tail()
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

sns.countplot(data.Title)

plt.xticks(rotation=90);
data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})

data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',

                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')

sns.countplot(x='Title', data=data);

plt.xticks(rotation=90);
data[data.Cabin.isnull()].Fare.hist()
data['hasCabin'] = ~data.Cabin.isnull()

data.head()
# drop columns ['PassengerId', 'Name', 'Ticket', 'Cabin']

data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

data.head()
data.info()
data['Age'] = data.Age.fillna(data.Age.median())

data['Fare'] = data.Fare.fillna(data.Fare.median())

data['Embarked'] = data.Embarked.fillna('S')    # as most of passsengers embarked from Southampton

data.info()
data['CatAge'] = pd.qcut(data.Age, q=4, labels=False)

data['CatFare'] = pd.qcut(data.Fare, q=4, labels=False)

data.info()
# Now we can drop 'Age' and 'Fare' column

data.drop(['Age', 'Fare'], axis=1, inplace=True)

data.head()
data['FamSize'] = data.SibSp + data.Parch

data.head()
# drop 'SibSp' and 'Parch'

data.drop(['SibSp', 'Parch'], axis=1, inplace=True)

data.head()
data_dum = pd.get_dummies(data, drop_first=True)

data_dum.head()
data_train = data_dum[:891]

data_test = data_dum[891:]



X = data_train.values

y = survived_train.values

test = data_test.values
# setup the hyperparameter grid

dep = np.arange(1, 9)

param_grid = {'max_depth': dep}



clf = tree.DecisionTreeClassifier()

clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)

clf_cv.fit(X, y)



print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))

print("Best score is {}".format(clf_cv.best_score_))
y_pred = clf_cv.predict(test)
out = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_pred})

out.to_csv('feature_engg4.csv', index=False)