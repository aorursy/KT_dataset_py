# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: htatps://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')



data_train.sample(3)#randomly taking 3 values
data_test.head(3)
data_train.describe()
data_train.shape,data_test.shape
data_train.isnull().sum().sort_values(ascending=False)
#we have to find the null cols

null_cols = data_train.columns[data_train.isnull().any()]

null_cols

a = data_train.isnull().sum()

a[a>0]
b = data_test.isnull().sum()

b[b>0]
import seaborn as sns

g = sns.FacetGrid(data_train,col = 'Sex',row = 'Survived')

g.map(plt.hist,'Age')
sns.boxplot(x= 'Sex',y = 'Age',hue  = 'Survived',data = data_train)
sns.boxplot(x = 'Pclass',y = 'Age',hue = 'Survived',data = data_train)
g = sns.FacetGrid(data_train,'Survived',col = 'Pclass',margin_titles=True,palette={1:"green",0:"red"})

g = g.map(plt.scatter,'Fare','Age').add_legend();
data_train.Embarked.value_counts().plot(kind = 'bar')

plt.title('passengers/boarding location')

plt.xticks(rotation=0)
sns.barplot('Embarked','Pclass',hue = 'Survived',data = data_train)
sns.barplot(y = 'Pclass',x = 'Sex',data = data_train)
sns.boxplot('Pclass','Age',data = data_train,hue = 'Sex')
data_train.Age[data_train.Pclass == 1].plot(kind='kde')    

data_train.Age[data_train.Pclass == 2].plot(kind='kde')

data_train.Age[data_train.Pclass == 3].plot(kind='kde')

 # plots an axis lable

plt.xlabel("Age")    

plt.title("Age Distribution within classes")

# sets our legend for our graph.

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') ;
plt.figure(figsize=(10,10))

sns.heatmap(data_train.corr(),square=True,annot=True,linewidths=.01,linecolor='white',vmax=.8)
data_train.corr()['Survived']
surv_col = 'green'

nosurv_col = 'red'

cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']

g = sns.pairplot(data = data_train.dropna(),hue = 'Survived', vars = cols,palette= [nosurv_col,surv_col],size=3)

g.set(xticklabels=[])

plt.figure(figsize=(8,8))

sns.violinplot('Embarked','Age',hue = 'Survived',data = data_train,split = True,dodge=True)
def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df    

    

def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = drop_features(df)

    return df



data_train = transform_features(data_train)

data_test = transform_features(data_test)

data_train.head()
from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test

    

data_train, data_test = encode_features(data_train, data_test)

data_train.head()
from sklearn.model_selection import train_test_split



X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)

y_all = data_train['Survived']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)
X_train.shape, X_test.shape, y_train.shape, y_test.shape 
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [10, 15, 20], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)



predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))
from sklearn.cross_validation import KFold

import warnings

warnings.filterwarnings('ignore')



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



run_kfold(clf)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# Support Vector Machines

from sklearn.svm import SVC

clf = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)

acc_svc


knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian
# Perceptron

import warnings

warnings.filterwarnings("ignore")

perceptron = Perceptron()

perceptron.fit(X_train, y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)

acc_sgd
ids = data_test['PassengerId']

predictions = clf.predict(data_test.drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('titanic-predictions.csv', index = False)

output.head()