from IPython.display import HTML

HTML('''<script>

  function code_toggle() {

    if (code_shown){

      $('div.input').hide('500');

      $('#toggleButton').val('Show Code')

    } else {

      $('div.input').show('500');

      $('#toggleButton').val('Hide Code')

    }

    code_shown = !code_shown

  }

  $( document ).ready(function(){

    code_shown=false;

    $('div.input').hide()

  });

</script>

<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
# connecting packeges

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# decision tree packeges

from sklearn.tree import DecisionTreeClassifier 

from sklearn.tree import export_graphviz 

# boosting

import xgboost

from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score

# PCA

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
# Hide warnings

import warnings

warnings.filterwarnings('ignore')
# load data

# surviving passengers by id

gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

# train data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

# test data

test = pd.read_csv('/kaggle/input/titanic/test.csv')
print('gender_submission')

gender_submission.head()

print('Train data')

train.head()

print('Test data')

test.head()
# remove the passenger's name

del train['Name']

del test['Name']

del train['Cabin']

del test['Cabin']

del train['Ticket']

del test['Ticket']
# Let's look at the number of NaN values in the columns

print('Train data')

train.isnull().sum()

print('Test data')

test.isnull().sum()
# replace empty values with averages

train = train.fillna({'Age':train.Age.median()})

test = test.fillna({'Age':test.Age.median()})

# drop all anither NaN values

train.dropna(inplace=True)

test.dropna(inplace=True)
train.Embarked = train.Embarked.replace({'S':'Southampton', 'C':'Cherbourg',  'Q':'Queenstown'})

test.Embarked = test.Embarked.replace({'S':'Southampton', 'C':'Cherbourg',  'Q':'Queenstown'})
train = pd.get_dummies(train)

test = pd.get_dummies(test)
# Look at ready data

print('gender_submission')

gender_submission.head()

print('Train data')

train.head()

print('Test data')

test.head()
X = train.drop(['PassengerId','Survived'], axis=1)

X.head()
Y = train.Survived
clf = DecisionTreeClassifier()

from sklearn.model_selection import GridSearchCV

parametrs = {'criterion': ['gini', 'entropy'], 'max_depth':range(1, 100)}

grid_search_cv_clf = GridSearchCV(DecisionTreeClassifier(), parametrs, iid=True, cv=5)

best_params = grid_search_cv_clf.fit(X, Y)

best_criterion = best_params.best_params_['criterion']

best_depth = best_params.best_params_['max_depth']

print('Best criterion:', best_criterion)

print('Best depth of the tree:', best_depth)
clf = DecisionTreeClassifier(criterion=best_criterion, max_depth=best_depth)

imp_tree = pd.DataFrame(clf.fit(X, Y).feature_importances_, 

              index=X.columns, columns=['importance'])

ax = imp_tree.sort_values('importance').plot(kind='barh', figsize=(5, 5))
best_clf = grid_search_cv_clf.fit(X, Y).best_estimator_

X_test = test.merge(gender_submission, how='left', left_on='PassengerId', right_on='PassengerId').drop(['PassengerId','Survived'], axis=1)

Y_test = test.merge(gender_submission, how='left', left_on='PassengerId', right_on='PassengerId').Survived

res_tree = best_clf.score(X_test, Y_test)

print('Percentage of correctly predicted passengers who survived %.2f%%:' % (round(res_tree*100, 2)))
xmodel = xgboost.XGBClassifier()

xmodel.fit(X, Y)
imp_xgb = pd.DataFrame(xmodel.feature_importances_, 

              index=X.columns, columns=['importance'])

ax = imp_xgb.sort_values('importance').plot(kind='barh', figsize=(5, 5))
predictions = [round(value) for value in xmodel.predict(X_test)]
accuracy = accuracy_score(Y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
sc = StandardScaler() 



pca_train = sc.fit_transform(X) 

pca_test = sc.transform(X_test) 
pca = PCA(n_components = 2) 

 

pca_train = pca.fit_transform(pca_train) 

pca_test = pca.transform(pca_test) 

 

explained_variance = pca.explained_variance_ratio_ 
classifier = LogisticRegression(random_state = 0) 

classifier.fit(pca_train, Y)
y_pred = classifier.predict(pca_test) 
accuracy = accuracy_score(Y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
class Perceptron():

    def __init__(self, rate = 0.01, niter = 10):

        self.rate = rate

        self.niter = niter



    def fit(self, X, y):

        self.weight = np.zeros(1 + X.shape[1])

        self.errors = []



        for i in range(self.niter):

            err = 0

            for xi, target in zip(X, y):

                delta_w = self.rate * (target - self.predict(xi))

                self.weight[1:] += delta_w * xi

                self.weight[0] += delta_w

                err += int(delta_w != 0.0)

            self.errors.append(err)

        return self



    def net_input(self, X):



        return np.dot(X, self.weight[1:]) + self.weight[0]



    def predict(self, X):

        """Return class label after unit step"""

        return np.where(self.net_input(X) >= 0.0, 1, -1)
pn = Perceptron(0.1, niter=100)

pn.fit(np.array(X), np.array(Y))
ax = plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')

ax = plt.xlabel('Epochs')

ax = plt.ylabel('Number of misclassifications')

ax = plt.show()

ax
y_pred = pn.predict(np.array(X_test)) 

accuracy = accuracy_score(Y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))