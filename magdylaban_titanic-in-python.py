# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_set = pd.read_csv('/kaggle/input/titanic/train.csv')
test_set = pd.read_csv('/kaggle/input/titanic/test.csv')
train_set.head()
test_set.head()
train_set.info()
train_set.describe()
total_missing = train_set.isnull().sum().sort_values(ascending= False)
missing_percent = (train_set.isnull().sum()/train_set.isnull().count()) * 100
rounded_percent = round(missing_percent,1).sort_values(ascending= False)
missing_train = pd.concat([total_missing,rounded_percent], axis = 1, keys=['Total','Percent %'])
missing_train
age_sex_plot, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
ax = sns.distplot(train_set[(train_set['Sex']=='female') & (train_set['Survived'] == 1)].Age.dropna(), bins = 18,
                  label = 'Survived',ax = axes[0], kde = False)
ax = sns.distplot(train_set[(train_set['Sex']=='female') & (train_set['Survived'] == 0)].Age.dropna(), bins = 40,
                  label = 'Not Survived', ax = axes[0], kde = False)
ax.legend()
ax.set_title('(Age vs Survived) Female')
ax = sns.distplot(train_set[(train_set['Sex']=='male') & (train_set['Survived'] == 1)].Age.dropna(), bins = 18,
                  label = 'Survived', ax = axes[1], kde = False)
ax = sns.distplot(train_set[(train_set['Sex']=='male') & (train_set['Survived'] == 0)].Age.dropna(), bins = 40,
                  label = 'Not Survived', ax = axes[1], kde = False)
ax.legend()
ax.set_title('(Age vs Survived) male')
FacetGrid = sns.FacetGrid(train_set, row = 'Embarked', size = 4.5, aspect = 1.6)
FacetGrid.map(sns.pointplot,'Pclass','Survived','Sex', order=None, hue_order=None)
FacetGrid.add_legend()
sns.barplot(x=train_set['Pclass'],y=train_set['Survived'])
grid = sns.FacetGrid(train_set, col= 'Survived', row= 'Pclass', size= 2.2, aspect= 1.6)
grid.map(plt.hist,'Age', alpha = 0.5, bins = 20)
grid.add_legend()
full_data = [train_set, test_set]

for dataset in full_data:
    dataset['Relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['Relatives'] > 0, 'Not_alone'] = 0
    dataset.loc[dataset['Relatives'] == 0, 'Not_alone'] = 1
    dataset['Not_alone'] = dataset['Not_alone'].astype(int)
train_set['Not_alone'].value_counts()
axes = sns.factorplot('Relatives','Survived', data = train_set, aspect = 2.5)
train_set = train_set.drop(['PassengerId'], axis = 1)
import re 
Deck = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'U':8}
full_data = [train_set,test_set]

for dataset in full_data:
    dataset['Cabin'] = dataset['Cabin'].fillna('U0')
    dataset['Deck'] = dataset['Cabin'].map(
        lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(Deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
    
train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)
full_data = [train_set, test_set]

for dataset in full_data:
    age_mean = train_set['Age'].mean()
    age_std = test_set['Age'].std()
    total_age_nulls = dataset['Age'].isnull().sum()
    rand_age = np.random.randint(age_mean - age_std, age_mean + age_std,
                                 size = total_age_nulls)
    age_part = dataset['Age'].copy()
    age_part[np.isnan(age_part)] = rand_age
    dataset['Age'] = age_part
    dataset['Age'] = train_set['Age'].astype(int)

train_set['Age'].isnull().sum()
train_set['Embarked'].value_counts()
full_data = [train_set, test_set]

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train_set.info()
full_data = [train_set, test_set]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in full_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map(titles)
    dataset['Title'] = dataset['Title'].fillna(0)

pd.crosstab(train_set['Title'], train_set['Survived'])
train_set.drop(['Name'], axis = 1, inplace = True)
test_set.drop(['Name'], axis = 1, inplace = True)
gender = {'male': 0, 'female': 1}
full_data = [train_set, test_set]

for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map(gender)
train_set.drop(['Ticket'], axis = 1, inplace = True)
test_set.drop(['Ticket'], axis = 1, inplace = True)
embarked = {'S': 1, 'C': 2, 'Q': 3}
full_data = [train_set, test_set]

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked)
full_data = [train_set, test_set]

for dataset in full_data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
train_set['Age'].value_counts()
full_data = [train_set, test_set]

for dataset in full_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
train_set['Fare'] = train_set['Fare'].astype(int)
test_set.loc[test_set['Fare'].isnull(), 'Fare'] = 2
test_set['Fare'] = test_set['Fare'].astype(int)
test_set['Fare'].isnull().sum()
train_set['Fare'].value_counts()
full_data = [train_set, test_set]

for dataset in full_data:
    dataset['Age_per_class'] = dataset['Age'] * dataset['Pclass']
for dataset in full_data:
    dataset['Fare_per_person'] = dataset['Fare'] / (dataset['Relatives'] + 1)
    dataset['Fare_per_person'] = dataset['Fare_per_person'].astype(int)
train_set.head(10)
test_set.head(10)
X_train = train_set.drop('Survived', axis = 1)
Y_train = train_set['Survived']
X_test = test_set.drop(['PassengerId'], axis = 1).copy()
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
sgd = linear_model.SGDClassifier(max_iter = 5, tol = None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
Y_pred = log_reg.predict(X_test)
acc_log = round(log_reg.score(X_train, Y_train) * 100, 2)
print(acc_log)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)
gaussian = GaussianNB() 
gaussian.fit(X_train, Y_train)  
Y_pred = gaussian.predict(X_test)  
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result = results.sort_values(by='Score', ascending=False)
result = result.set_index('Score')
result
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
important_features = pd.DataFrame({'feature':X_train.columns,
                            'importance':np.round(random_forest.feature_importances_,3)})
important_features = important_features.sort_values('importance',ascending=False).set_index('feature')
important_features
important_features.plot.bar()
train_set = train_set.drop('Not_alone', axis=1)
test_set = test_set.drop('Not_alone', axis=1)

train_set = train_set.drop('Parch', axis=1)
test_set = test_set.drop('Parch', axis=1)
random_forest = RandomForestClassifier(n_estimators = 100, oob_score = True )
random_forest.fit(X_train, Y_train)
Y_pred_ = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)
print('out_of_bag score: ', round(random_forest.oob_score_, 4)* 100)
param_grid = { "criterion" : ["gini", "entropy"], 
              "min_samples_leaf" : [1, 5, 10, 25, 50, 70],
              "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35],
              "n_estimators": [100, 400, 700, 1000, 1500]}

from sklearn.model_selection import GridSearchCV, cross_val_score
rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True,
                            random_state=40, n_jobs=-1)
clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
clf.fit(X_train, Y_train)
clf.best_params_
random_forest = RandomForestClassifier(criterion = "entropy", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 16,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=40, 
                                       n_jobs=-1)

random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

print(random_forest.score(X_train, Y_train))

print("out_of_bag score:", round(random_forest.oob_score_, 4)*100)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
prediction = cross_val_predict(random_forest, X_train, Y_train, cv= 3)
confusion_matrix(Y_train, prediction)
from sklearn.metrics import precision_score, recall_score
print('Precision : ', precision_score(Y_train, prediction))
print('Recall : ', recall_score(Y_train, prediction))
from sklearn.metrics import f1_score
print('F-Score : ', f1_score(Y_train, prediction))
from sklearn.metrics import precision_recall_curve
Y_scores = random_forest.predict_proba(X_train)
Y_scores = Y_scores[:,1]
precision, recall, threshold = precision_recall_curve(Y_train, Y_scores)
def plot_precision_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])
    
plt.figure(figsize =(14, 7))
plot_precision_recall(precision, recall, threshold)
from sklearn.metrics import roc_curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, Y_scores)

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(Y_train, Y_scores)
print("ROC AUC Score:", r_a_score)
submission = pd.DataFrame(Y_prediction,test_set['PassengerId'], columns= ['Survived'])
submission.reset_index(inplace = True)
submission.to_csv('titanic.csv', index = False)
