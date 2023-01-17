# load the pandas library
import pandas as pd

# load the data
demographic = pd.read_csv('../input/national-health-and-nutrition-examination-survey/demographic.csv', index_col=False)
labs = pd.read_csv('../input/national-health-and-nutrition-examination-survey/labs.csv', index_col=False)
questionnaire = pd.read_csv('../input/national-health-and-nutrition-examination-survey/questionnaire.csv', index_col=False)
demographic.shape
demographic.head()
# get only gender and age columns
demographic = demographic[['SEQN', 'RIDAGEYR', 'RIAGENDR']]

# change the names of columns
demographic.rename(columns={'SEQN': 'ID', 'RIDAGEYR': 'Age', 'RIAGENDR': 'Gender'}, inplace=True)

demographic.head()
# Load the visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(demographic['Gender'], label='Count', palette='husl')
plt.title("Distrubiton of Gender")
plt.show()
male = (demographic['Gender']==1).sum()
female = (demographic['Gender']==2).sum()

print("There are {} male attenders".format(male))
print("There are {} female attenders".format(female))
print("There are {} different ages".format(len(demographic['Age'].unique())))
labs.shape
labs.head()
# Get the columns of id and B12
labs = labs[['SEQN', 'LBDB12']]

# Change the names of columns
labs.rename(columns={'SEQN': 'ID', 'LBDB12': 'B12'}, inplace=True)

labs.head()
labs['B12'].dtype
labs['B12'].isnull().sum()
labs['B12'].describe()
questionnaire.shape
questionnaire.head()
# Get the columns
questionnaire = questionnaire[['SEQN', 'DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090', 'DPQ100', 'SLQ050', 'SLQ060', 'IND235']]

# Change the name of id column
questionnaire.rename(columns={'SEQN': 'ID'}, inplace=True)

questionnaire.head()
questionnaire['DPQ020'].unique()
questionnaire['SLQ050'].unique()
# categorize the data
demographic['Age'] = pd.cut(demographic['Age'], bins=[20, 30, 40, 50, 60, 70, 80], labels=[1, 2, 3, 4, 5, 6], include_lowest=True)

demographic.head()
# visualize the distrubiton of age
sns.countplot(demographic['Age'], palette='husl')
plt.ylabel('Count')
plt.title('Distrubiton of Age')
plt.show()
# Drop the records with null value in column B12
labs.dropna(axis=0, inplace=True)

labs.shape
# Categorize the data
labs['B12'] = pd.cut(labs['B12'], bins=[0, 300, 950, 27000], labels=[1, 2, 3])

labs.head()

# Visualize the distribution of B12 values
sns.countplot(labs['B12'], label='Count', palette='husl')
plt.title('Distribution of B12')
plt.show()
labs['B12'].value_counts()
data = labs.merge(demographic, on="ID").merge(questionnaire, on="ID")

data.head()
data.drop('ID', axis=1, inplace=True)

data.head()
data.hist(figsize=(15, 10),grid=False)
plt.show()
data.info()
data.drop('DPQ100', axis=1, inplace=True)
data.head()
data.info()
data.dropna(subset=['DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090'], axis=0, inplace=True)
data.info()
data['IND235'].describe()
data['IND235'].replace({77: 7, 99: 7, None: 7}, inplace=True)

data[['DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090', 'SLQ050', 'SLQ060']].replace({7: 0, 9: 0})
data.info()
data['B12'] = data['B12'].astype(int)
data['Age'] = data['Age'].astype(int)
# load the numpy library
import numpy as np

# graph size
plt.figure(figsize=(15, 10))

# to get the lower diagonal of the correlation matrix, using the numpy library
mask = np.triu(data.iloc[:, 1:].corr())

sns.heatmap(data.iloc[:, 1:].corr(), annot=True, mask=mask)

# change the direction of ticks on y axis
plt.yticks(rotation=0)

plt.show()
# load the libraries
from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import density

from sklearn import metrics

# split the data into features and target
X = data.iloc[:, 1:15]
y = data.iloc[:, 0]

# split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True) # 70% training and 30% test
from time import time
# Benchmark classifier
def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time
# Different algorithms
results = []
for clf, name in (
        (tree.DecisionTreeClassifier(), "Decision Tree"),
        (RandomForestClassifier(n_estimators=100), "Random Forest"),
        (GaussianNB(), "Gauissian Naive Bayes"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (LogisticRegression(), "Logistic Regression"),
        (svm.SVC(), "Support Vector Machines"),
    ):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf, X_train, y_train, X_test, y_test))
# plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
from sklearn.model_selection import GridSearchCV

forest = RandomForestClassifier(random_state=1, n_estimators = 10, min_samples_split = 1)

n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(X_train, y_train)
# best parameters on model
gridF.best_params_
gridF.refit
preds = gridF.predict(X_test)
probs = gridF.predict_proba(X_test)

# accuracy score
np.mean(preds == y_test)
print(metrics.classification_report(y_test, preds, target_names=['Low', 'Normal', 'High']))
balanced_data = data.copy()

balanced_data[balanced_data['B12'] == 2] = balanced_data[balanced_data['B12'] == 2].sample(1000, random_state=42)
balanced_data = balanced_data[balanced_data['B12'].notnull()]
balanced_data['B12'].value_counts()
# split the data into features and target
X_balanced = balanced_data.iloc[:, 1:15]
y_balanced = balanced_data.iloc[:, 0]

# split the data into test and train
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42, shuffle=True) # 70% training and 30% test
results_balanced = []
for clf, name in (
        (tree.DecisionTreeClassifier(), "Decision Tree"),
        (RandomForestClassifier(n_estimators=100), "Random Forest"),
        (GaussianNB(), "Gauissian Naive Bayes"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (LogisticRegression(), "Logistic Regression"),
        (svm.SVC(), "Support Vector Machines"),
    ):
    print('=' * 80)
    print(name)
    results_balanced.append(benchmark(clf, X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced))
# make plots

indices = np.arange(len(results_balanced))

results_balanced = [[x[i] for x in results_balanced] for i in range(4)]

clf_names, score, training_time, test_time = results_balanced
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
gridF_balanced = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)
bestF_balanced = gridF_balanced.fit(X_train_balanced, y_train_balanced)
gridF_balanced.best_params_
gridF_balanced.refit
preds = gridF_balanced.predict(X_test_balanced)
probs = gridF_balanced.predict_proba(X_test_balanced)

# accuracy score
np.mean(preds == y_test_balanced)
print(metrics.classification_report(y_test_balanced, preds, target_names=['Low', 'Normal', 'High']))
model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy score is {}".format(metrics.accuracy_score(y_test, pred)))
print(metrics.classification_report(y_test, pred, target_names=['Low', 'Normal', 'High']))
rslt = metrics.confusion_matrix(y_test, pred)
sns.heatmap(pd.DataFrame(rslt, index=['Low Predicted', 'Normal Predicted', 'High Predicted'], columns=['Actual Low', 'Actual Normal', 'Actual High']), annot=True, fmt='d', cmap='Blues')
plt.title('Unbalanced Data Confision Matrix')
plt.show()
model.fit(X_train_balanced, y_train_balanced)
pred_balanced = model.predict(X_test_balanced)

print("Accuracy score for balanced data is {}".format(metrics.accuracy_score(y_test_balanced, pred_balanced)))
print(metrics.classification_report(y_test_balanced, pred_balanced, target_names=['Low', 'Normal', 'High']))
rslt_balanced = metrics.confusion_matrix(y_test_balanced, pred_balanced)
sns.heatmap(pd.DataFrame(rslt_balanced, index=['Low Predicted', 'Normal Predicted', 'High Predicted'], columns=['Actual Low', 'Actual Normal', 'Actual High']), annot=True, fmt='d', cmap='Blues')
plt.title('Balanced Data Confision Matrix')
plt.show()