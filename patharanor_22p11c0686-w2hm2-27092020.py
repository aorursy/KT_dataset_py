import numpy as np
import pandas as pd 

from matplotlib import pyplot as plt
from sklearn import tree, metrics
import graphviz

files = []

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
gender_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
print(train_df.info())
print('\nSurvive:\n{}'.format(train_df['Survived'].value_counts()))
train_df.tail()
print(gender_df.info())
print('\nSurvive:\n{}'.format(gender_df['Survived'].value_counts()))
gender_df
test_df.info()
print('PassengerId:\n')
print(np.array(train_df['PassengerId']))
print('--------------------------------------')
print('Survived:\n')
print(train_df['Survived'].value_counts())
print('--------------------------------------')
print('Pclass:\n')
print(train_df['Pclass'].value_counts())
print('--------------------------------------')
print('Name:\n')
print(train_df['Name'].value_counts())
print('--------------------------------------')
print('Sex:\n')
print(train_df['Sex'].value_counts())
print('--------------------------------------')
print('Age (mean {}):\n'.format(train_df['Age'].mean()))
print(train_df['Age'].value_counts())
print('--------------------------------------')
print('SibSp:\n')
print(train_df['SibSp'].value_counts())
print('--------------------------------------')
print('Parch:\n')
print(train_df['Parch'].value_counts())
print('--------------------------------------')
print('Ticket:\n')
print(np.array(train_df['Ticket']))
print('--------------------------------------')
print('Fare:\n')
print(train_df['Fare'].value_counts())
print('--------------------------------------')
print('Cabin ({} rooms):\n'.format(len(train_df['Cabin'].unique())))
print(train_df['Cabin'].value_counts())
print('Unique:', train_df['Cabin'].unique())
print('--------------------------------------')
print('Embarked:\n')
print(train_df['Embarked'].value_counts())
print('--------------------------------------')
def encode_sex(df):
    return df.apply(lambda x : 1 if x['Sex']=='male' else 0, axis=1)
def encode_cabin_zone(df):
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    return df.apply(lambda x : alphabet.index(x['Cabin'][:1].lower()) if isinstance(x['Cabin'], str) else 0, axis=1)
tmp_train_df = train_df.copy(True)
tmp_train_df['Sex'] = encode_sex(tmp_train_df)
tmp_train_df['Zone'] = encode_cabin_zone(tmp_train_df)
tmp_train_df = tmp_train_df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
tmp_train_df = tmp_train_df.dropna()

y_train = tmp_train_df['Survived']
tmp_train_df = tmp_train_df.drop(['Survived'], axis=1)

print(tmp_train_df)

tmp_test_df = test_df.copy(True)
tmp_test_df['Sex'] = encode_sex(tmp_test_df)
tmp_test_df['Zone'] = encode_cabin_zone(tmp_test_df)
tmp_test_df = tmp_test_df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
tmp_test_df = tmp_test_df.dropna()

print(tmp_test_df)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(tmp_train_df, y_train)
predict_result = clf.predict(tmp_test_df)

fig = plt.figure(figsize=(25,20))
tree.plot_tree(clf, filled = True)
fig.savefig("decistion_tree.svg")

dot_data = tree.export_graphviz(clf, out_file=None) 
text_representation = tree.export_text(clf)
graph = graphviz.Source(dot_data) 
graph.render("iris") 
truth = gender_df[gender_df['PassengerId'].isin(tmp_test_df['PassengerId'])]
print('\nAccuracy:', metrics.accuracy_score(truth['Survived'], predict_result))
naive_bayes_train_df = train_df.copy(True)


# Import LabelEncoder
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(tmp_train_df, y_train)
nb_predict = nb_clf.predict(tmp_test_df)

truth = gender_df[gender_df['PassengerId'].isin(tmp_test_df['PassengerId'])]
print('\nAccuracy:', metrics.accuracy_score(truth['Survived'], nb_predict))
from sklearn.neural_network import MLPClassifier

nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), max_iter=10000)
nn_clf.fit(tmp_train_df, y_train)
nn_predict = nn_clf.predict(tmp_test_df)

truth = gender_df[gender_df['PassengerId'].isin(tmp_test_df['PassengerId'])]
print('\nAccuracy:', metrics.accuracy_score(truth['Survived'], nn_predict))
def plot_cv(cv, title, prec, rec, f1, f1avg):
    k_fold_cv = [i for i in range(cv)]
    
    plt.rcParams["figure.figsize"] = (20,15)
    
    plt.subplot(221)
    plt.plot(k_fold_cv, prec, label='Precision')
    plt.plot(k_fold_cv, rec, label='Recall')
    plt.plot(k_fold_cv, f1, label='F1-Measure')
    plt.plot(k_fold_cv, f1avg, label='AVG. F1-Measure')
    plt.title(label='Compare 4-Scoring of 5-fold {}'.format(title))
    plt.legend()
    
    plt.subplot(222)
    plt.plot(k_fold_cv[cv-1], prec[cv-1], marker='o', markersize=3, label='Precision')
    plt.plot(k_fold_cv[cv-1], rec[cv-1], marker='o', markersize=3, label='Recall')
    plt.plot(k_fold_cv[cv-1], f1[cv-1], marker='o', markersize=3, label='F1-Measure')
    plt.plot(k_fold_cv[cv-1], f1avg[cv-1], marker='o', markersize=3, label='AVG. F1-Measure')
    plt.title(label='Last 4-scores of 5-fold {}'.format(title))
    plt.legend()
    
    plt.show()
def plot_compare(scroing_names, max_model_scoring):
    
    plt.rcParams["figure.figsize"] = (15,10)
    # set width of bar
    barWidth = 0.25

    # set height of bar
    bars1 = max_model_scoring['decision tree']
    bars2 = max_model_scoring['naive bayes']
    bars3 = max_model_scoring['neural network']

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='Decision Tree')
    plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='Naive Bayes')
    plt.bar(r3, bars3, width=barWidth, edgecolor='white', label='Neural Network')

    # Add xticks on the middle of the group bars
    plt.xlabel('4-Scoring of 5-fold in each classifier', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], scroing_names)

    # Create legend & Show graphic
    plt.legend()

    
    plt.show()
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, recall_score

k_fold = 5

models = {
    'decision tree': clf,
    'naive bayes': nb_clf,
    'neural network': nn_clf,
}

scoring = {
    'precision': 'precision_macro',
    'recall': make_scorer(recall_score, average='macro'),
    'f1': 'f1',
    'f1_avg': 'f1_macro',
}

scroing_names = [scroing_name for scroing_name, scroing in scoring.items()]
max_model_scoring = {
    'decision tree': [],
    'naive bayes': [],
    'neural network': [],
}

def f1_measure(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

for model_name, model in models.items():
    scores = cross_validate(model, tmp_train_df, y_train, scoring=scoring, cv=5, return_train_score=True)
    sorted_score = sorted(scores.keys())
    
    print('Score of model {} : '.format(model_name))
    for s_index in range(len(sorted_score)):
        print(' - {} => {}'.format(sorted_score[s_index], scores[sorted_score[s_index]]))
        
    plot_cv(k_fold, model_name, scores['test_precision'],scores['test_recall'],scores['test_f1'],scores['test_f1_avg'])
    
    # Record max
    max_model_scoring[model_name].append(scores['test_precision'][k_fold-1])
    max_model_scoring[model_name].append(scores['test_recall'][k_fold-1])
    max_model_scoring[model_name].append(scores['test_f1'][k_fold-1])
    max_model_scoring[model_name].append(scores['test_f1_avg'][k_fold-1])
    
print(max_model_scoring)
plot_compare(scroing_names, max_model_scoring)