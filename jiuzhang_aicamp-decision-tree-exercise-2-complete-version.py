# 必要的引入
%matplotlib inline
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv(os.path.join("../input", "titanic.csv"), sep=",")
data.info()
data.head(3)
data['survived'].value_counts(normalize=True)
sns.countplot(data['survived'])
sns.countplot(data['pclass'], hue=data['survived'])
data['name'].head()
data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
data['name_title'].value_counts()
data['survived'].groupby(data['name_title']).mean()
data['name_len'] = data['name'].apply(lambda x: len(x))
data['survived'].groupby(pd.qcut(data['name_len'], 5)).mean()
data['sex'].value_counts(normalize=True)
data['survived'].groupby(data['sex']).mean()
data['survived'].groupby(pd.qcut(data['age'], 5)).mean()
data['embarked'].value_counts()
data['survived'].groupby(data['embarked']).mean()
sns.countplot(data['embarked'], hue=data['pclass'])
data['survived'].groupby(data['home.dest'].apply(lambda x: str(x).split(',')[-1])).mean()
def name(data):
    data['name_len'] = data['name'].apply(lambda x: len(x))
    data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
    del data['name']
    return data

def age(data):
    data['age_flag'] = data['age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    grouped_age = data.groupby(['name_title', 'pclass'])['age']
    data['age'] = grouped_age.transform(lambda x: x.fillna(data['age'].mean()) if pd.isnull(x.mean()) else x.fillna(x.mean()))
    return data

def embark(data):
    data['embarked'] = data['embarked'].fillna('Southampton')
    return data


def dummies(data, columns=['pclass','name_title','embarked', 'sex']):
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data
drop_columns = ['row.names', 'home.dest', 'room', 'ticket', 'boat'] #+ ['ticket_len', 'ticket_title']
data = data.drop(drop_columns, axis=1)
data.head()
data = name(data)
data = age(data)
data = embark(data)
data = dummies(data)
data.head()
from sklearn.model_selection import train_test_split
from sklearn import tree
trainX, testX, trainY, testY = train_test_split(data.iloc[:,1:], data.iloc[:,0], test_size=0.2, random_state=33)

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
model.fit(trainX, trainY)

from sklearn import metrics
def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    y_pred = clf.predict(X)
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")
    
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y, y_pred), "\n")
    
    if show_confussion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y, y_pred), "\n")
measure_performance(testX, testY, model)
sub_columns = ['age', 'sex_male','sex_female']
sub_trainX = trainX[sub_columns]
sub_testX = testX[sub_columns]
sub_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
sub_model.fit(sub_trainX, trainY)
measure_performance(sub_testX, testY, sub_model)
import graphviz
dot_data = tree.export_graphviz(model, out_file=None, feature_names=trainX.columns) 
graph = graphviz.Source(dot_data) 
#graph.render("titanic") 
#graph.view()
graph
pd.concat([pd.DataFrame(trainX.columns, columns=['variable']),
         pd.DataFrame(model.feature_importances_, columns=['importance'])],
         axis=1).sort_values(by='importance', ascending=False)[:20]

