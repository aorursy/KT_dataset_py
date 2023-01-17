# Data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_recall_fscore_support
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
train_df.head()
train_df.describe()
test_df.describe()
pd.crosstab(train_df['Pclass'], train_df['Survived']).plot(kind='bar',stacked=True);
pd.crosstab(train_df['Sex'], train_df['Survived']).plot(kind='bar',stacked=True);
pd.crosstab(train_df['SibSp'], train_df['Survived']).plot(kind='bar',stacked=True);
sns.catplot(x="Pclass", y="Survived", hue="Sex",
            palette={"male": "g", "female": "m"},
            markers=["^", "o"], linestyles=["-", "--"],
            kind="point", data=train_df);
train_df.head()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['PassengerId', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin'], axis=1)
test_df = test_df.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
train_df.head()
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
train_df.describe()
train_df.describe(include=['O'])
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
test_df.head(10)
X_train = train_df[["Pclass", "Sex", "SibSp", "Parch", "Embarked"]]
Y_train = train_df["Survived"]
X_test  = test_df[["Pclass", "Sex", "SibSp", "Parch", "Embarked"]].copy()
X_train.shape, Y_train.shape, X_test.shape
# Decision Tree
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                           random_state=1234)
# decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
#                                            random_state=1234,
#                                            max_leaf_nodes=5)
decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
Y_pred = decision_tree.predict(X_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
# Confusion matrix
conf_mat = confusion_matrix(Y_pred, Y_train)
conf_mat = pd.DataFrame(conf_mat, 
                        index=['Predicted = dead', 'Predicted = survived'], 
                        columns=['Actual = dead', 'Actual = survived'])
conf_mat

import graphviz 
clf = tree.DecisionTreeClassifier()
dot_data = tree.export_graphviz(decision_tree, out_file=None,
                                feature_names=["Pclass", "Sex", "SibSp", "Parch", "Embarked"], 
                                class_names=['Dead','Suevived'],
                                filled=True, rounded=True, 
                                special_characters=True) 
graph = graphviz.Source(dot_data)
graph
Y_pred = decision_tree.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)