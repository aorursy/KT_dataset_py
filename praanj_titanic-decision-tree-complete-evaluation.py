import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn import metrics



from sklearn import preprocessing



from yellowbrick.classifier import ConfusionMatrix

from yellowbrick.classifier import ClassificationReport

from yellowbrick.classifier import ROCAUC

plt.style.use('ggplot')



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
dataset = pd.read_csv('../input/titanic/train.csv')

dataset.head()
dataset.describe(include = "all")
dataset.shape
dataset.isnull().sum(axis=0)
sns.countplot(dataset['Embarked'])
# File missing values in embarked with S which is the most frequent item.

dataset = dataset.fillna({"Embarked": "S"})
## One hot encoding is used since no ordering is available for Sex (male, female) feature.

dataset = pd.get_dummies(dataset, columns=['Sex'])

dataset.head()
## One hot encoding is used since no ordering is available for Sex (male, female) feature.

dataset = pd.get_dummies(dataset, columns=['Embarked'])

dataset.head()
feat_names = ['Pclass', 'Sex_male', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Parch', 'SibSp', 'Fare']

targ_names = ['Dead (0)', 'Survived (1)'] # 0 - Dead, 1 - Survived



train_class = dataset[['Survived']]

train_feature = dataset[feat_names]

train_feature.head()
clf = DecisionTreeClassifier(random_state=0)

scoring = {'acc': 'accuracy',

           'prec_macro': 'precision_macro',

           'rec_macro': 'recall_macro',

           'f1_macro': 'f1_macro'}

scores = cross_validate(clf, train_feature, train_class, cv=10, scoring=scoring)

# print(scores.keys())



print ('Accuracy score : %.3f' % scores['test_acc'].mean())

print ('Precisoin score : %.3f' % scores['test_prec_macro'].mean())

print ('Recall score : %.3f' % scores['test_rec_macro'].mean())

print ('F1 score : %.3f' % scores['test_f1_macro'].mean())
para_grid = {

    'min_samples_split' : range(10,500,20),

    'max_depth': range(1,20,2),

    'criterion': ("gini", "entropy")

}



clf_tree = DecisionTreeClassifier()

clf_cv = GridSearchCV(clf_tree,

                   para_grid,

                   scoring='accuracy',

                   cv=5,

                   n_jobs=-1)

clf_cv.fit(train_feature,train_class)



best_parameters = clf_cv.best_params_

print(best_parameters)
clf = clf_cv.best_estimator_

scoring = {'acc': 'accuracy',

           'prec_macro': 'precision_macro',

           'rec_macro': 'recall_macro',

           'f1_macro': 'f1_macro'}

scores = cross_validate(clf, train_feature, train_class, cv=10, scoring=scoring)

#print(scores.keys())



print ('Accuracy score : %.3f' % scores['test_acc'].mean())

print ('Precisoin score : %.3f' % scores['test_prec_macro'].mean())

print ('Recall score : %.3f' % scores['test_rec_macro'].mean())

print ('F1 score score : %.3f' % scores['test_f1_macro'].mean())
# Create a holdout sample for further testing

# train_class, train_feature

X_train, X_test, y_train, y_test = train_test_split(train_feature, train_class, test_size=0.33)

print (str(X_train.shape) +","+ str(y_train.shape))

print (str(X_test.shape) +","+ str(y_test.shape))
clf2 = clf_cv.best_estimator_

clf2.fit(X_train,y_train)

predictions = clf2.predict(X_test)

print(metrics.classification_report(y_test,predictions, target_names=targ_names, digits=3))
fig, ax = plt.subplots(figsize=(7,3))

visualizer = ClassificationReport(clf2, classes=targ_names, support=True, cmap='RdPu')

visualizer.score(X_test, y_test)

for label in visualizer.ax.texts:

    label.set_size(14)

g = visualizer.poof()
fig, ax = plt.subplots(figsize=(3,3))

cm = ConfusionMatrix(clf2, classes=[0, 1], cmap='RdPu')

cm.score(X_test, y_test)

for label in cm.ax.texts:

    label.set_size(14)

cm.poof()
modelviz = clf_cv.best_estimator_

visualizer = ROCAUC(modelviz, classes=["Dead", "Survived"])



visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.show()                       # Finalize and render the figure
import graphviz

from sklearn.tree import DecisionTreeClassifier, export_graphviz



data = export_graphviz(clf,out_file=None,feature_names=feat_names,class_names=targ_names,   

                         filled=True, rounded=True,  

                         special_characters=True)

graph = graphviz.Source(data)

graph
importances = clf.feature_importances_

indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [feat_names[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
# Loading test dataset

test = pd.read_csv('../input/titanic/test.csv')



# Fit the model

clf.fit(train_feature, train_class)



# Replace missing Fare values with mean

meanFare = dataset['Fare'].mean()

test = test.fillna({"Fare": meanFare})

# Categorical -> One hot encoding

test = pd.get_dummies(test, columns=['Sex'])

test = pd.get_dummies(test, columns=['Embarked'])



#set ids as PassengerId and predict survival

ids = test['PassengerId']

test_feature = test[feat_names]

predictions = clf.predict(test_feature)



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.head()
output.to_csv('submission.csv', index=False)