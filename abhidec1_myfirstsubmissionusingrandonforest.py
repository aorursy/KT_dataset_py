import pandas as pd

from sklearn.preprocessing import Imputer

from sklearn import tree

from sklearn import metrics

import numpy as np

import matplotlib.pyplot as plt

% matplotlib inline

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df
train_df["Sex"] = train_df["Sex"].apply(lambda sex: 0 if sex == 'male' else 1)
y = targets = labels = train_df["Survived"].values



columns = ["Fare", "Pclass", "Sex", "Age", "SibSp"]

features = train_df[list(columns)].values

features
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

X = imp.fit_transform(features)

X
my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)

my_tree_one = my_tree_one.fit(X, y)
#The feature_importances_ attribute make it simple to interpret the significance of the predictors you include

print(my_tree_one.feature_importances_) 

print(my_tree_one.score(X, y))
with open("titanic.dot", 'w') as f:

    f = tree.export_graphviz(my_tree_one, out_file=f, feature_names=columns)
test_df.head()
test_df["Sex"] = test_df["Sex"].apply(lambda sex: 0 if sex == 'male' else 1)

#features_test = train_df[list(columns)].values

features_test = test_df[list(columns)].values

imp_test = Imputer(missing_values='NaN', strategy='mean', axis=0)

X_test = imp_test.fit_transform(features_test)

X_test
pred = my_tree_one.predict(X_test)

pred

#Print Confusion matrix 

pred = my_tree_one.predict(X)

df_confusion = metrics.confusion_matrix(y, pred)

df_confusion


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):

    plt.matshow(df_confusion, cmap=cmap) # imshow

    plt.title('Confusion Matrix')

    plt.colorbar()

    plt.ylabel('Actual')

    plt.xlabel('Predicted')



plot_confusion_matrix(df_confusion)
#Setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two

max_depth = 10

min_samples_split = 5

my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

my_tree_two = my_tree_two.fit(X, y)



#Print the score of the new decison tree

print(my_tree_two.score(X, y))
pred = my_tree_two.predict(X)
df_confusion = metrics.confusion_matrix(y, pred)

df_confusion
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):

    plt.matshow(df_confusion, cmap=cmap) # imshow

    plt.title('Confusion Matrix')

    plt.colorbar()

    plt.ylabel('Actual')

    plt.xlabel('Predicted')



plot_confusion_matrix(df_confusion)
# Add new feature and then train the model



train_df["family_size"] = train_df.SibSp + train_df.Parch + 1

from sklearn.ensemble import RandomForestClassifier



# Building and fitting my_forest

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(X, y)



# Print the score of the fitted random forest

print(my_forest.score(X, y))



pred = my_forest.predict(X)
df_confusion = metrics.confusion_matrix(y, pred)

df_confusion
fpr, tpr, _ = metrics.roc_curve(y, pred)

roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
# ROC curve on Predicted probabilities

pred_proba = my_forest.predict_proba(X)

fpr, tpr, _ = metrics.roc_curve(y, pred_proba[:,1])

roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
PassengerId = train_df['PassengerId']

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': pred})

StackingSubmission.to_csv("MySubmission.csv", index=False)