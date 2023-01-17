import pandas as pd
import numpy as np
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv('../input/titanic/train.csv')
df_original = df.copy()
df.head()
df.isnull().sum()
df.dtypes
df['Age'].plot(kind='box', figsize=(6,6))
df['Age'].describe()
df_nullage = df[['Age', 'SibSp', 'Parch']][df['Age'].isnull()]
df_nullage
df_nullagewithfamily = df[['Age', 'SibSp', 'Parch']][df['Age'].isnull() & (df['SibSp'] + df['Parch'])>0]
df_nullagewithfamily
print(df_nullage.shape[0], '  ', df_nullagewithfamily.shape[0])
df = df[df['Age'].notnull()]
df.isnull().sum()
df['Embarked'][df['Embarked'].isnull()] = df['Embarked'].value_counts().idxmax()
df['Embarked'].value_counts()
df.isnull().sum()
df.drop('Cabin', axis=1, inplace=True)
df.head()
df.drop('Name', axis=1, inplace=True)
df.head()
df.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler
Fare = df[['Fare']]
df['Fare'] = StandardScaler().fit_transform(Fare)
Age = df[['Age']]
df['Age'] = StandardScaler().fit_transform(Age)
df.head()
df = pd.concat([df, pd.get_dummies(df[['Sex', 'Embarked']])], axis=1)
df.head()
df.drop(['Sex', 'Embarked'], axis=1, inplace=True)
df.head()
X = df.drop('Survived', axis=1)
y = df['Survived']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)
LR
yhat = LR.predict(x_test)
yhat[0:5]
yhat_prob = LR.predict_proba(x_test)
yhat_prob[0:5]
from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat)
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
for depth in range(2, 10):
    from sklearn.tree import DecisionTreeClassifier
    survivalTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
    survivalTree.fit(x_train,y_train)
    predTree = survivalTree.predict(x_test)
    from sklearn import metrics
    print("Max depth is: ", depth, "DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
survivalTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
survivalTree.fit(x_train,y_train)
predTree = survivalTree.predict(x_test)
from sklearn import metrics
print (predTree [0:5])
print (y_test [0:5])
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, predTree, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predTree, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
X.head()
y.head()
survivalTree.fit(X,y)
df_test = pd.read_csv('../input/titanic/test.csv')
df_test_copy = df_test.copy()
df_test.head()
df_test.isnull().sum()
df_test.drop(['PassengerId','Name'], axis=1, inplace=True)
df_test.head()
df_test.drop('Cabin', axis=1, inplace=True)
df_test['Age'].replace(np.nan, df_test['Age'].astype('float').mean(axis=0), inplace=True)
df_test['Fare'].replace(np.nan, df_test['Fare'].astype('float').mean(axis=0), inplace=True)
df_test.drop('Ticket', axis=1, inplace=True)
df_test.head()
df_test = pd.concat([df_test,pd.get_dummies(df_test[['Sex', 'Embarked']])], axis=1)
df_test.drop(['Sex', 'Embarked'], axis=1, inplace=True)
df_test.head()
Fare = df_test[['Fare']]
df_test['Fare'] = StandardScaler().fit_transform(Fare)
Age = df_test[['Age']]
df_test['Age'] = StandardScaler().fit_transform(Age)
df_test.head()
predTree = survivalTree.predict(df_test)
predTree[0:10]
df_test['Label'] = predTree
df_predict = pd.concat([df_test_copy['PassengerId'], df_test['Label']], axis=1)
df_predict.columns = ['PassengerId','Survived']
df_predict.set_index('PassengerId', drop=True, inplace=True) 
predicted = df_predict.to_csv('Prediction.csv')
df_predict