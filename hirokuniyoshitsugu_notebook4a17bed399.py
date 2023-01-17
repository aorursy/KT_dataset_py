import sys
print(sys.version)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from IPython.display import Image
Image(url='http://graphics8.nytimes.com/images/section/learning/general/onthisday/big/0415_big.gif')
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image

# Pandasの設定をします
pd.set_option('chained_assignment', None)

# matplotlibのスタイルを指定します。これでグラフが少しかっこよくなります。
plt.style.use('ggplot')
plt.rc('xtick.major', size=0)
plt.rc('ytick.major', size=0)
df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
df_train.tail()
df_test.tail()
df_train.groupby('Survived').count()
df_test.count()
x = df_train['Sex']
y = df_train['Survived']
x.head()
y.head()
y_pred = x.map({'female': 1, 'male': 0}).astype(int)
y_pred.head()
print('Accuracy: {:.3f}'.format(accuracy_score(y, y_pred)))
y.tail(n=10)
y_pred.tail(n=10)
print(classification_report(y, y_pred))
cm = confusion_matrix(y, y_pred)
print(cm)
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    fig.colorbar(im)

    target_names = ['not survived', 'survived']

    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()

plot_confusion_matrix(cm)
x_test = df_test['Sex']
y_test_pred = x_test.map({'female': 1, 'male': 0}).astype(int)
df_kaggle = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived':np.array(y_test_pred)})
df_kaggle.to_csv('kaggle_gendermodel.csv', index=False)
df_kaggle.head()
## 4. ロジスティック回帰による生存者推定¶ 
X = df_train[['Age', 'Pclass', 'Sex']]
y = df_train['Survived']
X.tail()
X['AgeFill'] = X['Age'].fillna(X['Age'].mean())
X = X.drop(['Age'], axis=1)
X['Gender'] = X['Sex'].map({'female': 0, 'male': 1}).astype(int)
X.tail()
X['Pclass_Gender'] = X['Pclass'] + X['Gender']
X.tail()
X = X.drop(['Pclass', 'Sex', 'Gender'], axis=1)
X.head()
np.random.seed = 0

xmin, xmax = -5, 85
ymin, ymax = 0.5, 4.5

index_survived = y[y==0].index
index_notsurvived = y[y==1].index

fig, ax = plt.subplots()
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
sc = ax.scatter(X.loc[index_survived, 'AgeFill'],
                X.loc[index_survived, 'Pclass_Gender']+(np.random.rand(len(index_survived))-0.5)*0.1,
                color='r', label='Not Survived', alpha=0.3)
sc = ax.scatter(X.loc[index_notsurvived, 'AgeFill'],
                X.loc[index_notsurvived, 'Pclass_Gender']+(np.random.rand(len(index_notsurvived))-0.5)*0.1,
                color='b', label='Survived', alpha=0.3)
ax.set_xlabel('AgeFill')
ax.set_ylabel('Pclass_Gender')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.legend(bbox_to_anchor=(1.4, 1.03))
plt.show()
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=1)
X_train.head()
X_val.head()
X_train.count()
X_val.count()
print('Num of Training Samples: {}'.format(len(X_train)))
print('Num of Validation Samples: {}'.format(len(X_val)))
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(y_train, y_train_pred)))
print('Accuracy on Validation Set: {:.3f}'.format(accuracy_score(y_val, y_val_pred)))
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

plot_confusion_matrix(cm)
h = 0.02
xmin, xmax = -5, 85
ymin, ymax = 0.5, 4.5
xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
levels = np.linspace(0, 1.0, 5)
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
contour = ax.contourf(xx, yy, Z, cmap=cm, levels=levels, alpha=0.8)
ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1]+(np.random.rand(len(X_train))-0.5)*0.1, c=y_train, cmap=cm_bright)
ax.scatter(X_val.iloc[:, 0], X_val.iloc[:, 1]+(np.random.rand(len(X_val))-0.5)*0.1, c=y_val, cmap=cm_bright, alpha=0.5)
ax.set_xlabel('AgeFill')
ax.set_ylabel('Pclass_Gender')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
fig.colorbar(contour)

x1 = xmin
x2 = xmax
y1 = -1*(clf.intercept_[0]+clf.coef_[0][0]*xmin)/clf.coef_[0][1]
y2 = -1*(clf.intercept_[0]+clf.coef_[0][0]*xmax)/clf.coef_[0][1]
ax.plot([x1, x2] ,[y1, y2], 'k--')

plt.show()
clf_log = LogisticRegression()
clf_svc_lin = SVC(kernel='linear', probability=True)
clf_svc_rbf = SVC(kernel='rbf', probability=True)
titles = ['Logistic Regression', 'SVC with Linear Kernel', 'SVC with RBF Kernel',]

h = 0.02
xmin, xmax = -5, 85
ymin, ymax = 0.5, 4.5
xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))

fig, axes = plt.subplots(1, 3, figsize=(12,4))
levels = np.linspace(0, 1.0, 5)
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
for i, clf in enumerate((clf_log, clf_svc_lin, clf_svc_rbf)):
    clf.fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axes[i].contourf(xx, yy, Z, cmap=cm, levels=levels, alpha=0.8)
    axes[i].scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cm_bright)
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('AgeFill')
    axes[i].set_ylabel('Pclass_Gender')
    axes[i].set_xlim(xmin, xmax)
    axes[i].set_ylim(ymin, ymax)
    fig.tight_layout()
clf = SVC(kernel='rbf', probability=True, gamma='auto')
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)

print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(y_train, y_train_pred)))
print('Accuracy on Validation Set: {:.3f}'.format(accuracy_score(y_val, y_val_pred)))
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=33)

clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train )

y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)

print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(y_train, y_train_pred)))
print('Accuracy on Test Set: {:.3f}'.format(accuracy_score(y_val, y_val_pred)))
Image(url='http://scott.fortmann-roe.com/docs/docs/MeasuringError/crossvalidation.png')
def cross_val(clf, X, y, K=5, random_state=0):
    cv = KFold(K, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=cv)
    return scores
cv = KFold(5, shuffle=True, random_state=0)
cv
import warnings
warnings.filterwarnings('ignore')
clf = LogisticRegression()
scores = cross_val(clf, X, y)
print('Scores:', scores)
print('Mean Score: {0:.3f} (+/-{1:.3f})'.format(scores.mean(), scores.std()*2))
X = df_train[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']]
y = df_train['Survived']
X_test = df_test[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']]
X.tail()
X['AgeFill'] = X['Age'].fillna(X['Age'].mean())
X_test['AgeFill'] = X_test['Age'].fillna(X['Age'].mean())

X = X.drop(['Age'], axis=1)
X_test = X_test.drop(['Age'], axis=1)
le = LabelEncoder()
le.fit(X['Sex'])
X['Gender'] = le.transform(X['Sex'])
X_test['Gender'] = le.transform(X_test['Sex'])
classes = {gender: i for (i, gender) in enumerate(le.classes_)}
print(classes)
X.tail()
X = X.join(pd.get_dummies(X['Embarked'], prefix='Embarked'))
X_test = X_test.join(pd.get_dummies(X['Embarked'], prefix='Embarked'))
X.tail()
X = X.drop(['Sex', 'Embarked'], axis=1)
X_test = X_test.drop(['Sex', 'Embarked'], axis=1)
clf = LogisticRegression()
scores = cross_val(clf, X, y)
print('Scores:', scores)
print('Mean Score: {0:.3f} (+/-{1:.3f})'.format(scores.mean(), scores.std()*2))
clf = DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=2)
scores = cross_val(clf, X, y, 5)
print('Scores:', scores)
print('Mean Score: {0:.3f} (+/-{1:.3f})'.format(scores.mean(), scores.std()*2))
Image(url='https://raw.githubusercontent.com/PyDataTokyo/pydata-tokyo-tutorial-1/master/images/titanic_decision_tree.png')
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=2)
scores = cross_val(clf, X, y, 5)
print('Scores:', scores)
print('Mean Score: {0:.3f} (+/-{1:.3f})'.format(scores.mean(), scores.std()*2))
clf = DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=2)

param_grid = {'max_depth': [2, 3, 4, 5], 'min_samples_leaf': [2, 3, 4, 5]}
cv = KFold(5, shuffle=True, random_state=0)

grid_search = GridSearchCV(clf, param_grid, cv=cv, n_jobs=-1, verbose=1,return_train_score=True)
grid_search.fit(X, y)
print('Scores: {:.3f}'.format(grid_search.best_score_))
print('Best Parameter Choice:', grid_search.best_params_)
grid_search.cv_results_['mean_test_score']
grid_search.cv_results_['mean_test_score']
scores = grid_search.cv_results_['mean_test_score'].reshape(4, 4)

fig, ax = plt.subplots()
cm = plt.cm.Blues
mat = ax.matshow(scores, cmap=cm)
ax.set_xlabel('min_samples_leaf')
ax.set_ylabel('max_depth')
ax.set_xticklabels(['']+param_grid['min_samples_leaf'])
ax.set_yticklabels(['']+param_grid['max_depth'])
fig.colorbar(mat)
plt.show()
y_test_pred = grid_search.predict(X_test)
df_kaggle = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived':np.array(y_test_pred)})
df_kaggle.to_csv('kaggle_decisiontree.csv', index=False)