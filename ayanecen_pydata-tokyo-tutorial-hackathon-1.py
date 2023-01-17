# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

warnings.simplefilter("ignore")



%matplotlib inline



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

np.seterr(invalid='ignore') # Workaround
df = pd.read_csv("../input/train.csv")
df[df.Age == 65][["Name", "Age"]]
df.head(2)

df.tail()

df[['Name', 'Age', 'Sex']].head(3)

df.describe()

max_age = df['Age'].max()

print('年齢の最大値: {0}'.format(max_age))



mean_age = df['Age'].mean()

print('年齢の平均値: {0}'.format(mean_age))
df[df.Sex=='female'][['Name', 'Sex', 'Age']].sort_values(by='Age', ascending=False).head(10)
df['Cabin'].isnull().sum()
df[['Name', 'Ticket']].head()
df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
df.head()
df.loc[4:10]
df.loc[4:6][['Name', 'Age']].interpolate()
female_age_mean = round(df[df.Sex=='female']['Age'].mean())

male_age_mean = round(df[df.Sex=='male']['Age'].mean())



print('女性の平均年齢は{0}歳、男性は{1}歳です。この平均年齢で補間します。'.format(female_age_mean, male_age_mean))
round(df[df.Sex=='male']['Age'].mean())
df[df.PassengerId==6][['PassengerId', 'Name', 'Sex', 'Age']]
df_female = df[df.Sex=='female'].fillna({'Age': female_age_mean})

df_male = df[df.Sex=='male'].fillna({'Age': male_age_mean})



filled_df = df_female.append(df_male)
filled_df[filled_df.PassengerId==6][['PassengerId', 'Name', 'Sex', 'Age']]
def classification_age(age):

    if age <= 19:

        return '1'

    elif age <= 34:

        return '2'

    elif age <= 49:

        return '3'

    elif age >= 50:

        return '4'    

    else:

        return '0'

        

filled_df['AgeClass'] = filled_df.Age.map(classification_age)

filled_df.head()
filled_df['Survived'].plot(alpha=0.6, kind='hist', bins=2)

plt.xlabel('Survived')

plt.ylabel('N')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))



for i, sex in enumerate(['male', 'female']):

    filled_df['Survived'][filled_df.Sex==sex].hist(alpha=0.5, bins=2, ax=axes[i])

    axes[i].set_title(sex)



fig.subplots_adjust(hspace=0.3)

fig.tight_layout()
plt.hist([filled_df[(filled_df.Survived==0) & (filled_df.Sex=='male')]['Age'], filled_df[(filled_df.Survived==1) & (filled_df.Sex=='male')]['Age']],

          alpha=0.6, range=(1,80), bins=10, stacked=True,

          label=('Died', 'Survived'))

plt.legend()

plt.xlabel('Age')

plt.ylabel('N')

plt.title('male')
plt.hist([filled_df[(filled_df.Survived==0) & (filled_df.Sex=='female')]['Age'],

          filled_df[(filled_df.Survived==1) & (filled_df.Sex=='female')]['Age']],

          alpha=0.6, range=(1,80), bins=10, stacked=True,

          label=('Died', 'Survived'))

plt.legend()

plt.xlabel('Age')

plt.ylabel('N')

plt.title('female')
fig = plt.figure(figsize=[15, 5])



ax1 = fig.add_subplot(121)



plt.hist([filled_df[(filled_df.Survived==0) & (filled_df.Sex=='female')]['Age'],

          filled_df[(filled_df.Survived==1) & (filled_df.Sex=='female')]['Age']],

          alpha=0.6, range=(1,80), bins=10, stacked=True,

          label=('Died', 'Survived'))



plt.xlabel('Age')

plt.yticks([0, 40, 80, 120])

plt.ylabel('N')

plt.title('female')

plt.legend()



ax2 = fig.add_subplot(122)



plt.hist([filled_df[(filled_df.Survived==0) & (filled_df.Sex=='male')]['Age'],

          filled_df[(filled_df.Survived==1) & (filled_df.Sex=='male')]['Age']],

          alpha=0.6, range=(1,80), bins=10, stacked=True,

          label=('Died', 'Survived'))



plt.xlabel('Age')

plt.yticks([0, 40, 80, 120])

plt.ylabel('N')

plt.title('male')

plt.legend()



plt.show()
mean_age = df['Age'].mean()



for pclass in [1, 2, 3]:

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[10, 10])



    sex_n=0

    for sex in ['male', 'female']:

        for survived in [0, 1]:

                fig = filled_df[((filled_df.Survived==survived) & (filled_df.Sex==sex) & (filled_df.Pclass==pclass) )].Age.hist(alpha=0.6, bins=10, ax=axes[sex_n][survived])

                fig.set_xlabel("Age")    

                fig.set_ylabel('N ('+sex+str(survived)+' )')  

                axes[sex_n][survived].set_ylim(0,70)

                fig.set_title('Pclass = {0} / mean_age = {1}'.format(pclass, round(mean_age)))

                

        sex_n += 1

    plt.subplots_adjust(hspace=0.5)

    plt.show()


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
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.tail()
df_test.tail()
x = df_train['Sex']

y = df_train['Survived']
y_pred = x.map({'female': 1, 'male': 0}).astype(int)
print('Accuracy: {:.3f}'.format(accuracy_score(y, y_pred)))
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
X_train
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
clf = SVC(kernel='rbf', probability=True)

clf.fit(X_train, y_train)



y_train_pred = clf.predict(X_train)

y_val_pred = clf.predict(X_val)



print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(y_train, y_train_pred)))

print('Accuracy on Validation Set: {:.3f}'.format(accuracy_score(y_val, y_val_pred)))
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=35)



clf = LogisticRegression()

clf.fit(X_train, y_train)



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
clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=10)

scores = cross_val(clf, X, y, 30)

print('Scores:', scores)

print('Mean Score: {0:.3f} (+/-{1:.3f})'.format(scores.mean(), scores.std()*2))
clf = DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=2)



param_grid = {'max_depth': [2, 3, 4, 5], 'min_samples_leaf': [2, 3, 4, 5]}

cv = KFold(5, shuffle=True, random_state=0)



grid_search = GridSearchCV(clf, param_grid, cv=cv, n_jobs=-1, verbose=1,return_train_score=True)

grid_search.fit(X, y)
print('Scores: {:.3f}'.format(grid_search.best_score_))

print('Best Parameter Choice:', grid_search.best_params_)
grid_search.cv_results_
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