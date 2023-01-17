import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore



warnings.filterwarnings("ignore")

plt.style.use('ggplot')

sns.set(style="ticks", context = 'talk', palette = 'bright', rc={'figure.figsize':(11,8.27)})
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gen = pd.read_csv('../input/titanic/gender_submission.csv')
train.info()
train.describe()
train.head()
print("Unique values in PassengerId Column:", len(train['PassengerId'].unique()))

print("Unique values in Name Column:", len(train['Name'].unique()))

print("Unique values in Ticket Column:", len(train['Ticket'].unique()))
train.isnull().sum().sort_values(ascending=False)
train['Survived'].value_counts(normalize=True).plot(kind = 'bar')
sns.countplot(x='Sex', hue='Survived', data=train)

sns.catplot(x="Sex", y="Age", hue="Survived", data=train, height=9)
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train[train['Sex']=='female']

men = train[train['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
sns.countplot(x='Embarked', hue='Survived', data=train)
emb = train.groupby(['Embarked', 'Survived']).size()

emb_pct = emb.groupby(level=0).apply(lambda x: 100*x/float(x.sum()))

emb_pct.to_frame().unstack().plot(kind='bar', stacked=True)

plt.ylabel('Survival Chance in %')

current_handles, _ = plt.gca().get_legend_handles_labels()

reversed_handles = reversed(current_handles)



labels = reversed(train['Survived'].unique())



plt.legend(reversed_handles,labels,loc='lower right')

plt.show()
sns.countplot(x='Pclass', hue="Survived", data=train)
pc = train.groupby(['Pclass', 'Survived']).size()

pc_pct = pc.groupby(level=0).apply(lambda x: 100*x/float(x.sum()))

pc_pct.to_frame().unstack().plot(kind='bar', stacked=True)

plt.ylabel('Survival Chance in %')

current_handles, _ = plt.gca().get_legend_handles_labels()

reversed_handles = reversed(current_handles)



labels = reversed(train['Survived'].unique())



plt.legend(reversed_handles,labels,loc='lower right')

plt.show()
g = sns.FacetGrid(train, row='Embarked', size = 7)

g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')

g.add_legend()
train['Family'] = train['SibSp'] + train['Parch']

train['Alone'] = 0

train['Alone'] = np.where(train['Family']>0, 0, 1)

train['Alone'].value_counts()
test['Family'] = test['SibSp'] + test['Parch']

test['Alone'] = 0

test['Alone'] = np.where(test['Family']>0, 0, 1)

test['Alone'].value_counts()
ax = pd.crosstab(train['Family'], train['Survived']).apply(lambda row: row/row.sum(), axis=1).plot(kind='bar')

for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.yticks([])



# Add this loop to add the annotations

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.0%}'.format(height), (x, y + height + 0.01))
sns.catplot('Survived', 'Fare', data = train)
sns.catplot('Pclass', 'Fare', data = train)
train['Fare_range'] = pd.qcut(train['Fare'], 4)

sns.barplot(x='Fare_range', y='Survived', data = train)
train.isna().sum()
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train['Age'].fillna(round(train['Age'].mean(), 1), inplace=True)
train.drop(['Cabin', 'PassengerId', 'Ticket', 'Name', 'Fare_range', 'SibSp', 'Parch'], axis = 1, inplace = True)
test['Age'].fillna(round(test['Age'].mean(), 1), inplace=True)

test['Fare'].fillna(round(test['Fare'].mean(), 1), inplace=True)

test.drop(['Cabin', 'PassengerId', 'Ticket', 'Name', 'SibSp', 'Parch'], axis = 1, inplace = True)
test.isna().sum()
train['Sex'].replace('male', 1, inplace=True)

train['Sex'].replace('female', 0, inplace=True)



train['Embarked'].replace('S', 0, inplace=True)

train['Embarked'].replace('Q', 1, inplace=True)

train['Embarked'].replace('C', 2, inplace=True)
test['Sex'].replace('male', 1, inplace=True)

test['Sex'].replace('female', 0, inplace=True)



test['Embarked'].replace('S', 0, inplace=True)

test['Embarked'].replace('Q', 1, inplace=True)

test['Embarked'].replace('C', 2, inplace=True)
print('\nTrain')

print(train.dtypes)

print('\nTest')

print(test.dtypes)
X = train.drop('Survived', axis = 1)

Y = train['Survived']
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print('Training set {}, {}'.format(X_train.shape, y_train.shape))

print('Cross Validation set {}, {}'.format(X_cv.shape, y_cv.shape))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_cv = sc.transform(X_cv)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

import keras.backend as K



from sklearn.model_selection import GridSearchCV, validation_curve

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
def plot_confusion_matrix(y_true, y_pred):

    cf_matrix = confusion_matrix(y_true, y_pred)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in

              zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
report = pd.DataFrame(columns = ['Models', 'Train Accuracy', 'CV Accuracy', 'CV F1-Score']) 
model = LogisticRegression(random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_cv)



report = report.append({'Models': 'Logistic Regression', 

                        'Train Accuracy': round(model.score(X_train, y_train)*100, 2), 'CV Accuracy': round(model.score(X_cv, y_cv)*100, 2), 

                        'CV F1-Score': f1_score(y_cv, y_pred)}, ignore_index=True)
plot_confusion_matrix(y_cv, y_pred)
model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

model.fit(X_train, y_train)

y_pred = model.predict(X_cv)



report = report.append({'Models': 'KNN',  

                        'Train Accuracy': round(model.score(X_train, y_train)*100, 2), 'CV Accuracy': round(model.score(X_cv, y_cv)*100, 2), 

                        'CV F1-Score': f1_score(y_cv, y_pred)}, ignore_index=True)
plot_confusion_matrix(y_cv, y_pred)
model = SVC(kernel='linear', random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_cv)



report = report.append({'Models': 'SVC Linear',  

                        'Train Accuracy': round(model.score(X_train, y_train)*100, 2), 'CV Accuracy': round(model.score(X_cv, y_cv)*100, 2), 

                        'CV F1-Score': f1_score(y_cv, y_pred)}, ignore_index=True)
plot_confusion_matrix(y_cv, y_pred)
model = SVC(kernel='rbf', random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_cv)



report = report.append({'Models': 'SVC RBF',  

                        'Train Accuracy': round(model.score(X_train, y_train)*100, 2), 'CV Accuracy': round(model.score(X_cv, y_cv)*100, 2), 

                        'CV F1-Score': f1_score(y_cv, y_pred)}, ignore_index=True)
plot_confusion_matrix(y_cv, y_pred)
model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_cv)



report = report.append({'Models': 'Gaussian Naive Bayes',  

                        'Train Accuracy': round(model.score(X_train, y_train)*100, 2), 'CV Accuracy': round(model.score(X_cv, y_cv)*100, 2), 

                        'CV F1-Score': f1_score(y_cv, y_pred)}, ignore_index=True)
plot_confusion_matrix(y_cv, y_pred)
model = DecisionTreeClassifier(criterion='entropy', random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_cv)



report = report.append({'Models': 'Decision Tree',  

                        'Train Accuracy': round(model.score(X_train, y_train)*100, 2), 'CV Accuracy': round(model.score(X_cv, y_cv)*100, 2), 

                        'CV F1-Score': f1_score(y_cv, y_pred)}, ignore_index=True)
plot_confusion_matrix(y_cv, y_pred)
model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_cv)



report = report.append({'Models': 'Random Forests',  

                        'Train Accuracy': round(model.score(X_train, y_train)*100, 2), 'CV Accuracy': round(model.score(X_cv, y_cv)*100, 2), 

                        'CV F1-Score': f1_score(y_cv, y_pred)}, ignore_index=True)
plot_confusion_matrix(y_cv, y_pred)
model = Sequential()

model.add(Dense(12, input_dim=7, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=150, batch_size = 10, verbose=0)



_, train_acc = model.evaluate(X_train, y_train)

_, cv_acc = model.evaluate(X_cv, y_cv)



y_pred = model.predict_classes(X_cv)

y_pred.reshape(len(y_pred),)



report = report.append({'Models': 'Neural Network',  

                        'Train Accuracy': round(train_acc*100, 2), 'CV Accuracy': round(cv_acc*100, 2), 

                        'CV F1-Score': f1_score(y_cv, y_pred)}, ignore_index=True)
plot_confusion_matrix(y_cv, y_pred)
report.sort_values('CV F1-Score', ascending=False)
model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

model.fit(X, Y)

y_pred = model.predict(test)
submission = pd.read_csv('../input/titanic/test.csv')
submission = pd.concat([submission['PassengerId'], pd.DataFrame(y_pred)], axis=1)

submission.columns = ['PassengerId', 'Survived']
submission
submission.to_csv('submission.csv', index=False)
model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

model.fit(X, Y)

y_pred = model.predict(test)
submission = pd.concat([submission['PassengerId'], pd.DataFrame(y_pred)], axis=1)

submission.columns = ['PassengerId', 'Survived']

submission.to_csv('submission.csv', index=False)
model = Sequential()

model.add(Dense(12, input_dim=7, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size = 10, verbose=0)



y_pred = model.predict_classes(test)

y_pred.reshape(len(y_pred),)
submission = pd.concat([submission['PassengerId'], pd.DataFrame(y_pred)], axis=1)

submission.columns = ['PassengerId', 'Survived']

submission.to_csv('submission.csv', index=False)