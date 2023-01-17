%matplotlib inline

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style = "whitegrid", color_codes = True)

np.random.seed(sum(map(ord, "palettes")))



from sklearn.metrics import roc_auc_score



#Models

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier



from sklearn.model_selection import cross_val_score,train_test_split

from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import KFold

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.describe()
test.describe()
train.info()

print("++++++++++++++++++++++++++++++++++++++")

print()

test.info()
train.head()
test.head()
train.describe(include = ['O'])
test.describe(include = ['O'])
test_PassengerId = test["PassengerId"]  # save the id for submiting the final results



train.drop(['PassengerId', "Ticket", 'Cabin'], axis = 1, inplace = True)

test.drop(['PassengerId', "Ticket", 'Cabin'], axis=1, inplace = True)

train_test_data = [train, test] 
for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map({'male': 1, 'female': 0}).astype(int)
train_test_data[0].head() # train data
train_test_data[1].head()  # test data
train[['Sex', 'Survived']].groupby(['Sex'], 

                                        as_index = False).mean().sort_values(by = 'Survived', ascending = True)
train[['Pclass', 'Survived']].groupby(['Pclass'], 

                                        as_index = False).mean().sort_values(by = 'Survived', ascending = True)
age_fill = np.zeros((2,3)) # 2 for sex and 3 for Pclass

print(age_fill)

age_fill = np.zeros((2,3)) 

for dataset in train_test_data:

    for s in range(0, 2):

        for p in range(0, 3):

            age_fill_df = dataset[(dataset['Sex'] == s) &\

                               (dataset['Pclass'] == p + 1)]['Age'].dropna()

            age_to_fill = age_fill_df.median()



            # Convert random age float to nearest .5 age

            age_fill[s,p] = int( age_to_fill/0.5 + 0.5 ) * 0.5

            

    for s in range(0, 2):

        for p in range(0, 3):

            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == s) & (dataset.Pclass == p + 1),\

                    'Age'] = age_fill[s,p]



    dataset['Age'] = dataset['Age'].astype(int)



train.head()
test.head()
min(train['Age']), max(train['Age'])
train['AgeBins'] = pd.cut(train['Age'], 8)
train[['AgeBins', 'Survived']].groupby(['AgeBins'], 

                                       as_index = False).mean().sort_values(by = 'Survived', ascending = True)
for dataset in train_test_data:    

    dataset.loc[dataset['Age'] <= 10, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 20), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 60) & (dataset['Age'] <= 70), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 70, 'Age'] = 7
fig = sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train)

fig.get_axes().set_xticklabels(["Female", "Male"])

fig.get_axes().legend(["First Class", "Second Class", "Third Class"], 

                    loc='upper right');
fig = sns.pointplot(x="Sex", y="Survived", hue="Pclass", data=train);



fig.get_axes().set_xlabel("Sex")

fig.get_axes().set_xticklabels(["Female", "Male"])

fig.get_axes().set_ylabel("Mean(Survived")

fig.get_axes().legend(["First Class", "Second Class", "Third Class"], 

                    loc='upper left')
sns.countplot(x="AgeBins", data = train, palette = "GnBu_d");
sns.countplot( x ="AgeBins", hue="Pclass", data = train, palette="PuBuGn_d");
train.head()
train = train.drop(['AgeBins'], axis = 1)

train_test_data = [train, test]

train.head()
for dataset in train_test_data:

    dataset["FamilySize"] = dataset['SibSp'] + dataset['Parch']

train, test = train_test_data[0], train_test_data[1]

train.head()
train[['FamilySize', 'Survived']].groupby(['FamilySize'], 

                                        as_index = False).mean().sort_values(by = 'Survived', ascending = False)
sns.countplot(x="FamilySize", data = train, palette = "GnBu_d");
train = train.drop(['Parch', 'SibSp'], axis = 1)

test = test.drop(['Parch', 'SibSp'], axis = 1)

train_test_data = [train, test]

train.head()
test.head()
Embarking_freq = train.Embarked.dropna().mode()[0]

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna(Embarking_freq)

train, test = train_test_data[0], train_test_data[1]

train.head()   
train[['Embarked', 'Survived']].groupby(['Embarked'], 

                                       as_index = False).mean().sort_values(by = 'Survived', ascending = False)
for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

train.head()
Fare_freq = test.Fare.dropna().mode()[0]

for dataset in train_test_data:

    dataset['Fare'] = dataset['Fare'].fillna(Fare_freq)
train['FareBins'] = pd.qcut(train['Fare'], 5)

train[['FareBins', 'Survived']].groupby(['FareBins'], 

                                        as_index = False).mean().sort_values(by = 'Survived', ascending = True)
for dataset in train_test_data:    

    dataset.loc[dataset['Fare']  <=7.854, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.84)   & (dataset['Fare'] <= 10.5), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 10.5)   & (dataset['Fare'] <= 21.679), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3

    dataset.loc[(dataset['Fare'] > 39.688) & (dataset['Fare'] <= 5512.329), 'Fare'] = 4
train, test = train_test_data[0], train_test_data[1]

train = train.drop(['FareBins'], axis = 1)

train.head(6)
test.head(6)
def extract_title(df):

    # the Name feature includes last name, title, and first name. After splitting 

    # the title is in the second column or at index 1

    df["Title"] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip()) 

    return df

train = extract_title(train)

test = extract_title(test)
fig = sns.countplot(x = 'Title', data = train, palette = "GnBu_d")

fig = plt.setp(fig.get_xticklabels(), rotation = 45)
#for dset in train:

train_test_data = [train, test]

for dset in train_test_data:

    dset["Title"] = dset["Title"].replace(["Melkebeke", "Countess", "Capt", "the Countess", "Col", "Don",

                                         "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"] , "Lamped")

    dset["Title"] = dset["Title"].replace(["Lady", "Mlle", "Ms", "Mme"] , "Miss")

fig2 = sns.countplot(x = 'Title', data = train, palette = "GnBu_d")

fig2 = plt.setp(fig2.get_xticklabels(), rotation = 45)
train[['Title', 'Survived']].groupby(['Title'], 

                                        as_index = False).mean().sort_values(by = 'Survived', ascending = False)
for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 

                                             'Master': 4, 'Lamped': 5}).astype(int)

train.head()
train.drop(['Name'], axis = 1, inplace = True)

test.drop(['Name'], axis=1, inplace = True)

train.head()
colormap = plt.cm.viridis

plt.figure(figsize=(16,16))

plt.title('Correlation between Features', y=1.05, size = 20)

sns.heatmap(train.corr(),

            linewidths=0.1, 

            vmax=1.0, 

            square=True, 

            cmap=colormap, 

            linecolor='white', 

            annot=True)
y_train = train["Survived"]

X_train = train.drop(["Survived"], axis = 1 )



X_test = test

X_train.shape, y_train.shape, X_test.shape
LR = LogisticRegression(random_state = 0)

LR.fit(X_train, y_train)

y_pred_lr = LR.predict(X_test)

LR_score = LR.score(X_train, y_train)

print("LR Accuracy  score = {:.2f}".format(LR_score*100))
svc = SVC(random_state = 0)

svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

SVC_score = svc.score(X_train, y_train)

print("SVC Accuracy  score = {:.2f}".format(SVC_score*100))

KNN = KNeighborsClassifier(n_neighbors = 5)

KNN.fit(X_train, y_train)

y_pred_knn = KNN.predict(X_test)

KNN_score = KNN.score(X_train, y_train)

print("KNN accuracy score = {:.2f}".format(KNN_score*100))

GNB = GaussianNB()

GNB.fit(X_train, y_train)

y_pred_gnb = GNB.predict(X_test)

GNB_score = GNB.score(X_train, y_train)

print("GNB accuracy score = {:.2f}".format(GNB_score*100))
LSVC = LinearSVC()

LSVC.fit(X_train, y_train)

y_pred_lsvc = LSVC.predict(X_test)

LSVC_score = LSVC.score(X_train, y_train)

print("GNB accuracy score = {:.2f}".format(LSVC_score*100))
perceptron = Perceptron()

perceptron.fit(X_train, y_train)

y_pred_perceptron = perceptron.predict(X_test)

perceptron_score = perceptron.score(X_train, y_train)

print("perceptron accuracy score = {:.2f}".format(perceptron_score*100))

SGD = SGDClassifier()

SGD.fit(X_train, y_train)

y_pred_sgd = SGD.predict(X_test)

SGD_score = SGD.score(X_train, y_train)

print("Stochastic Gradient Descent accuracy score = {:.2f}".format(SGD_score*100))
DT = DecisionTreeClassifier()

DT.fit(X_train, y_train)

y_pred_dt = DT.predict(X_test)

DT_score = DT.score(X_train, y_train)

print("Decision Tree accuracy score = {:.2f}".format(DT_score*100))
RF = RandomForestRegressor(n_estimators = 1000)

RF.fit(X_train, y_train)

y_pred_rf = RF.predict(X_test)

RF_score = RF.score(X_train, y_train)

print("Random forest regressor accuracy score = {:.2f}".format(RF_score*100))

Predictive_models = pd.DataFrame({

    'Model': ['SVM', 'KNN', 'LR', 'RF', 'GNB', 

              'Perceptron','SGD', 'LSVC', 'DT'],

    'Score': [SVC_score, KNN_score, LR_score, RF_score, GNB_score, 

              perceptron_score, SGD_score, LSVC_score, DT_score]})

Predictive_models.sort_values(by ='Score', ascending=True)
DT = DecisionTreeClassifier()

y_train = train.loc[:,"Survived"]

X_train = train.drop(["Survived"], axis = 1)





kfold = KFold(n=len(train), n_folds = 5, shuffle = True, random_state = 0)

kfold_score = cross_val_score(DT, X_train, y_train, cv = kfold)



kfold_score_mean = np.mean(kfold_score)

  

print("Decision Tree accuracy score per fold : ", kfold_score, "\n")

print("Average accuracy score : {:.4f}".format(kfold_score_mean))

DT = DecisionTreeClassifier()

y_train = train.loc[:,"Survived"]

X_train = train.drop(["Survived"], axis = 1)



parameters = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],

             'max_features' : ["auto", None, "sqrt", "log2"],

             'random_state': [0, 25, 75, 125, 250],

             'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

clf = GridSearchCV(DT, parameters)

clf.fit(X_train, y_train)



DT_Model = clf.best_estimator_

print (clf.best_score_, clf.best_params_)

print(DT_Model)

DT = DecisionTreeClassifier(max_depth = 6, 

                            max_features = 'auto', 

                            min_samples_leaf = 2,

                            random_state = 125)

X_train = train.drop(["Survived"], axis = 1)

y_train = train.loc[:,"Survived"]



kfold = KFold(n=len(train), n_folds = 5, shuffle = True, random_state = 125)

DT.fit(X_train, y_train)

kfold_score = cross_val_score(DT, X_train, y_train, cv = kfold)

kfold_score_mean = np.mean(kfold_score)



y_pred_dt = DT.predict(X_test)



  

print("Decision Tree accuracy score per fold : ", kfold_score, "\n")

print("Average accuracy score : {:.4f}".format(kfold_score_mean)) 
DT.feature_importances_
feature_importances = pd.Series(DT.feature_importances_, index = X_train.columns).sort_values()

#feature_importances.sort()

feature_importances.plot(kind = "barh", figsize = (7,6));

plt.title(" feature ranking", fontsize = 20)

plt.show()

Titanic_submission = pd.DataFrame({

        "PassengerId": test_PassengerId,

        "Survived": y_pred_dt

    })
Titanic_submission.to_csv("Titanic_compet_submit.csv", index = False)
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cross_validation import cross_val_score

from sklearn.pipeline import Pipeline

pipe_svm = Pipeline([('scl', StandardScaler()),

            ('pca', PCA(n_components=3)),

            ('clf', SVC(random_state=0))])

scores = cross_val_score(estimator=pipe_svm, 

                          X=X_train, y=y_train, 

                          cv=10, n_jobs=-1)

print('CV accuracy scores: %s' % scores)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 
from sklearn.learning_curve import learning_curve

pipe_svm = Pipeline([('scl', StandardScaler()),            

                     ('pca', PCA(n_components=3)),

                    ('clf', SVC(random_state = 0))])

train_sizes, train_scores, valid_scores = learning_curve(estimator=pipe_svm, 

                       X=X_train, 

                       y=y_train, 

                       train_sizes=np.linspace(0.1, 1.0, 10), 

                       cv=10,

                       n_jobs=1)

train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)

valid_mean = np.mean(valid_scores, axis=1)

valid_std = np.std(valid_scores, axis=1)

plt.plot(train_sizes, train_mean, 

          color='blue', marker='o', 

          markersize=5, 

          label='training accuracy')

plt.fill_between(train_sizes, 

                  train_mean + train_std,

                  train_mean - train_std, 

                  alpha=0.15, color='blue')

plt.plot(train_sizes, valid_mean, 

          color='green', linestyle='--', 

          marker='s', markersize=5, 

          label='validation accuracy')

plt.fill_between(train_sizes, 

                  valid_mean + valid_std,

                  valid_mean - valid_std, 

                  alpha=0.15, color='green')

plt.grid()

plt.xlabel('Number of training samples')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')

plt.ylim([0.65, 0.9])

plt.show()
from sklearn.learning_curve import validation_curve

pipe_svm = Pipeline([('scl', StandardScaler()),            

#                    ('pca', PCA(n_components=3)),

                    ('clf', SVC(random_state = 0))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 1000.0, 10000.0]

train_scores, vald_scores = validation_curve(

                estimator=pipe_svm, 

                 X=X_train, 

                 y=y_train, 

                 param_name='clf__C', 

                 param_range=param_range,

                 cv=10)

train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)

valid_mean = np.mean(valid_scores, axis=1)

valid_std = np.std(valid_scores, axis=1)

plt.plot(param_range, train_mean, 

          color='blue', marker='o', 

          markersize=5, 

          label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,

                  train_mean - train_std, alpha=0.15,

                  color='blue')

plt.plot(param_range, valid_mean, 

          color='green', linestyle='--', 

          marker='s', markersize=5, 

          label='validation accuracy')

plt.fill_between(param_range, 

                  valid_mean + valid_std,

                  valid_mean - valid_std, 

                  alpha=0.15, color='green')

plt.grid()

plt.xscale('log')

plt.legend(loc='lower right')

plt.xlabel('Parameter C')

plt.ylabel('Accuracy')

plt.ylim([0.5, 0.9])

plt.show()
pipe_svm = Pipeline([('scl', StandardScaler()),

#                     ('pca', PCA(n_components = 2)),

                      ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range, 

                'clf__kernel': ['linear']},

               {'clf__C': param_range, 

                'clf__gamma': param_range, 

                'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svm, 

                   param_grid=param_grid, 

                   scoring='accuracy', 

                   cv=10,

                   n_jobs=-1)

clf = gs.fit(X_train, y_train)

print(clf.best_score_) 

print(clf.best_params_)
y_pred_pip = clf.predict(X_test)

print('y_pred_pip: {:.3f}',format(y_pred_pip),"\n")
gs = GridSearchCV(estimator=pipe_svm, 

                   param_grid=param_grid,

                   scoring='accuracy', 

                   cv=2, 

                   n_jobs=-1)

scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)

print('CV accuracy scores: {:.3f}',format(scores))

print('CV accuracy: {:.3f} +/- {:.3f}',format((np.mean(scores), np.std(scores))))
Titanic_submission = pd.DataFrame({

        "PassengerId": test_PassengerId,

        "Survived": y_pred_pip

    })
Titanic_submission.to_csv("Titanic_compet_submit_3.csv", index = False)