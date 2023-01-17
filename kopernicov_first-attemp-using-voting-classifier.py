import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.options.display.max_columns = 100



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import PolynomialFeatures



from sklearn.decomposition import PCA

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

import lightgbm as lgb

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression



import numpy as np

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



full_data= [train, test]



# Store our passenger ID for easy access

PassengerId = test['PassengerId']
print('The data has {} rows and {} columns.'.format(train.shape[0], train.shape[1]))

print('The data has {} rows and {} columns.'.format(test.shape[0], test.shape[1]))

print('\n------------------Information-------------------\n')

train.info()

train.describe()
def show_missing_values(train, test):

    train_null = train.isnull().sum()

    test_null = test.isnull().sum()

    total = pd.concat([train_null, test_null], axis=1)

    total.columns = ['Train', 'Test']

    print(total)
show_missing_values(train, test)
# Fill the nan values with the average of the age

test['Fare'] = test.groupby('Pclass').Fare.apply(lambda x: x.fillna(x.mean()))



# Fill the missing values for embarked with mode

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train['Cabin'].unique()
for dataset in full_data:

    dataset['Cabin'] = dataset['Cabin'].str[0]

    dataset['Cabin'] = dataset['Cabin'].fillna('U') # There is a lot of missing values so we create a new category for this missings.

    dataset.loc[dataset.Cabin=='G','Cabin']= 'U'

    dataset.loc[dataset.Cabin=='T','Cabin']= 'U'





sns.countplot(x='Cabin', hue='Survived', data=train)
show_missing_values(train, test)
for dataset in full_data:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)

    

train['Title'].value_counts()
for dataset in full_data: 

    dataset['TitleCat'] = train['Title']

    dataset.TitleCat.replace(to_replace=['Rev','Dr','Col','Major','Mlle','Ms','Countess','Capt','Dona','Don','Sir','Lady','Jonkheer','Mme'],

                            value=0, inplace=True)

    dataset.TitleCat.replace('Mr', 1, inplace=True)

    dataset.TitleCat.replace('Miss', 2, inplace=True)

    dataset.TitleCat.replace('Mrs', 3, inplace=True)

    dataset.TitleCat.replace('Master', 4, inplace=True)

    dataset.TitleCat = dataset.TitleCat.astype('category')

    dataset.TitleCat.value_counts(dropna=False)
train.loc[train.Age.isnull(), ['Age', 'TitleCat']].head()
print(train.groupby(['TitleCat']).median()['Age'].round(2))



for dataset in full_data:

    dataset['Age'] = dataset.groupby('TitleCat').Age.apply(lambda x: x.fillna(x.median()))

    

train['Age'].iloc[[5,17, 19, 26, 28]]
# from the information we can eliminate the ticket and name column because, in principle, we think this information doesn't matter

# aswell with PassengerId and title



for dataset in full_data:

    dataset.drop(['Name', 'Ticket', 'PassengerId', 'Title'], axis=1, inplace=True)
pd.qcut(train.Age, 5).unique()
pd.qcut(train.Fare, 4).unique()
for dataset in full_data:

    dataset.loc[dataset['Fare'] <= 7.91,'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 512.329), 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype('category')

    

    dataset.Pclass = dataset.Pclass.astype('category')



    # Mapping Age



    dataset.loc[dataset['Age'] <= 20, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 26), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 30.0), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 30.0) & (dataset['Age'] <= 38), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 38) & (dataset['Age'] <= 80), 'Age'] = 4

    dataset['Age'] = dataset['Age'].astype('category')



    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



    dataset.drop(['SibSp', 'Parch'], axis=1, inplace=True)



show_missing_values(train, test)
train_dummies = pd.get_dummies(train)

X_test = pd.get_dummies(test)

y_train = train_dummies['Survived'].astype(int)

X_train = train_dummies.drop(['Survived'], axis=1)



X_test.head()
params_KNN = {'n_neighbors': np.arange(5, 13),

             'metric': ['euclidean', 'cityblock']}



clf = KNeighborsClassifier()

grid = GridSearchCV(clf, param_grid=params_KNN, cv=10, verbose=False)

grid.fit(X_train, y_train)

print('El accuracy es {}'.format(grid.best_score_))

print('Los mejores parámetros son {}'.format(grid.best_params_))
params_SVC = {'C': [1.35, 1.40]}



clf = SVC()

grid = GridSearchCV(clf, param_grid=params_SVC, cv=10, verbose=False)

grid.fit(X_train, y_train)

print('El accuracy es {}'.format(grid.best_score_))

print('Los mejores parámetros son {}'.format(grid.best_params_))
params_LG = {'penalty': ['l1', 'l2'],

             'C': [0.40, 0.45]}



clf = LogisticRegression()

grid = GridSearchCV(clf, param_grid=params_LG, cv=10, verbose=False)

grid.fit(X_train, y_train)

print('El accuracy es {}'.format(grid.best_score_))

print('Los mejores parámetros son {}'.format(grid.best_params_))
# params_RF = {'n_estimators': [120, 130],

#              'max_depth': [12,14],

#             'min_samples_split': [3,4],

#             'min_samples_leaf': [5],

#             'max_features': ['sqrt']}



forrest_params = dict(     

    max_depth = [n for n in range(9, 14)],     

    min_samples_split = [n for n in range(4, 11)], 

    min_samples_leaf = [n for n in range(2, 5)],     

    n_estimators = [n for n in range(50, 130, 5)],

)





clf = RandomForestClassifier()

grid = GridSearchCV(clf, param_grid=forrest_params, cv=3, verbose=True, n_jobs=-1)

grid.fit(X_train, y_train)

print('El accuracy es {}'.format(grid.best_score_))

print('Los mejores parámetros son {}'.format(grid.best_params_))
estimator = [('KNN' , KNeighborsClassifier(metric='cityblock', n_neighbors=9)),

             ('RF', RandomForestClassifier(max_depth=10, max_features='sqrt', min_samples_leaf= 2, min_samples_split=7, n_estimators= 90)),

             ('lr', LogisticRegression(C=0.4, penalty='l2')),

             ('SVC', SVC(C=1.35, probability=True))

             ]



eclf = VotingClassifier(estimators=estimator, voting='soft')

cv_scores = cross_val_score(eclf, X_train, y_train, cv=10, verbose=False)

eclf.fit(X_train, y_train)

print('El accuracy es: ', cv_scores.mean())
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    """

    Generate a simple plot of the test and training learning curve.



    Parameters

    ----------

    estimator : object type that implements the "fit" and "predict" methods

        An object of that type which is cloned for each validation.



    title : string

        Title for the chart.



    X : array-like, shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples and

        n_features is the number of features.



    y : array-like, shape (n_samples) or (n_samples, n_features), optional

        Target relative to X for classification or regression;

        None for unsupervised learning.



    ylim : tuple, shape (ymin, ymax), optional

        Defines minimum and maximum yvalues plotted.



    cv : int, cross-validation generator or an iterable, optional

        Determines the cross-validation splitting strategy.

        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,

          - integer, to specify the number of folds.

          - An object to be used as a cross-validation generator.

          - An iterable yielding train/test splits.



        For integer/None inputs, if ``y`` is binary or multiclass,

        :class:`StratifiedKFold` used. If the estimator is not a classifier

        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.



        Refer :ref:`User Guide <cross_validation>` for the various

        cross-validators that can be used here.



    n_jobs : integer, optional

        Number of jobs to run in parallel (default 1).

    """

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)



plot_learning_curve(eclf, 'Learning curve', X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=2)
predict = eclf.predict(X_test)

df_submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predict})

# df_submission.to_csv('../input/prediction.csv', index=False)
df_submission.head(10)