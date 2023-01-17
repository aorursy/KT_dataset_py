import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV 
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.info()
class Selector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.feature_names].values
    
embarked_pipeline = Pipeline([
    ('selector', Selector( ['Embarked'] )),
    ('imputer', SimpleImputer( strategy='most_frequent' ))
])
first_letter = train['Cabin'].apply(lambda x: x[0] if not pd.isna(x) else 0)
sns.countplot(x=first_letter, hue=train['Survived'])
class CabinTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['First_Cabin_Letter'] = X['Cabin'].apply(lambda x: x[0] if not pd.isna(x) else 'X')
        return X

cabin_pipeline = Pipeline([
    ('cabin_tr', CabinTransformer()),
    ('selector', Selector(['First_Cabin_Letter']))
])
sns.barplot(x=train['Pclass'], y=train['Age'], hue=train['Sex'])
sns.barplot(x=train['Sex'], y=train['Age'])
sns.barplot(x=train['Parch'], y=train['Age'])
sns.barplot(x=train['SibSp'], y=train['Age'])
ax = sns.barplot(x=train['SibSp'] + train['Parch'] + 1, y=train['Age'])
ax.set(xlabel='Family Size')
sns.countplot(x=train['SibSp'] + train['Parch'] + 1)
male_mean_age = train[train['Sex'] == 'male'].groupby('Pclass').mean()['Age'].values
female_mean_age = train[train['Sex'] == 'female'].groupby('Pclass').mean()['Age'].values
class AgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        for index, _ in X.iterrows():
            if pd.isna(X.loc[index, 'Age']):

                if X.loc[index, 'Sex'] == 'male':
                    if X.loc[index, 'Pclass'] == 1:
                        X.loc[index, 'Age'] = male_mean_age[0]
                    elif X.loc[index, 'Pclass'] == 2:
                        X.loc[index, 'Age'] = male_mean_age[1]
                    else:
                        X.loc[index, 'Age'] = male_mean_age[2]

                else:      
                    if X.loc[index, 'Pclass'] == 1:
                        X.loc[index, 'Age'] = female_mean_age[0]
                    elif X.loc[index, 'Pclass'] == 2:
                        X.loc[index, 'Age'] = female_mean_age[1]
                    else:
                        X.loc[index, 'Age'] = female_mean_age[2]
        return X

age_pipeline = Pipeline([
    ('age_tr', AgeTransformer()),
    ('selector', Selector(['Age']))
])
train['Name'].head(10)
def getTitle(name):
    return name.split(',')[1].split('.')[0].split()[0]

train['Title'] = train['Name'].apply(getTitle)
test['Title'] = test['Name'].apply(getTitle)

train.Title.value_counts()
test.Title.value_counts()
class TitleAttributeAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        titles = X['Name'].apply(getTitle)
        less_titles_count = titles.value_counts()[4:].keys()
        titles = titles.apply(lambda x: 'residue' if x in less_titles_count else x)
        return titles.values.reshape(-1, 1)
    
name_pipeline = TitleAttributeAdder()
sns.countplot(name_pipeline.fit_transform(train).reshape(-1), hue=train['Survived'])
train['Family Size'] = train['SibSp'] + train['Parch'] + 1
train['Family Size'].value_counts()
sns.barplot(x=train['Family Size'], y=train['Survived'])
train['Family Size'] = train['Family Size'].replace(np.arange(5, 12), 'Large')
train['Family Size'] = train['Family Size'].replace([3, 4], 'Medium')
train['Family Size'] = train['Family Size'].replace(2, 'Small')
train['Family Size'] = train['Family Size'].replace(1, 'Alone')
sns.barplot(x=train['Family Size'], y=train['Survived'])
class FamilySize_Attribute_Adder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        family_size = X['SibSp'] + X['Parch'] + 1
        
        family_size = family_size.replace(np.arange(5, 12), 'Large')
        family_size = family_size.replace([3, 4], 'Medium')
        family_size = family_size.replace(2, 'Small')
        family_size = family_size.replace(1, 'Alone')
        
        return family_size.values.reshape(-1, 1)

family_pipeline = FamilySize_Attribute_Adder()
sns.barplot(x='Pclass', y='Survived', data=train)
sns.barplot(x='Sex', y='Survived', data=train)
pcl_sex_pipeline = Selector(['Pclass', 'Sex'])
# Getting 20 random instances of the train dataset
np.random.seed(42)
train.Ticket.loc[np.random.randint(0, len(train), size=20)]
is_numerical = train['Ticket'].apply(lambda x: x.split()[0][0])
is_numerical = is_numerical.replace('1 2 3 4 5 6 7 8 9'.split(), 1)
is_numerical = is_numerical.apply(lambda x: 1 if type(x)==int else 0)
is_numerical.value_counts()
sns.barplot(x=is_numerical, y=train.Survived)
sns.distplot(train.Fare)
train['Fare'] = train['Fare'].apply(np.sqrt)
sns.distplot(train.Fare)
first_quartile = np.quantile(train.Fare, 0.25)
third_quartile = np.quantile(train.Fare, 0.75)
interquartile_amplitude = third_quartile - first_quartile
lower_limit = first_quartile - 1.5 * interquartile_amplitude
higher_limit = third_quartile + 1.5 * interquartile_amplitude
train.loc[(train.Fare > higher_limit), 'Fare'] = higher_limit
sns.distplot(train.Fare)
class FareTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        fare_sqrt = X.Fare.apply(lambda x: np.log(x + 0.5))
        self.mean = np.mean(fare_sqrt)
        first_quartile = np.quantile(fare_sqrt, 0.25)
        third_quartile = np.quantile(fare_sqrt, 0.75)
        interquartile_amplitude = third_quartile - first_quartile
        lower_limit = first_quartile - 1.5 * interquartile_amplitude
        higher_limit = third_quartile + 1.5 * interquartile_amplitude
        
        self.higher_limit = higher_limit
        self.lower_limit = lower_limit
        return self
    def transform(self, X):
        fare_sqrt = X.Fare.apply(lambda x: np.log(x + 0.5))
        fare_sqrt.fillna(self.mean, inplace=True)
        fare_sqrt = fare_sqrt.where(fare_sqrt > self.lower_limit, self.lower_limit)
        fare_sqrt = fare_sqrt.where(fare_sqrt < self.higher_limit, self.higher_limit)
        return fare_sqrt.values.reshape(-1, 1)
    
fare_pipeline = FareTransformer()
numerical_pipeline = FeatureUnion([ 
    ('age_pipe', age_pipeline),
    ('fare_pipe', fare_pipeline)
])

numerical_pipeline = Pipeline([
    ('num_pipe', numerical_pipeline),
    ('scaler', StandardScaler())
])

categorical_pipeline = FeatureUnion([
    ('embarked_pipe', embarked_pipeline),
    ('pcl_sex_pipe', pcl_sex_pipeline),
    ('name_pipe', name_pipeline),
    ('family_pipe', family_pipeline),
    ('cabin_pipe', cabin_pipeline)
])

categorical_pipeline = Pipeline([
    ('cat_pipe', categorical_pipeline),
    ('encoder', OneHotEncoder(drop='first'))
])
prepared_data_pipeline = FeatureUnion([
    ('num_pipe', numerical_pipeline),
    ('cat_pipe', categorical_pipeline)
])
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

X_train = prepared_data_pipeline.fit_transform(train)
y_train = train.Survived
X_test = prepared_data_pipeline.transform(test)
scores = []
param_grid = {'n_estimators': np.arange(10, 100, 5),
              'max_depth': np.arange(3, 8)}

rf_clf_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, verbose=1)
rf_clf_grid.fit(X_train, y_train)

scores.append(('RFC', rf_clf_grid.best_score_))
rf_clf_best = rf_clf_grid.best_estimator_
param_grid = {'n_estimators': np.arange(10, 100, 5),
              'max_depth': np.arange(2, 8)}

et_clf_grid = GridSearchCV(ExtraTreesClassifier(), param_grid, cv=3, verbose=1)
et_clf_grid.fit(X_train, y_train)

scores.append(('ETC', et_clf_grid.best_score_))
et_clf_best = et_clf_grid.best_estimator_
param_grid = {'C': [0.1, 1, 2, 3, 4, 5]}

log_reg_grid = GridSearchCV(LogisticRegression(), param_grid, cv=3, verbose=1)
log_reg_grid.fit(X_train, y_train)

scores.append(("Logistic Regression",log_reg_grid.best_score_))
log_reg_best = log_reg_grid.best_estimator_
param_grid = {'C': np.linspace(1, 3, 11), 
              'kernel': ['rbf', 'poly','linear'], 
              'degree': [2, 3, 4]}

svc_grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, verbose=1)
svc_grid.fit(X_train, y_train)

scores.append(('SVC', svc_grid.best_score_))
svc_best = svc_grid.best_estimator_
param_grid = {'weights': ['uniform', 'distance'], 
              'n_neighbors': np.arange(2, 20)}

kn_clf_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, verbose=1)
kn_clf_grid.fit(X_train, y_train)

scores.append(('KNC', kn_clf_grid.best_score_))
kn_clf_best = kn_clf_grid.best_estimator_
scores
voting_clf = VotingClassifier([
    ('rf_clf', rf_clf_best),
    ('log_reg', log_reg_best),
    ('svc', svc_best),
    ('kn_clf', kn_clf_best)
], voting='soft')

voting_clf.fit(X_train, y_train)
predictions = voting_clf.predict(X_test)
submission = pd.Series(predictions, index=test.PassengerId, name='Survived')
submission.to_csv('titanic_submission.csv', header=True)