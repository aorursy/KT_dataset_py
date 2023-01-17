import os

dataset_path = os.path.join('/kaggle/input', 'titanic')

gender = os.path.join(dataset_path, 'gender_submission.csv')

test = os.path.join(dataset_path, 'test.csv')

train = os.path.join(dataset_path, 'train.csv')
import pandas as pd

test_set = pd.read_csv(test)

train_set = pd.read_csv(train)

gender_set = pd.read_csv(gender)

train_set
training = train_set

#training = train_set.drop('Survived', axis=1)

#training_labels = train_set['Survived'].copy()
training_copy = training.copy()
training.drop(['Name', 'Ticket'], axis=1, inplace=True)
training
trainig_cat_encoded, training_categories = training['Sex'].factorize()
training['Sex'] = trainig_cat_encoded

trainig_cat_encoded, training_categories = training['Embarked'].factorize()

training['Embarked'] = trainig_cat_encoded

training
corr_matrix = training.corr()

corr_matrix['Survived'].sort_values(ascending=False)
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']

cabins = []



def substr(substring, letters):

    if isinstance(substring, str):

        for it in letters:

            if substring.find(it) != -1:

                return it

        return 'Nan'

    else:

        return 'Nan'



training['Cabin'] = training['Cabin'].map(lambda x: substr(x, cabin_list))

        
cabin_encode_cat, cabin_categories = training['Cabin'].factorize()

training['Cabin'] = cabin_encode_cat
corr_matrix = training.corr()

corr_matrix['Survived'].sort_values(ascending=False)
training['Sex/class'] = training['Sex'] / training['Pclass']
training.drop('PassengerId', axis=1, inplace=True)
training_new = training.drop('Survived', axis=1)

training_labels = training['Survived'].copy()
sex = training_new['Sex']

new_sex = [sex+1 for sex in training_new['Sex']]

training_new['Sex'] = new_sex
training_new['Sex/class'] = training_new['Sex'] / training_new['Pclass']
training_new['Survived'] = training['Survived']

corr_matrix = training_new.corr()

corr_matrix['Survived'].sort_values(ascending=False)
training_new.drop('Sex', axis=1, inplace=True)
training_new.drop('Survived', axis=1, inplace=True)
training_new
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attributes_names):

        self.attributes_names = attributes_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attributes_names].values



pipeline_categories = Pipeline([

    ('selector', DataFrameSelector(['Cabin', 'Embarked', 'Pclass'])),

    ('onehotencoder', OneHotEncoder(categories='auto'))

])



pipeline_numeric = Pipeline([

    ('selector', DataFrameSelector(['Age', 'SibSp', 'Parch', 'Fare', 'Sex/class'])),

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])



preprocessing_pipeline = FeatureUnion(transformer_list=[

    ('num_pipe', pipeline_numeric),

    ('cat_pipe', pipeline_categories)

])
training_prepared = preprocessing_pipeline.fit_transform(training_new)
from sklearn.model_selection import GridSearchCV

param_grid = [

    { 'criterion': ['gini', 'entropy'], 'n_estimators': [50, 100, 300, 600, 800, 1000], 'max_features': [2, 3, 4, 5, 6], 'max_depth': [2, 4, 6, 8, 10] }

]



from sklearn.ensemble import RandomForestClassifier



tree_class = RandomForestClassifier()

grid_search = GridSearchCV(tree_class, param_grid, cv=5, verbose=3, n_jobs=-1)

grid_search.fit(training_prepared, training_labels)
final_model = grid_search.best_estimator_
grid_search.best_params_
from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score

predictions = final_model.predict(training_prepared)

accuracy_score(predictions, training_labels)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

model_score = cross_val_score(final_model, training_prepared, training_labels, cv=10)

model_score.mean()
test_set.drop('Ticket', axis=1, inplace=True)
test_set
test_cat_encoded, test_categories = test_set['Sex'].factorize()

test_set['Sex'] = test_cat_encoded

test_cat_encoded, test_categories = test_set['Embarked'].factorize()

test_set['Embarked'] = test_cat_encoded

test_set['Sex'] = test_set['Sex'] + 1

test_set['Sex/class'] = test_set['Sex'] / test_set['Pclass']

test_set.drop('Sex', axis=1, inplace=True)
passengerId = test_set['PassengerId'].copy()
test_set.drop(['Name', 'PassengerId'], axis=1, inplace=True)
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']

cabins = []



def substr(substring, letters):

    if isinstance(substring, str):

        for it in letters:

            if substring.find(it) != -1:

                return it

        return 'Nan'

    else:

        return 'Nan'



test_set['Cabin'] = test_set['Cabin'].map(lambda x: substr(x, cabin_list))

        
cabin_encode_cat, cabin_categories = test_set['Cabin'].factorize()

test_set['Cabin'] = cabin_encode_cat
test_set
test_prepared = preprocessing_pipeline.transform(test_set)
predictions_test = final_model.predict(test_prepared)
predictions_test
file_output = open('output.csv', 'w')

file_output.write('PassengerId,Survived\n')

for passId, surv in zip(passengerId, predictions_test):

    file_output.write("{0},{1}\n".format(passId, surv))



file_output.close()