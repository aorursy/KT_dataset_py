# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

import numpy as np

from sklearn.preprocessing import OneHotEncoder

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import reciprocal, expon

from sklearn.utils.fixes import loguniform

from sklearn.ensemble import RandomForestClassifier
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
train_data.head()
print(f"Dimension of the Training Dataset is {train_data.shape}")

print(f"Dimension of the Testing Dataset is {test_data.shape}")
train_data.info()
def print_the_number_missing_data(data):

    # Lets check the number of duplicated rows

    print("="*30)

    print(f'Number of Duplicate Rows {data[data.duplicated()].shape[0]}')



    #Let's check the percentage of missing values for the application dataset.

    total_missing = data.isnull().sum()

    percent = round((100*(total_missing/data.isnull().count())),2)



    #Making a table for both and get the top 20 Columns



    missing_value_app_data = pd.concat([total_missing, percent], axis =1, keys= ['Total_missing', 'percent'])

    missing_value_app_data.sort_values(by='Total_missing',ascending=False,inplace=True)

    print("="*30)

    print("Number and % of Missing Value")

    print(missing_value_app_data)
print_the_number_missing_data(train_data)
train_data.describe()
def percent_value_counts(df, feature):

    """

    This will take in a dataframe and a column and 

    finds the percentage of the value counts

    """

    percent = pd.DataFrame(round(df[feature].value_counts(dropna=False, normalize=True)*100,2))

    total = pd.DataFrame(df[feature].value_counts(dropna = False))

    total.columns = ["Total"]

    percent.columns = ['Percent']

    return pd.concat([percent, total], axis = 1)    
# Let's check that the target is indeed 0 or 1

percent_value_counts(train_data, "Survived")
percent_value_counts(train_data, "Pclass")
percent_value_counts(train_data, "Sex")
percent_value_counts(train_data, "Embarked")
# This will do the basic feature selection based on the number of features selected 

# This is a custom transformer & Scikit-Learn provides many useful transformers but in this case we would be needing a custom transformer

# Transformer works seamlessly with Scikit-Learn functionalities(such as pipelines), and since Scikit-Learn relies on duck typing (not inheritance),

# all we need is to create a class and implement three methods: fit()(returning self), transform(), and fit_transform().

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attributes_names):

        self.attributes_names = attributes_names

    def fit(self, X, y = None):

        return self

    def transform(self, X):

        return X[self.attributes_names]



# Scikit-Learn provides the Pipeline class to help with such sequences of transformations.

# pipeline for the numerical attributes

num_pipeline = Pipeline([

        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),

        ("imputer", SimpleImputer(strategy="median")),

    ])



# Select the most frequent impute it using the same

# Why can't we just simply use the Simple Impurter in categorical column ? it doesn't work on the caregorical columns

# Hence the workaround is this approach

class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):

        self.most_frequent = pd.Series([X[elem].value_counts().index[0] for elem in X],

                                      index = X.columns)

        return self

    def transform(self, X, y = None):

        return X.fillna(self.most_frequent)



# Categroical Column selection pipeline

cat_pipeline = Pipeline([

    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),

    ("imputer", MostFrequentImputer()),

    ("cat_encoder", OneHotEncoder(sparse=False))

])



# Lets put both the numerical and categorical pipelines together

preprocess_pipeline = FeatureUnion(transformer_list=[

    ("num_pipeline", num_pipeline),

    ("cat_pipeline", cat_pipeline)

])
# Lets see how the numeric pipeline is working

num_pipeline.fit_transform(train_data)
# Lets see how the categorical columns pipeline is working

cat_pipeline.fit_transform(train_data)
preprocess_pipeline.fit_transform(train_data)
# Lets try with ColumnTransformer as well

num_pipeline_mod = Pipeline([

    ("imputer", SimpleImputer(strategy="median"))

])



col_pipeline_mod = Pipeline([

    ("imputer", MostFrequentImputer()),

    ("cat_encoder", OneHotEncoder(sparse=False))

])



num_list = ['Age', 'SibSp', 'Parch', 'Fare']

column_list = ["Pclass", "Sex", "Embarked"]



preprocess_pipeline_mod = ColumnTransformer([

    ('num', num_pipeline_mod, num_list),

    ('cat', col_pipeline_mod, column_list)

])
preprocess_pipeline_mod.fit_transform(train_data)
# Lets compare the Column Transformer Results with the FeatureUnion results

(preprocess_pipeline.fit_transform(train_data) == preprocess_pipeline_mod.fit_transform(train_data) ).all()

# And they are same
# Preprocessing pipeline that takes the raw data and 

# outputs numerical input features that we can feed to any ML model



# Lets prepare the training data

X_train = preprocess_pipeline_mod.fit_transform(train_data.drop(columns='Survived'))

y_train = train_data.Survived
# Ready to train a classifier. Let's start with an SVC:

svm_clf = SVC(gamma="auto")

svm_clf.fit(X_train, y_train)



svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10, scoring='accuracy')

svm_scores.mean()



# 73% accuracy, clearly better than random chance, but it's not a great score.

# Lets try to find the ideal hyperparameter that can boost up the performance
# How to come an ideal combination of Hypeparameters? Answer is GridSearch CV

# All you need to do is tell it which hyperparameters you want it to experiment with, and what values to

# try out, and it will evaluate all the possible combinations of hyperparameter values,

# using cross-validation.

param_list = [

    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [100, 1000, 10000], 'shrinking' : [True, False], 'decision_function_shape' : ['ovo', 'ovr']}]



svm_clf = SVC(random_state=42, verbose=True)

grid_search = GridSearchCV(svm_clf, param_list, cv = 3, verbose = 1, scoring='accuracy', return_train_score=True, n_jobs= -1)

grid_search.fit(X_train, y_train)
grid_search.best_estimator_
grid_search.best_score_
def plot_perfomance_of_cross_validation(arr_list_scores, label):

    plt.figure(figsize=(8, 4))

    plt.boxplot(arr_list_scores)

    plt.plot([1]*len(arr_list_scores), arr_list_scores, ".")

    plt.ylabel("Accuracy", fontsize=14)

    plt.xlabel(label, fontsize = 14)

    plt.show()



cvscores = grid_search.cv_results_

arr_list_svm = []

for mean_score, params in zip(cvscores["mean_test_score"], cvscores["params"]):

    print(round(mean_score,2), params)

    arr_list_svm.append(round(mean_score,2))



plot_perfomance_of_cross_validation(arr_list_svm, "SVM with GridSearchCV")
# The grid search approach is fine when you are exploring relatively few combinations

# but when the hyperparameter search space is large, it is often preferable to use RandomizedSearchCV.

# Instead of trying out all possible combinations,it evaluates a given number of random combinations by selecting a random

# value for each hyperparameter at every iteration.



# Advantages - 

# If you let the randomized search run for, say, 1,000 iterations, this approach will

# explore 1,000 different values for each hyperparameter (instead of just a few values

# per hyperparameter with the grid search approach).

# You have more control over the computing budget you want to allocate to hyperparameter

# search, simply by setting the number of iterations.



param_list = {

    'kernel':['rbf'],

    'C': loguniform(1e0, 1e3),

    'gamma': expon(scale=1.0),

    'shrinking' : [True, False],

    'decision_function_shape' : ['ovo', 'ovr']

}



svm_clf = SVC(random_state=42, verbose=True)

rnd_search = RandomizedSearchCV(svm_clf, param_distributions=param_list,

                                cv=5, scoring='accuracy',n_iter= 50,

                               verbose=1, random_state=42, n_jobs= -1, return_train_score=True)

rnd_search.fit(X_train, y_train)
rnd_search.best_estimator_
rnd_search.best_score_
cvscores = rnd_search.cv_results_

arr_list_svm_rndm = []

for mean_score, params in zip(cvscores["mean_test_score"], cvscores["params"]):

    print(round(mean_score,2), params)

    arr_list_svm_rndm.append(round(mean_score,2))



plot_perfomance_of_cross_validation(arr_list_svm_rndm, "SVM with Randomized")
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv = 10, scoring= 'accuracy')

forest_scores.mean()
# Grid search CV

n_estimators = range(500, 2000, 400 );

max_depth = range(7,12);

criterions = ['gini', 'entropy'];

cv = StratifiedShuffleSplit(n_splits=5, test_size=.30, random_state=15)



parameters = {'n_estimators':n_estimators,

              'max_depth':max_depth,

              'criterion': criterions

              }



random_forest_grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto', n_jobs = -1, random_state = 42),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1, 

                                 verbose= 1, 

                                 return_train_score =True, 

                                 scoring = 'accuracy')

random_forest_grid.fit(X_train, y_train)
random_forest_grid.best_estimator_
random_forest_grid.best_score_
cvscores = random_forest_grid.cv_results_

arr_list_rndc_grid = []

for mean_score, params in zip(cvscores["mean_test_score"], cvscores["params"]):

    arr_list_rndc_grid.append(round(mean_score,2))



plot_perfomance_of_cross_validation(arr_list_rndc_grid, "Random Forest Classifier with Grid Search CV")
# Randomized Search CV



n_estimators = range(500, 2000, 400 );

max_depth = range(7,12);

criterions = ['gini', 'entropy'];

n_estimators = [50, 70, 100, 150]

cv = StratifiedShuffleSplit(n_splits= 10, test_size=.30, random_state=15)



parameters = {'n_estimators':n_estimators,

              'max_depth':max_depth,

              'criterion': criterions,

              'n_estimators': n_estimators,

              'bootstrap':[True, False],

              'min_samples_leaf': range(1,5),

              'warm_start':[True, False],

              'class_weight': ["balanced", "balanced_subsample", None]

             }



random_forest_grid = RandomizedSearchCV(estimator=RandomForestClassifier(max_features='auto', n_jobs = -1, random_state = 42),

                                 param_distributions=parameters,

                                 cv=cv,

                                 n_jobs = -1, 

                                 verbose= 1, 

                                 n_iter= 100,

                                 scoring= 'accuracy',

                                 random_state= 42,

                                 return_train_score =True)

random_forest_grid.fit(X_train, y_train)
random_forest_grid.best_score_
random_forest_grid.best_estimator_
cvscores = random_forest_grid.cv_results_

arr_list_rndc_rndm = []

for mean_score, params in zip(cvscores["mean_test_score"], cvscores["params"]):

    arr_list_rndc_rndm.append(round(mean_score,2))



plot_perfomance_of_cross_validation(arr_list_rndc_rndm, "Random Forest Classifier with Randomized Search CV")
plt.figure(figsize=(15, 4))

plt.plot([1]*len(arr_list_svm), arr_list_svm, ".")

plt.plot([2]*len(arr_list_svm_rndm), arr_list_svm_rndm, ".")

plt.plot([3]*len(arr_list_rndc_grid), arr_list_rndc_grid, ".")

plt.plot([4]*len(arr_list_rndc_rndm), arr_list_rndc_rndm, ".")

plt.boxplot([arr_list_svm, arr_list_svm_rndm,arr_list_rndc_grid, arr_list_rndc_rndm ], labels=("SVM Grid Search","SVM Randomized Search", "Random Forest Grid Search", "Random Forest Randomized Search"))

plt.ylabel("Accuracy", fontsize=14)

plt.ylim(0.6, 0.85)

plt.show()
# Lets build the final Pipeline that will have some data preprocessing steps and the final model as well

prepare_select_predict_pipeline = Pipeline([

    ('preparation', preprocess_pipeline_mod ),

    ('model', RandomForestClassifier(criterion='entropy', max_depth=9, n_estimators=50,

                       n_jobs=-1, random_state=42)  )

])



prepare_select_predict_pipeline.fit(train_data.drop(columns='Survived'), y_train )
# This basically tells how to use the final trained model for prediction 

# So the whole model building steps gets consized in some few steps

rndm_index = np.random.randint(0, len(train_data))

data_output = prepare_select_predict_pipeline.predict(train_data.drop(columns='Survived').iloc[[rndm_index]]).tolist()

print(f"Index choosen is: {rndm_index} and the predicted o/p is: {data_output}, and the actual output should be: {y_train.iloc[[rndm_index]].values}")
# Lets create a Custom Transformer that would add some columns

# But prior to that lets see if some columns can be combined or not



train_data['Total_Members'] = train_data.SibSp + train_data.Parch

train_data[["Total_Members", "Survived"]].groupby(['Total_Members']).mean()



# As it can be seen below lower as the number of members more is the chances of survival
# if the Age Column can also be transformed into some interesting Categorical Columns

train_data["AgeBucket"] = train_data["Age"] // 15 * 15

train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()



# As it can also be seen below age do have a lot of importance like infants/childs have higher chances of survival

# Followed by older people and then the middle aged people
# Lets revert it back to the previous state

train_data = pd.read_csv('../input/titanic/train.csv')
X_train = train_data.copy()
# Lets create a transformer that would do the job of creating these new features & removing the existing ones

class CreateNewFeatureAndRemoveSome(BaseEstimator, TransformerMixin):

    def __init__(self, age_bucket = 15):

        self.age_bucket = age_bucket

    def fit(self, X, y = None):

        return self

    def transform(self, X):

        X['Total_Members'] =  X['SibSp'] + X['Parch']

        X['AgeBucket'] = X['Age'] // self.age_bucket * self.age_bucket

        X['Total_Members'] = X['Total_Members'].astype('object')

        X['AgeBucket'] = X['AgeBucket'].astype('object')

        return X.drop(columns = ['SibSp','Parch','Age'])



attr_adder = CreateNewFeatureAndRemoveSome()

train_data_extra_attr = attr_adder.transform(X_train.copy())

train_data_extra_attr.head()
# Lets try to incorporate the new Transformer CreateNewFeatureAndRemoveSome into the existing pipeline



num_pipeline_mod = Pipeline([

    ("imputer", SimpleImputer(strategy="median"))

])



col_pipeline_mod = Pipeline([

    ("imputer", MostFrequentImputer()),

    ("cat_encoder", OneHotEncoder(sparse=False))

])



feature_adder = Pipeline([

    ("add_features",CreateNewFeatureAndRemoveSome(15)),

    ("treat_as_categorical",col_pipeline_mod )

])



feature_selection_list = ['SibSp', 'Parch', 'Age']

num_list = ['Fare']

column_list = ["Pclass", "Sex", "Embarked"]



preprocess_pipeline_mod = ColumnTransformer([

    ('num', num_pipeline_mod, num_list),

    ('cat', col_pipeline_mod, column_list),

    ('feature_adder', feature_adder, feature_selection_list )

])





X_train_mod = preprocess_pipeline_mod.fit_transform(X_train.drop(columns='Survived').copy())

X_train_mod.shape
# As it can be seen below these are the categories that got transfored into one hot encoded vectors

# As we can see above we have 24 number of features and below it can be seen what are their values

processed_cat = preprocess_pipeline_mod.named_transformers_["cat"]["cat_encoder"].categories_



cat_one_hot_encoded_column = num_list.copy()

attributes = []

for index, elem in enumerate(column_list):

    attributes += list(map(lambda x: f'{elem}_{x}', processed_cat[index]))



cat_one_hot_encoded_column += attributes



# As it can be seen below these are new added categories that got transfered into one hot encoded vectors

added_new_features = preprocess_pipeline_mod.named_transformers_["feature_adder"]["treat_as_categorical"]["cat_encoder"].categories_

added_new_features

attributes = []

for index, elem in enumerate(["Total_Members", "AgeBucket"]):

    attributes += list(map(lambda x: f'{elem}_{x}', added_new_features[index]))

cat_one_hot_encoded_column += attributes

cat_one_hot_encoded_column
n_estimators = range(10, 2000, 50 );

max_depth = range(1,12);

criterions = ['gini', 'entropy'];

n_estimators = [50, 70, 100, 150]

cv = StratifiedShuffleSplit(n_splits= 10, test_size=.30, random_state=15)



parameters = {'n_estimators':n_estimators,

              'max_depth':max_depth,

              'criterion': criterions,

              'n_estimators': n_estimators,

              'bootstrap':[True, False],

              'min_samples_leaf': range(1,5),

              'warm_start':[True, False],

              'class_weight': ["balanced", "balanced_subsample", None]

             }



random_forest_grid = RandomizedSearchCV(estimator=RandomForestClassifier(max_features='auto', n_jobs = -1, random_state = 42),

                                 param_distributions=parameters,

                                 cv=cv,

                                 n_jobs = -1, 

                                 verbose= 1, 

                                 n_iter= 50,

                                 scoring= 'accuracy',

                                 random_state= 42,

                                 return_train_score =True)



random_forest_grid.fit(X_train_mod, y_train)
random_forest_grid.best_estimator_
random_forest_grid.best_score_
# As it can be seen below there are lots of features available but all of them is not so important 

# So lets get the top features and using that lets try to build the models

feature_importance = random_forest_grid.best_estimator_.feature_importances_

sorted(zip(feature_importance, cat_one_hot_encoded_column), reverse=True)
# Lets create a transformer that will select top features based on feature importance

def indices_of_top_k(arr,k):

    return np.sort(np.argpartition(np.array(arr),-k)[-k:])



class TopFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_importances, k):

        self.feature_importances = feature_importances

        self.k = k

    def fit(self, X, y = None):

        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)

        return self

    def transform(self, X):

        return X[:, self.feature_indices_]
top_k_feature_indices = indices_of_top_k(feature_importance, 5)

np.array(cat_one_hot_encoded_column)[top_k_feature_indices]
# This will be the Final Pipeline

# This will take the raw data, do Featue Engineering, Then Select Top Feature and Finally apply the learning algorithm

# So this code is quite consise and its very readable as well

k = 5



final_outcome_pipeline = Pipeline([

    ("preprocess_pipeline_mod", preprocess_pipeline_mod),

    ("feature_selection", TopFeatureSelector(feature_importance, k)),

    ("random_forest_classifier", RandomForestClassifier(criterion='entropy', max_depth=9, n_estimators=50, n_jobs=-1, random_state=42))

])



final_outcome_pipeline.fit(X_train.drop(columns='Survived').copy(), y_train)
final_outcome_pipeline
final_outcome_pipeline.predict(X_train.drop(columns='Survived').iloc[:4])
y_train.iloc[:4]
# Now it can be seen as it as well how easy it is to apply on the test data

test_data['Survived'] = final_outcome_pipeline.predict(test_data)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_data.Survived})

output.to_csv('my_submission.csv', index=False)



print("your submission was successfully saved!")