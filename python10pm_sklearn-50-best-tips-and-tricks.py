# basic libraries

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.display import Image



# this will allow us to print all the files as we generate more in the kernel

def print_files(directory = "output"):

    if directory.lower() == "input":

        for dirname, _, filenames in os.walk('/kaggle/input'):

            for filename in filenames:

                print(os.path.join(dirname, filename))

    else:

        for dirname, _, filenames in os.walk('/kaggle/working'):

            for filename in filenames:

                print(os.path.join(dirname, filename))





def generate_sample_data(): # creates a fake df for testing

    number_or_rows = 20

    num_cols = 7

    cols = list("ABCDEFG")

    df = pd.DataFrame(np.random.randint(1, 20, size = (number_or_rows, num_cols)), columns=cols)

    df.index = pd.util.testing.makeIntIndex(number_or_rows)

    return df





def generate_sample_data_datetime(): # creates a fake df for testing

    number_or_rows = 365*24

    num_cols = 2

    cols = ["sales", "customers"]

    df = pd.DataFrame(np.random.randint(1, 20, size = (number_or_rows, num_cols)), columns=cols)

    df.index = pd.util.testing.makeDateIndex(number_or_rows, freq="H")

    return df



# show several prints in one cell. This will allow us to condence every trick in one cell.

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



print_files("input")



import warnings

warnings.filterwarnings("ignore")
#------------------------------------------------------------

# import libraries

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import make_column_selector, make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



#------------------------------------------------------------

# get the data

lc = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

df = pd.read_csv("/kaggle/input/titanic/train.csv")

X = df[lc]

y = df['Survived']



X_test = pd.read_csv("/kaggle/input/titanic/train.csv", usecols = lc)



# first let's define the steps that will allow us to filter our columns

numeric_select = make_column_selector(dtype_include = "number")

no_numeric_select = make_column_selector(dtype_exclude = 'number')



preprocessor = make_column_transformer(

    (make_pipeline(SimpleImputer(strategy = "mean"), StandardScaler()), numeric_select), # select all numeric columns, impute mean and scale the data

    (make_pipeline(SimpleImputer(strategy = "constant"), OneHotEncoder(handle_unknown='ignore')), no_numeric_select) # select all non numeric columns, impute most frequent value and OneHotEncode

) 



pipe = make_pipeline(preprocessor, LogisticRegression())



# see the crossvalidation score

print("CV score is {}".format(cross_val_score(pipe, X, y).mean()))



# apply the same pipeline and predict

# fit the pipeline and make predictions

pipe.fit(X, y)

y_pred = pipe.predict(X_test)



print("Here we have our predictions",y_pred[:10])
#------------------------------------------------------------

# import libraries

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



#------------------------------------------------------------

# get the data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

X = df[['Sex', 'Name', 'Age']]

y = df['Survived']



# this will be the first Pipeline step

ct = ColumnTransformer(

    [('ohe', OneHotEncoder(), ['Sex']),

     ('vectorizer', CountVectorizer(), 'Name'),

     ('imputer', SimpleImputer(), ['Age'])])



# each of these models will take a turn as the second Pipeline step

clf1 = LogisticRegression(solver='liblinear', random_state=1)

clf2 = RandomForestClassifier(random_state=1)



# create the Pipeline

pipe = Pipeline([('preprocessor', ct), ('classifier', clf1)])



# create the parameter dictionary for clf1

params1 = {}

params1['preprocessor__vectorizer__ngram_range'] = [(1, 1), (1, 2)]

params1['classifier__penalty'] = ['l1', 'l2']

params1['classifier__C'] = [0.1, 1, 10]

params1['classifier'] = [clf1]



# create the parameter dictionary for clf2

params2 = {}

params2['preprocessor__vectorizer__ngram_range'] = [(1, 1), (1, 2)]

params2['classifier__n_estimators'] = [100, 200]

params2['classifier__min_samples_leaf'] = [1, 2]

params2['classifier'] = [clf2]



# create a list of parameter dictionaries

params = [params1, params2]



# this will search every parameter combination within each dictionary

grid = GridSearchCV(pipe, params)
# the best parameters for each method

grid.fit(X, y)

grid.best_params_
#------------------------------------------------------------

# import libraries

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import SelectPercentile, chi2

from sklearn.linear_model import LogisticRegression

from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline



#------------------------------------------------------------

# get the data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

X = df[['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']]

y = df['Survived']



imp_constant = SimpleImputer(strategy='constant')

ohe = OneHotEncoder()



imp_ohe = make_pipeline(imp_constant, ohe)

vect = CountVectorizer()

imp = SimpleImputer()



# pipeline step 1

ct = make_column_transformer(

    (imp_ohe, ['Embarked', 'Sex']),

    (vect, 'Name'),

    (imp, ['Age', 'Fare']),

    ('passthrough', ['Parch']))



# pipeline step 2

selection = SelectPercentile(chi2, percentile=50)



# pipeline step 3

logreg = LogisticRegression(solver='liblinear')



# display estimators as diagrams

from sklearn import set_config

set_config('diagram')



pipe = Pipeline([('preprocessor', ct), ('feature selector', selection), ('classifier', logreg)])



# now we can use our pipeline to fit_transform everything

X_all = pipe.fit(X, y)

pipe.score(X, y)
# but can also access just a part of the pipeline

pipe[1]

X_parcial = pipe[0].fit_transform(X)

X_parcial
#------------------------------------------------------------

# import libraries

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV



#------------------------------------------------------------

# get the data

df = pd.read_csv("/kaggle/input/titanic/train.csv", usecols = ['Survived', 'Pclass', 'Parch', 'SibSp', 'Fare'])

df.dropna(inplace = True)

# separate X and y

X = df.drop("Survived", axis = "columns")

y = df["Survived"]



#------------------------------------------------------------

# instanciate individual models

lr = LogisticRegression()

rf = RandomForestClassifier()

mnb = MultinomialNB()

dt = DecisionTreeClassifier()



#------------------------------------------------------------

# create an ensemble for improved accuracy

vc = VotingClassifier([('clf1', lr), ('clf2', rf), ("clf3", mnb), ("clf4", dt)], voting = 'soft')

print("CV Score of a VotingClassifier is {}".format(cross_val_score(vc, X, y).mean()))



#------------------------------------------------------------

# GridSearch the best parameters

params = {'voting':['hard', 'soft'],

          'weights':[(1,1,1,1), (2,1,1,1), (1,2,1,1), (1,1,2,1), (1,1,1,2)]}



# find the best set of parameters

grid = GridSearchCV(vc, params)

grid.fit(X, y)

grid.best_params_



# accuracy has improved

print("CV Score of a VotingClassifier with GridSearchCV is {}".format(grid.best_score_))
#------------------------------------------------------------

# import libraries

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import cross_val_score



#------------------------------------------------------------

# get the data

df = pd.read_csv("/kaggle/input/titanic/train.csv", usecols = ['Survived', 'Pclass', 'Parch', 'SibSp', 'Fare'])

df.dropna(inplace = True)

# separate X and y

X = df.drop("Survived", axis = "columns")

y = df["Survived"]



#------------------------------------------------------------

# let's fit the individual models

lr = LogisticRegression()

print("CV Score of LogisticRegression is {}".format(cross_val_score(lr, X, y).mean()))



rf = RandomForestClassifier()

print("CV Score of RandomForest is {}".format(cross_val_score(rf, X, y).mean()))



#------------------------------------------------------------

# create an ensemble for improved accuracy

vc = VotingClassifier([('clf1', lr), ('clf2', rf)], voting = 'soft')

print("CV Score of a VotingClassifier is {}".format(cross_val_score(vc, X, y).mean()))
# a very fast way to create features in sklearn

# but be careful, it might be time consuming and impracticale for some algorithms



#------------------------------------------------------------

# import libraries

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(include_bias=False, interaction_only=True)



#------------------------------------------------------------

# get the data

X = pd.DataFrame({'A':[1, 2, 3], 'B':[4, 4, 4], 'C':[0, 10, 100]})





#------------------------------------------------------------

# create new features

# Output columns: A, B, C, A*B, A*C, B*C

poly.fit_transform(X)
#------------------------------------------------------------

# import libraries

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



#------------------------------------------------------------

# get the data

df = pd.read_csv("/kaggle/input/titanic/train.csv", usecols = ["Survived","Sex","Name", "Age"])

df.dropna(inplace = True)

# separate X and y

X = df.drop("Survived", axis = "columns")

y = df["Survived"]



#------------------------------------------------------------

# make a ColumTransformer, instaciate a model and build a pipeline



ct = ColumnTransformer([("ohe", OneHotEncoder(), ["Sex"]),

                       ("vectorizer", CountVectorizer(), "Name"),

                       ("imputer", SimpleImputer(), ["Age"])])



clf = LogisticRegression(solver='liblinear', random_state=1)



pipe = Pipeline([("preprocessor", ct), ("classifier", clf)])



# set the parameters for the GridSearch



params = {}

params['preprocessor__ohe__drop'] = [None, 'first']

params['preprocessor__vectorizer__min_df'] = [1, 2, 3]

params['preprocessor__vectorizer__ngram_range'] = [(1, 1), (1, 2)]

params['classifier__C'] = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

params['classifier__penalty'] = ['l2']
# let's time the GridSearch without all CPU



grid = GridSearchCV(pipe, params)

%time grid.fit(X, y);
# let's time the GridSearch with all CPU

# as you can see, when using -1, it's much faster

grid = GridSearchCV(pipe, params, n_jobs = -1)

%time grid.fit(X, y);
#------------------------------------------------------------

# import libraries

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.pipeline import make_pipeline



#------------------------------------------------------------

# get the data

df = pd.read_csv("/kaggle/input/titanic/train.csv", usecols = ["Survived", "Pclass", "Sex", "Embarked"])

df.dropna(inplace = True)

# separate X and y

X = df.drop("Survived", axis = "columns")

y = df[["Survived"]]



#------------------------------------------------------------

# instanciate 2 encoders

oe = OrdinalEncoder()

ohe = OneHotEncoder()



X_oe = oe.fit_transform(df)

X_ohe = ohe.fit_transform(df)



#------------------------------------------------------------

# let's see the shape of a resulting DataFrame using OE or OHE

print('The dataframe transformed with OrdinalEncoder has a shape of {}'.format(X_oe.shape))

print('The dataframe transformed with OneHotEncoder has a shape of {}'.format(X_ohe.shape))



#------------------------------------------------------------

# let's measure the time needed for training and the corresponing cv score

oe_pipe = make_pipeline(oe, RandomForestClassifier(random_state = 175))

ohe_pipe = make_pipeline(ohe, RandomForestClassifier(random_state = 175))
%time cross_val_score(oe_pipe, X, y).mean()
# as you can see, oe is slightly faster and accuracy doesn't suffer that much.

%time cross_val_score(ohe_pipe, X, y).mean()
#------------------------------------------------------------

# import libraries

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.compose import make_column_transformer



#------------------------------------------------------------

# load some data

X = pd.DataFrame({'A':[1, 2, np.nan],

                  'B':[10, 20, 30],

                  'C':[100, 200, 300],

                  'D':[1000, 2000, 3000],

                  'E':[10000, 20000, 30000]})



# use ColumnTransformer to

# 1. imputer A

# 2. ignore/passthrough B,C

# 3. drop D, E



ct = make_column_transformer((SimpleImputer(), ["A"]),

                            ("passthrough", ["B", "C"]),

                             remainder = "drop")



ct.fit_transform(X)
# use ColumnTransformer to

# 1. imputer A

# 2. drop D, C

# 3. ignore/passthrough B, E



ct = make_column_transformer((SimpleImputer(), ["A"]),

                            ("drop", ["D", "C"]),

                             remainder = "passthrough")



ct.fit_transform(X)
#------------------------------------------------------------

# import libraries

from sklearn.preprocessing import OneHotEncoder



#------------------------------------------------------------

# load some data

df = pd.DataFrame({'Shape':['circle', 'oval', 'square', 'square'],

                  'Color': ['pink', 'yellow', 'pink', 'yellow']})



# as you can see, color is a binary columns

df
# the default behaviour creates a colum per category

# Note: we change sparse = False, to see our matrix

ohe_default = OneHotEncoder(sparse = False).fit_transform(df)

ohe_default
# the default behaviour creates a colum per category

# add drop = "first" to drop first category in each columns

ohe_drop_first = OneHotEncoder(sparse = False, drop = "first").fit_transform(df)

ohe_drop_first
# the default behaviour creates a colum per category

# new in 0.23 allows you to drop if it's binary

# ohe_if_binary = OneHotEncoder(sparse = False, drop = "if_binary").fit_transform(df)

# ohe_if_binary



# in older version you get this error

# ValueError: Wrong input for parameter `drop`. Expected 'first', None or array of objects, got <class 'str'>
#------------------------------------------------------------

# import libraries

from sklearn.linear_model import LogisticRegression



#------------------------------------------------------------

# in old versions of sklearn, when you instanciate an estimator class you will see all it's parameters

# in 0.23 and above, you will only see the parameters that have been changed



clf_old = LogisticRegression(C = 0.1, solver = "liblinear")

clf_old

# in the new version however, you will only see:

# LogisticRegression(C=0.1, solver='liblinear')



# to see all parameters

clf_old.get_params()



# restore old default behaviour

# makes sense in 0.23 and above

from sklearn import set_config

set_config(print_changed_only = False)

clf_old
#------------------------------------------------------------

# import libraries

from sklearn.datasets import load_iris



#------------------------------------------------------------

# load the iris dataset as a DataFrame

# this will work in 0.23: as_frame = True

# df = load_iris(as_frame = True)["frame"]



# return DataFrame with features and Series with target

# X, y = load_iris(as_frame=True, return_X_y=True)



#------------------------------------------------------------

# load a dataframe "old school" method

data = load_iris()



X = data["data"]

y = data["target"]

target_names = data["target_names"] # original values of the target columns

columns = data["feature_names"]



df = pd.DataFrame(X, columns = columns)

df["target"] = y

df.head()
#------------------------------------------------------------

# import libraries

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import make_column_transformer



#------------------------------------------------------------

# get the data: select only 4 columns and drop all na

X = pd.read_csv("/kaggle/input/titanic/train.csv", usecols = ['Embarked', 'Sex', 'Parch', 'Fare']).dropna()



#------------------------------------------------------------

# make a column transform

ct = make_column_transformer((OneHotEncoder(), ["Embarked", "Sex"]),

                            remainder = "passthrough")



#------------------------------------------------------------

# fit transform our columns transformer and see the resulting shape

ct.fit_transform(X).shape



#------------------------------------------------------------

# get the names of the of the features we have created

# in sklearn 0.23 you won't see this error

# NotImplementedError: get_feature_names is not yet supported when using a 'passthrough' transformer.

# ct.get_feature_names()
# When creating large pipelines, create a pipeline to visualize it easier

# New in sklearn 0.23



#------------------------------------------------------------

# import libraries

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import SelectPercentile, chi2

from sklearn.linear_model import LogisticRegression

from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline



#------------------------------------------------------------

# get the data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

X = df[['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']]

y = df['Survived']



imp_constant = SimpleImputer(strategy='constant')

ohe = OneHotEncoder()



imp_ohe = make_pipeline(imp_constant, ohe)

vect = CountVectorizer()

imp = SimpleImputer()



# pipeline step 1

ct = make_column_transformer(

    (imp_ohe, ['Embarked', 'Sex']),

    (vect, 'Name'),

    (imp, ['Age', 'Fare']),

    ('passthrough', ['Parch']))



# pipeline step 2

selection = SelectPercentile(chi2, percentile=50)



# pipeline step 3

logreg = LogisticRegression(solver='liblinear')



# display estimators as diagrams

from sklearn import set_config

set_config('diagram')



pipe = make_pipeline(ct, selection, logreg)

pipe
# Starting from sklearn 0.23 we must pass the parameters for the functions and classes as 

# keyword and not as positional argument. Otherwise a warning will be raised



import sklearn

from sklearn.svm import SVC



# here we won't see because, Kaggle uses a previous version

print(sklearn.__version__)



# positional argument

clf = SVC(0.1, 'linear')



# keyword arguments

clf = SVC(C=0.1, kernel='linear')
# We don't need to use .values when passing a df or a pandas series to sklearn

# It knows internally how to acess the values and deal with them



#------------------------------------------------------------

# import libraries

from sklearn.linear_model import LogisticRegression



#------------------------------------------------------------

# get the data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

X = df[['Pclass', 'Fare']]

y = df["Survived"]



#------------------------------------------------------------

# check the X and y types

print(type(X))

print(type(y))



#------------------------------------------------------------

# instanciate our classes

model = LogisticRegression()



# we fit directly a df and a series and sklearn deals with the rest

model.fit(X, y)
#------------------------------------------------------------

# import libraries

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectPercentile, chi2



#------------------------------------------------------------

# get the data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

X = df["Name"]

y = df["Survived"]



#------------------------------------------------------------

# instanciate our classes

vectorizer = CountVectorizer()

model = LogisticRegression()



#------------------------------------------------------------

# make the pipeline without feature selection

pipe = make_pipeline(vectorizer, model)

score = cross_val_score(pipe, X, y, scoring = 'accuracy').mean()

print("Score of pipeline without feature selection is {}".format(score))



#------------------------------------------------------------

# make the pipeline without feature selection



# keep 50% of features with the best chi-squared scores

selection = SelectPercentile(chi2, percentile = 50)



# add the selection after preprocessing but before model

pipe = make_pipeline(vectorizer, selection, model)

score = cross_val_score(pipe, X, y, scoring = 'accuracy').mean()

print("Score of pipeline with feature selection is {}".format(score))
#------------------------------------------------------------

# import libraries

from sklearn.compose import make_column_transformer

from sklearn.preprocessing import FunctionTransformer



#------------------------------------------------------------

# get some data

X = pd.DataFrame({'Fare':[200, 300, 50, 900],

                  'Code':['X12', 'Y20', 'Z7', np.nan],

                  'Deck':['A101', 'C102', 'A200', 'C300']})



#------------------------------------------------------------

# use an existing column and make the function compatible with Pipeline

clip_values = FunctionTransformer(np.clip, kw_args={'a_min':100, 'a_max':600})



#------------------------------------------------------------

# create a custom function

def first_letter(string_column):

    return string_column.apply(lambda x: x.str.slice(0, 1))



# now use FunctionTransformer to make the function compatible with Pipeline

get_first_letter = FunctionTransformer(first_letter)



#------------------------------------------------------------

# create the column Transformer

ct = make_column_transformer(

    (clip_values, ['Fare']),

    (get_first_letter, ['Code', 'Deck']))



#------------------------------------------------------------

# Original X

print("Original X")

X



#------------------------------------------------------------

# Modified X

print("Modified X")

ct.fit_transform(X)
#------------------------------------------------------------

# import libraries

from sklearn.datasets import load_wine

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score



# get X and y for classification

X, y = load_wine(return_X_y=True)



# select only a few features

X = X[:, 0:2]



# instanciate the model for regression

model_clf = LogisticRegression()



# Multiclass AUC with train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model_clf.fit(X_train, y_train)

y_score = model_clf.predict_proba(X_test)



# use 'ovo' (One-vs-One) or 'ovr' (One-vs-Rest)

print("roc_auc_score is {} for one vs one strategy with train test split".format(roc_auc_score(y_test, y_score, multi_class = 'ovo')))

print("roc_auc_score is {} for one vs rest strategy with train test split".format(roc_auc_score(y_test, y_score, multi_class = 'ovr')))

print("-----------------------------------------")



# Multiclass AUC with cross-validation

# use 'roc_auc_ovo' (One-vs-One) or 'roc_auc_ovr' (One-vs-Rest)

print("cross_val_score is {} for one vs one strategy with cross validation".format(cross_val_score(model_clf, X, y, cv = 5, scoring = 'roc_auc_ovo').mean()))

print("cross_val_score is {} for one vs rest strategy with cross validation".format(cross_val_score(model_clf, X, y, cv = 5, scoring = 'roc_auc_ovr').mean()));
#------------------------------------------------------------

# import libraries

from sklearn.datasets import load_diabetes

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold, StratifiedKFold



# get X and y for regression

X_reg, y_reg = load_diabetes(return_X_y = True)



# get X and y for classification

df = pd.read_csv("/kaggle/input/titanic/train.csv", usecols = ['Pclass', 'Fare', 'SibSp', 'Survived']).dropna()



# separate X and y

X_clf = df[['Pclass', 'Fare', 'SibSp']]

y_clf = df[["Survived"]]



# instanciate the model for regression

model_reg = LinearRegression()



# instanciate the model for classification

model_clf = LogisticRegression()



# Use KFold for regression

kf = KFold(5, shuffle = True, random_state = 1)

print("cross_val_score for regression model")

cross_val_score(model_reg, # the regression model, in our case, LinearRegression

                X_reg, # X: features to learn from

                y_reg, # y: what the predict

                cv = kf, # cross_validation scheme we have created earlier

                scoring = "r2") # metric to use to validate the quality of the model



# Use StratifiedKFold for classification

skf = StratifiedKFold(5, shuffle = True, random_state = 1)

print("cross_val_score for classification model")

cross_val_score(model_clf, # the model, in our case, LogisticRegression

                X_clf, # X: features to learn from

                y_clf, # y: what the predict

                cv = skf, # cross_validation scheme we have created earlier

                scoring = "accuracy") # metric to use to validate the quality of the model
#------------------------------------------------------------

# import libraries

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline



#------------------------------------------------------------

# instanciate the classes

ohe = OneHotEncoder()

model = LogisticRegression()



#------------------------------------------------------------

# get the data

df = pd.read_csv("/kaggle/input/titanic/train.csv", usecols = ['Embarked', 'Survived']).dropna()



# separate X and y

X = df[["Embarked"]]

y = df[["Survived"]]



# create a pipeline and fit the X and y

pipe = Pipeline([("ohe", ohe),

                 ("clf", model)])

pipe.fit(X, y)



# inspect the coefficients

print("1 way to show model coefficients")

pipe.named_steps.clf.coef_

print("2 way to show model coefficients")

pipe.named_steps["clf"].coef_

print("3 way to show model coefficients")

pipe["clf"].coef_

print("4 way to show model coefficients")

pipe[1].coef_
#------------------------------------------------------------

# import libraries

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.compose import make_column_transformer



#------------------------------------------------------------

# get train data

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")



# drop any nans

X = df_train[["Name", "Cabin"]].dropna()



#------------------------------------------------------------

# instanciate CountVectorizer

count_vect = CountVectorizer()



#------------------------------------------------------------

# instanciate CountVectorizer

# You can pass the CountVectorizer multiple times and it will learn

# separate vocabularies.

# to do so, you must use make_column_transformer

ct = make_column_transformer((count_vect, 'Name'), (count_vect, 'Cabin'))

X_transform = ct.fit_transform(X)

X_transform
#------------------------------------------------------------

# import libraries

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

import joblib



#------------------------------------------------------------

# get train data

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")



# drop any nans

df_train.dropna(axis = "rows", inplace = True)



# separate X and y

cols_for_x = ["Embarked", "Sex"]

X_train = df_train[cols_for_x]

y_train = df_train["Survived"]



#------------------------------------------------------------

# get test data

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")



# drop any nans

df_test.dropna(axis = "rows", inplace = True)



X_test = df_test[cols_for_x]



#------------------------------------------------------------

# instanciate Ohe and make_pipeline

ohe = OneHotEncoder()

model = LogisticRegression()



#------------------------------------------------------------

# create the pipeline

pipe = make_pipeline(ohe, model)



#------------------------------------------------------------

# predict_using pipeline

pipe.fit(X_train, y_train)



#------------------------------------------------------------

# save pipeline

joblib.dump(pipe, 'pipe.joblib')



# print our newly saved pipeline

print_files()



#------------------------------------------------------------

# save pipeline

new_pipe = joblib.load('/kaggle/working/pipe.joblib')



#------------------------------------------------------------

# predict using the same pipe and the old pipe

print("------------")

print("Old pipe.")

pipe.predict(X_test)

print("------------")

print("Old pipe.")

new_pipe.predict(X_test)

print("Notice that both pipes predict the same result.")
#------------------------------------------------------------

# get some fake data

d = {"Shape_Original":["square", "square", "square", "oval", "circle", np.nan]}

df = pd.DataFrame(d)



#------------------------------------------------------------

# import libraries

from sklearn.impute import SimpleImputer



#------------------------------------------------------------

# impute values using most frequent

df["most_frequent"] = SimpleImputer(strategy = "most_frequent").fit_transform(df[["Shape_Original"]])



#------------------------------------------------------------

# impute values using most constant

df["constant"]  = SimpleImputer(strategy = "constant", fill_value = "missing").fit_transform(df[["Shape_Original"]])



#------------------------------------------------------------

# the result of our imputation

df.style.apply(lambda x: ['background: lightgreen' if x.name == 5 else '' for i in x], axis=1)
#------------------------------------------------------------

# import libraries

from sklearn.model_selection import train_test_split



#------------------------------------------------------------

# generate some data and separate X and y

df = pd.DataFrame({'feature':list(range(8)), 'target':['not fraud']*6 + ['fraud']*2})

X = df[['feature']]

y = df['target']



#------------------------------------------------------------

# train and test without stratify



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 1)



print("y_train withous stratify")

y_train

print("y_test withous stratify")

y_test





#------------------------------------------------------------

# train and test with stratify



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, stratify = y, random_state = 1)



print("Notice how using statify preserves the fraud and not fraud percentage.")

print("y_train with stratify")

y_train

print("y_test with stratify")

y_test
import sklearn

print(sklearn.__version__)



#------------------------------------------------------------

# import libraries

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier



#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

df['Sex'] = df['Sex'].map({'male':0, 'female':1})

X = df[["Pclass", "Fare", "Sex"]]

y = df["Survived"]



#------------------------------------------------------------

# basic model and evaluation

model = DecisionTreeClassifier(random_state = 175)

model.fit(X, y)

score = cross_val_score(model, X, y, scoring = "accuracy")

print("Our DecissionTree with {} nodes has scored {}".format(model.tree_.node_count, score.mean()))



#------------------------------------------------------------

# prun the tree and see cross validation score

# Notice that the score went up. Prunnig trees has a lot of benefits, the main one is reducing overfitting.

# ccp_alpha is the parameter that controls the decision tree complexity (cost complexity parameter).

# Greater values of ccp_alpha increase the number of nodes pruned.

# documentation 

# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html



model = DecisionTreeClassifier(ccp_alpha = 0.001, random_state = 175)

model.fit(X, y)

score = cross_val_score(model, X, y, scoring = "accuracy")

print("Our prunned DecissionTree with {} nodes has scored {}".format(model.tree_.node_count, score.mean()))
import sklearn

print(sklearn.__version__)



#------------------------------------------------------------

# import libraries

from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text



#------------------------------------------------------------

# create our instances

model = DecisionTreeClassifier()



#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

df['Sex'] = df['Sex'].map({'male':0, 'female':1})

X = df[["Pclass", "Fare", "Sex"]]

y = df["Survived"]



features = ["Pclass", "Fare", "Sex"]

classes = ["Survived"]



#------------------------------------------------------------

# train test split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



#------------------------------------------------------------

# fit and predict



model.fit(X_train, y_train)



#------------------------------------------------------------

# plot the tree



plt.figure(figsize = (20, 10))

plot_tree(model, feature_names = features, filled = True);
#------------------------------------------------------------

# show the text

# I will plot only the first 200 characters of the tree since it grows rapidly

print(export_text(model, feature_names = features, show_weights=True)[:200]);
#------------------------------------------------------------

# import libraries

from sklearn.datasets import load_diabetes

from sklearn.linear_model import LinearRegression



#------------------------------------------------------------

# load data and separate X and y

dataset = load_diabetes()

X, y = dataset.data, dataset.target

features = dataset.feature_names



#------------------------------------------------------------

# fit model

model = LinearRegression()

model.fit(X, y)



#------------------------------------------------------------

# intercept and coef

model.intercept_

model.coef_

list(zip(features, model.coef_))
#------------------------------------------------------------

# Two types of ROC Curve



# If the pipeline ends in a classifier or regressor, you use the fit and predict methods

# If the pipeline ends in a transformer you use the fit_transform and transform methods



path = "/kaggle/input/roc-curve/ROC Curve.jpeg"

Image(path)
import sklearn

print(sklearn.__version__)



#------------------------------------------------------------

# import libraries

from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



#------------------------------------------------------------

# create our instances

lr = LogisticRegression()

dt = DecisionTreeClassifier()

rf = RandomForestClassifier()



#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

X = df[["Pclass", "Fare"]]

y = df["Survived"]



#------------------------------------------------------------

# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



#------------------------------------------------------------

# fit and predict

lr.fit(X_train, y_train)

dt.fit(X_train, y_train)

rf.fit(X_train, y_train)



#------------------------------------------------------------

# plot roc curve

disp = plot_roc_curve(lr, X_test, y_test)

plot_roc_curve(dt, X_test, y_test, ax = disp.ax_) # ax = disp.ax_ this line will share the x axis

plot_roc_curve(rf, X_test, y_test, ax = disp.ax_)
import sklearn

print(sklearn.__version__)



#------------------------------------------------------------

# import libraries

from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix

from sklearn.linear_model import LogisticRegression



#------------------------------------------------------------

# create our instances

model = LogisticRegression(random_state = 1)



#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

X = df[["Pclass", "Fare"]]

y = df["Survived"]



#------------------------------------------------------------

# train test split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



#------------------------------------------------------------

# fit and predict



model.fit(X_train, y_train)



#------------------------------------------------------------

# plot confusion matrix

# notice that you have to pass the model, X_test and y_test

# plot_confusion_matrix predicts with the model and plots the values



disp = plot_confusion_matrix(model, X_test, y_test, cmap = "Blues", values_format = ".3g")



print("The classical confusion matrix")

disp.confusion_matrix
#------------------------------------------------------------

# C: inverse of regularization strength

# penalty: type of regularization

# solver: algorithm used for optimization



path = "/kaggle/input/logisticregression/LogisticRegression.jpg"

Image(path)
#------------------------------------------------------------

# import libraries

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression



#------------------------------------------------------------

# create our instances

ohe = OneHotEncoder()

vect = CountVectorizer()

ct = make_column_transformer((ohe, ["Sex"]), (vect, "Name"))

model = LogisticRegression(solver = "liblinear", random_state = 1)



#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

X = df[["Sex", "Name", "Fare"]]

y = df["Survived"]



#------------------------------------------------------------

# make pipeline

pipeline = make_pipeline(ct, model)



#------------------------------------------------------------

# cross validate the entire pipeline

print("Notice the score of our entire pipeline is {}".format(cross_val_score(pipeline, X, y, cv = 5, scoring = "accuracy").mean()))

cross_val_score(pipeline, X, y, cv = 5, scoring = "accuracy").mean()



#------------------------------------------------------------

# gridsearch the entire pipeline



# set the parameters

params = {"columntransformer__countvectorizer__min_df":[1, 2],

         "logisticregression__C":[0.1, 1, 10],

         "logisticregression__penalty":["l1", "l2"]}



grid = GridSearchCV(pipeline, params, cv = 5, scoring = "accuracy")

grid.fit(X, y)



# convert to a pandas DataFrame



results = pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "rank_test_score"]]

results.sort_values("rank_test_score")
#------------------------------------------------------------

# import libraries

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from sklearn.naive_bayes import MultinomialNB

import scipy as sp



#------------------------------------------------------------

# create our instances

vect = CountVectorizer()

model = MultinomialNB()



#------------------------------------------------------------

# make pipeline

pipeline = make_pipeline(vect, model)



#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

X = df['Name']

y = df['Survived']



#------------------------------------------------------------

# set the params to optimize



params = {}



params["countvectorizer__min_df"] = [1, 2, 3, 4]

params["countvectorizer__lowercase"] = [True, False]

params["multinomialnb__alpha"] = sp.stats.uniform(scale = 1)



#------------------------------------------------------------

# optimize



rand = RandomizedSearchCV(pipeline, params, n_iter = 10, cv = 5, scoring = "accuracy", random_state = 1)

rand.fit(X, y)



#------------------------------------------------------------

# best score and params



print("Best score achieved with our search is:")

rand.best_score_



print("Best params are:")

rand.best_params_
#------------------------------------------------------------

# import libraries

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression



#------------------------------------------------------------

# create our instances

ohe = OneHotEncoder()

vect = CountVectorizer()

ct = make_column_transformer((ohe, ["Sex"]), (vect, "Name"))

model = LogisticRegression(solver = "liblinear", random_state = 1)



#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

X = df[["Sex", "Name", "Fare"]]

y = df["Survived"]



#------------------------------------------------------------

# make pipeline

pipeline = make_pipeline(ct, model)



#------------------------------------------------------------

# cross validate the entire pipeline

print("Notice the score of our entire pipeline is {}".format(cross_val_score(pipeline, X, y, cv = 5, scoring = "accuracy").mean()))

cross_val_score(pipeline, X, y, cv = 5, scoring = "accuracy").mean()



#------------------------------------------------------------

# gridsearch the entire pipeline



# set the parameters

params = {"columntransformer__countvectorizer__min_df":[1, 2],

         "logisticregression__C":[0.1, 1, 10],

         "logisticregression__penalty":["l1", "l2"]}



grid = GridSearchCV(pipeline, params, cv = 5, scoring = "accuracy")

grid.fit(X, y)



# see the best score

print("#-------------------------------------------------------------------------")

print("Best score of the GridSearchCV is ")

grid.best_score_



# see the best params

print("#-------------------------------------------------------------------------")

print("Best parameters of the GridSearchCV are ")

grid.best_params_
#------------------------------------------------------------

# If you have missing values you can:

# 1. Drop all rows with missing values

# 2. Drop all colmns with missing values

# 3. Impute missing values

# 4. Use a model that handles missing values



#------------------------------------------------------------

# import libraries

from sklearn.experimental import enable_hist_gradient_boosting # this import enables "experimental" packages and clases in sklearn

from sklearn.ensemble import HistGradientBoostingClassifier # this is an experimental package

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

print("We can see that we have missing values")



df.isnull().sum()



#------------------------------------------------------------

# split target and feature



features = [col for col in df.columns if df[col].dtype != "object"] # select only numerical columns

features.remove("Survived")

features.remove("PassengerId")



X = df[features]

y = df["Survived"]



#------------------------------------------------------------

# Train a model that handles missing values



model = HistGradientBoostingClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 175)



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print("Accuracy of our model while having missing values is {}%".format(round(accuracy_score(y_test, y_pred), 2)*100))

#------------------------------------------------------------

# import libraries

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline



#------------------------------------------------------------

# create each instance

si = SimpleImputer()

model = LogisticRegression()



#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")



#------------------------------------------------------------

# select columns to transform

X = df[["Fare", "Age"]].head()

X["Age"].iloc[0] = np.nan # create a missing values

X.head()



y = df[["Survived"]].head()



#------------------------------------------------------------

# use make_pipeline



pipeline = make_pipeline(si, model)



pipeline.fit(X, y)



#------------------------------------------------------------

# let's see the statistics of each step

print("These are the imputed values with the SimpleImputer")

pipeline.named_steps.simpleimputer.statistics_



print("Display the coefficients of the linear model")

pipeline.named_steps.logisticregression.coef_
#------------------------------------------------------------

# import libraries

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.compose import make_column_transformer, ColumnTransformer

from sklearn.pipeline import make_pipeline, Pipeline



#------------------------------------------------------------

# create each instance

ohe = OneHotEncoder()

si = SimpleImputer()

model = LogisticRegression()



#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")



#------------------------------------------------------------

# select columns to transform

X = df[["Fare", "Embarked", "Sex", "Age"]].head()

X["Age"].iloc[0] = np.nan # create a missing values

X.head()



y = df[["Survived"]].head()



#------------------------------------------------------------

# use make_pipeline



column_transformer = make_column_transformer(

(ohe, ["Embarked", "Sex"]),

(si, ["Age"]),

remainder = "passthrough"

)



pipeline = make_pipeline(column_transformer, model)



pipeline.fit(X, y)



#------------------------------------------------------------

# use Pipeline

# The main difference is that we must name each step



column_transformer = ColumnTransformer(

[("encoder", ohe, ["Embarked", "Sex"]), # notice how we must name each step

("imputer", si, ["Age"])],

remainder = "passthrough"

)



pipeline = Pipeline([("preprocessing", column_transformer), ("model", model)]) # notice how we must name each step

pipeline.fit(X, y)
#------------------------------------------------------------

# import libraries

from sklearn.impute import KNNImputer



#------------------------------------------------------------

# create some random data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

df.isnull().sum() # inspect nulls

df.head()



#------------------------------------------------------------

# let's see KNNInputer in action

knn_inputer = KNNImputer()



X = df[["SibSp", "Fare", "Age"]]

nan_index = X[X["Age"].isnull()].index



print("Data with nans")

X[X.index.isin(nan_index)].head(10)



print("Transformed data with no nans, and the values are based on the NN.")

X_transformed = pd.DataFrame(knn_inputer.fit_transform(X), columns = ["SibSp", "Fare", "Age"], index = X.index)

X_transformed[X.index.isin(nan_index)].head(10)
#------------------------------------------------------------

# import libraries

from sklearn.model_selection import train_test_split



#------------------------------------------------------------

# create some random data



df = generate_sample_data()

target = [np.random.choice([0, 1]) for i in range(len(df))]

df["target"] = target

print("A sneak peak at our df.")

df.head(3)



#------------------------------------------------------------

# train and target columns

features = list(df.columns)[:-1] # all except the last one



#------------------------------------------------------------

# split nr 1 using random_state to reproduce results

X_train1, X_test1, y_train1, y_test1 = train_test_split(df[features], df["target"], test_size = 0.1, random_state = 175)



#------------------------------------------------------------

# split nr 2 using random_state to reproduce results

X_train2, X_test2, y_train2, y_test2 = train_test_split(df[features], df["target"], test_size = 0.1, random_state = 175)



#------------------------------------------------------------

# split nr 3 using with no random_state

X_train3, X_test3, y_train3, y_test3 = train_test_split(df[features], df["target"], test_size = 0.1)



#------------------------------------------------------------

# let's look at our results

print("First train test split.")

X_train1.head()

print("Second train test split. Equal to the first one.")

X_train2.head()

X_train1.index == X_train2.index

print("Third train test split. Different than the others.")

X_train3.head()

X_train1.index == X_train3.index

#------------------------------------------------------------

# import the libraries

from sklearn.impute import SimpleImputer



#------------------------------------------------------------

# create some train and test data



# create a train df

d = {

"Age":[10, 20, np.nan, 30, 15, 10, 40, 10, np.nan]

}



print("Train data")

df_train = pd.DataFrame(d)

df_train



#------------------------------------------------------------

# Sometimes there is a relathionship between missing values and the target

# We can use this information creating a new features while performing an imputation of a missing values



# normal SimpleImputer()

imputer = SimpleImputer()

df_transformed = imputer.fit_transform(df_train)

df_transformed



# SimpleImputer() with the parameter add_indicator

print("Notice aditional column with an indicator of 1 next to the previously missing values")

imputer = SimpleImputer(add_indicator = True)

df_transformed = imputer.fit_transform(df_train)

df_transformed

#------------------------------------------------------------

# import the libraries

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline



#------------------------------------------------------------

# create some train and test data



# create a train df

d = {

"feat1":[10, 20, np.nan, 2],

"feat2":[25, 20, 5, 3],

"target":["A", "A", "B", "B"]

}



print("Train data")

df_train = pd.DataFrame(d)

df_train



# create a test df

d = {

"feat1":[30, 5, 15],

"feat2":[12, 10, np.nan]

}



print("Test data")

df_test = pd.DataFrame(d)

df_test



#------------------------------------------------------------

# simple ML project step by step



# create each instance

si = SimpleImputer()

model = LogisticRegression()

pipeline = make_pipeline(si, model)



# separate the data between target and features

features = ["feat1", "feat2"]

X_train, y_train = df_train[features], df_train["target"]

X_test = df_test[features]



#------------------------------------------------------------

# use pipeline to fit and predict



# First pipeline will use the SimpleImputer to imputer the missing values

# Then it will train using LogisticRegression

pipeline.fit(X_train, y_train)



# When used pipeline to predict, it will do the same steps as in fit

pipeline.predict(X_test)
#------------------------------------------------------------

# create a train df

d = {

"Categorical":["A", "A", "B", "C"]

}



df_train = pd.DataFrame(d)

df_train



#------------------------------------------------------------

# import the libraries

from sklearn.preprocessing import OneHotEncoder



#------------------------------------------------------------

# transform data during train part

print("fit_transform using OneHotEncoder.")

ohe = OneHotEncoder(sparse = False, handle_unknown = "ignore") # if you don't put false, you will get a sparse matrix object

X_train = ohe.fit_transform(df_train[["Categorical"]])

X_train



#------------------------------------------------------------

# create a test df



d = {

"Categorical":["A", "A", "B", "C", "D"] # new value, D, previously not seen in train

}



df_test = pd.DataFrame(d)

df_test





print("transform using OneHotEncoder. Notice that we have a line with zeros for categorical value of D")

X_test = ohe.transform(df_test[["Categorical"]])

X_test

#------------------------------------------------------------

# create a df

d = {

"Shape":["square", "square", "oval", "circle"],

"Class":["third", "first", "second", "third"],

"Size":["S", "S", "L", "XL"]

}



df = pd.DataFrame(d)

df



#------------------------------------------------------------

# import the libraries

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OrdinalEncoder



#------------------------------------------------------------

# transform data using OneHotEncoder

print("Transform categorical data using OneHotEncoder")

ohe = OneHotEncoder(sparse = False) # if you don't put false, you will get a sparse matrix object

shaped_transformed = ohe.fit_transform(df[["Shape"]]) # if you pass as a series, you will need to reshape the data. Notice the double square bracket

shaped_transformed



#------------------------------------------------------------

# transform data using OrdinalEncoder

print("Transform categorical data using OrdinalEncoder")

print("When using OrdinalEncoder, your data has to have a order: like first class, second class, third class")

oe = OrdinalEncoder(categories = [["first", "second", "third"], # order for the column Class

                                  ["S", "M", "L", "XL"]]) # order for the column Size

categorical_ordinal_transformed = oe.fit_transform(df[["Class", "Size"]])

categorical_ordinal_transformed
#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")



#------------------------------------------------------------

# select columns to transform

X = df[["Fare", "Embarked", "Sex", "Age"]].head()

X["Age"].iloc[0] = np.nan # create a missing values

X



#------------------------------------------------------------

# import the libraries

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import make_column_selector

from sklearn.compose import make_column_transformer



#------------------------------------------------------------

# instanciate the classes

ohe = OneHotEncoder()



#------------------------------------------------------------

# create the pipeline and select the columns by name



print("Select the column by name")



ct = make_column_transformer(

(ohe, ["Embarked", "Sex"])# if you have null values it will give and error. You must first fill those values before doing an ohe

)



# fit_transform the columns

X_transformed = ct.fit_transform(X) 

X_transformed



#------------------------------------------------------------

# create the pipeline and select the columns by position



print("Select the column by position")



ct = make_column_transformer(

(ohe, [1, 2])# if you have null values it will give and error. You must first fill those values before doing an ohe

)



# fit_transform the columns

X_transformed = ct.fit_transform(X) 

X_transformed



#------------------------------------------------------------

# create the pipeline and select the columns using slice



print("Select the column using slice")



ct = make_column_transformer(

(ohe, slice(1, 3))# if you have null values it will give and error. You must first fill those values before doing an ohe

)



# fit_transform the columns

X_transformed = ct.fit_transform(X) 

X_transformed



#------------------------------------------------------------

# create the pipeline and select the columns using a boolean mask



print("Select the column using a boolean mask")



ct = make_column_transformer(

(ohe, [False, True, True, False])# if you have null values it will give and error. You must first fill those values before doing an ohe

)



# fit_transform the columns

X_transformed = ct.fit_transform(X) 

X_transformed



#------------------------------------------------------------

# create the pipeline and select the columns using make_column_selector and regex



print("Select the column using using make_column_selector and regex. New in pandas 0.22")



ct = make_column_transformer(

(ohe, make_column_selector(pattern = "E|S"))# if you have null values it will give and error. You must first fill those values before doing an ohe

)



# fit_transform the columns

X_transformed = ct.fit_transform(X) 

X_transformed



#------------------------------------------------------------

# create the pipeline and select the columns using make_column_selector and dtype_include



print("Select the column using using make_column_selector and dtype_include. New in pandas 0.22")



ct = make_column_transformer(

(ohe, make_column_selector(dtype_include = object))# if you have null values it will give and error. You must first fill those values before doing an ohe

)



# fit_transform the columns

X_transformed = ct.fit_transform(X) 

X_transformed



#------------------------------------------------------------

# create the pipeline and select the columns using make_column_selector and dtype_exclude



print("Select the column using using make_column_selector and dtype_exclude. New in pandas 0.22")



ct = make_column_transformer(

(ohe, make_column_selector(dtype_exclude = "number"))# if you have null values it will give and error. You must first fill those values before doing an ohe

)



# fit_transform the columns

X_transformed = ct.fit_transform(X) 

X_transformed
#------------------------------------------------------------

# import data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

df.head()



#------------------------------------------------------------

# select columns to transform

X = df[["Fare", "Embarked", "Sex", "Age"]].head()

X["Age"].iloc[0] = np.nan # create a missing values

X



#------------------------------------------------------------

# import the libraries

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import make_column_transformer



#------------------------------------------------------------

# instanciate the classes

ohe = OneHotEncoder()

imp = SimpleImputer()



#------------------------------------------------------------

# create the pipeline

ct = make_column_transformer(

(ohe, ["Embarked", "Sex"]), # if you have null values it will give and error. You must first fill those values before doing an ohe

(imp, ["Age"]), 

remainder = "passthrough" # this means that the column Fare will appear the last one.

)



#------------------------------------------------------------

# fit_transform the columns

X_transformed = ct.fit_transform(X) 

X_transformed