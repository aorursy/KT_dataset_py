# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.graph_objects as go

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier

import time

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
#Custom Transformer that extracts columns passed as argument to its constructor 

class FeatureSelector(BaseEstimator, TransformerMixin ):

  #Class Constructor 

  def __init__( self, feature_names ):

    self.feature_names = feature_names 

    

  #Return self nothing else to do here    

  def fit( self, X, y = None ):

    return self 

    

  #Method that describes what we need this transformer to do

  def transform(self, X, y = None):

    return X[self.feature_names]
#converts certain features to categorical

class CategoricalTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, model=0):

    """Class constructor method that take: 

    model: 

      - 0: Sex column (categorized), Pclass (raw)

      - 1: Sex column (get_dummies(drop_first=True)), Pclass (raw)

      - 2: Sex column (get_dummies(drop_first=True)), Pclass (get_dummies(drop_first=False))

      - 3: Sex column (get_dummies(drop_first=True)), Pclass (raw), Age (get_dummies(drop_first=False))

      - 4: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (categorized)

      - 5: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))

      - 6: New Sex column (get_dummies(drop_first=False)), Pclass (raw), Embarked (get_dummies(drop_first=False))

      - 7: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))

      - 8: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size

      - 9: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size, Embarked (get_dummies(drop_first=False))

    """

    self.model = model



  #Return self nothing else to do here    

  def fit( self, X, y = None ):

    return self 



  def create_dummies(self, df, column_name, drop_first_col):

    """Create Dummy Columns from a single Column

    """

    dummies = pd.get_dummies(df[column_name],prefix=column_name, drop_first=drop_first_col)

    return dummies



  def process_family_size(self, df):

    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

    titles = {

        "Mr" :         "man",

        "Mme":         "woman",

        "Ms":          "woman",

        "Mrs" :        "woman",

        "Master" :     "boy",

        "Mlle":        "woman",

        "Miss" :       "woman",

        "Capt":        "man",

        "Col":         "man",

        "Major":       "man",

        "Dr":          "man",

        "Rev":         "man",

        "Jonkheer":    "man",

        "Don":         "man",

        "Sir" :        "man",

        "Countess":    "woman",

        "Dona":        "woman",

        "Lady" :       "woman"

    } 



    # new gender: man, woman, boy

    df["Gender"] = df["Title"].map(titles)



    # family surname

    df["family"] = df["Name"].str.extract('([A-Za-z]+)\,',expand=False)



    # count the number of boy and women by family

    boy_women = df[df["Gender"] != "man"].groupby(by=["family"])["Name"].agg("count")



    # fill with zero that passengers are traveling alone or with family without boy and women

    df["family_size"] = df["family"].map(boy_women).fillna(0.0)



    if self.model in [8,9]:

      return pd.DataFrame(df["family_size"],columns=["family_size"])

    else:

      return None



  def process_sex(self, df):

    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

    titles = {

        "Mr" :         "man",

        "Mme":         "woman",

        "Ms":          "woman",

        "Mrs" :        "woman",

        "Master" :     "boy",

        "Mlle":        "woman",

        "Miss" :       "woman",

        "Capt":        "man",

        "Col":         "man",

        "Major":       "man",

        "Dr":          "man",

        "Rev":         "man",

        "Jonkheer":    "man",

        "Don":         "man",

        "Sir" :        "man",

        "Countess":    "woman",

        "Dona":        "woman",

        "Lady" :       "woman"

    }

    

    if self.model == 0:

      df["Sex"] = pd.Categorical(df.Sex).codes

      return pd.DataFrame(df["Sex"],columns=["Sex"])

    elif self.model in [1,2,3,4,5]:  

      sex_dummies = self.create_dummies(df,"Sex",True)

      return sex_dummies

    elif self.model == 6:

      df["Sex"] = df["Title"].map(titles)

      sex_dummies = self.create_dummies(df,"Sex",False)

      return sex_dummies

    elif self.model in [7,8,9]:

      df["Sex"] = df["Title"].map(titles)

      sex_dummies = self.create_dummies(df,"Sex",False)

      sex_dummies.drop(labels="Sex_woman",axis=1,inplace=True)

      return sex_dummies

    else:

      return None



  def process_embarked(self, df):

    if self.model in [0,1,2,3,8]:

      return None

    elif self.model == 4:

      # fill null values using the mode

      df["Embarked"].fillna("S",inplace=True)

      df["Embarked"] = pd.Categorical(df.Embarked).codes

      return pd.DataFrame(df["Embarked"],columns=["Embarked"])

    elif self.model in [5,6,7,9]:

      df["Embarked"].fillna("S",inplace=True)

      embarked_dummies = self.create_dummies(df,"Embarked",False)

      return embarked_dummies



  #Transformer method we wrote for this transformer 

  def transform(self, X , y = None ):

    df = X.copy()

    sex = self.process_sex(df)

    embarked = self.process_embarked(df)

    family_size = self.process_family_size(df)



    if self.model in [0,1,2,3]:

      return sex

    elif self.model in [4,5,6,7]:

      return pd.concat([sex,embarked],axis=1)

    elif self.model == 8:

      return pd.concat([sex,family_size],axis=1)

    elif self.model == 9:

      return pd.concat([sex,family_size,embarked],axis=1)

    else:

      return None
# for validation purposes only

select = FeatureSelector(train.select_dtypes(include=["object"]).columns).transform(train)



# change the value of model 0,1,2,3,....7

model = CategoricalTransformer(model=9)

df_cat = model.transform(select)
#converts certain features to categorical

class NumericalTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, model=0):

    """Class constructor method that take: 

    model: 

      - 0: Sex column (categorized), Pclass (raw)

      - 1: Sex column (get_dummies(drop_first=True)), Pclass (raw)

      - 2: Sex column (get_dummies(drop_first=True)), Pclass (get_dummies(drop_first=False))

      - 3: Sex column (get_dummies(drop_first=True)), Pclass (raw), Age (get_dummies(drop_first=False))

      - 4: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (categorized)

      - 5: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))

      - 6: New Sex column (get_dummies(drop_first=False)), Pclass (raw), Embarked (get_dummies(drop_first=False))

      - 7: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))

      - 8: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size

      - 9: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size, Embarked (get_dummies(drop_first=False))

    """

    self.model = model



  #Return self nothing else to do here    

  def fit( self, X, y = None ):

    return self 



  def create_dummies(self, df, column_name, drop_first_col):

    """Create Dummy Columns from a single Column

    """

    dummies = pd.get_dummies(df[column_name],prefix=column_name, drop_first=drop_first_col)

    return dummies



  # manipulate column "Age"

  def process_age(self, df):

    # fill missing values with -0.5

    df["Age"] = df["Age"].fillna(-0.5)



    # divide age column into a range of values

    cut_points = [-1,0,5,12,18,35,60,100]

    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

    df["Age_categories"] = pd.cut(df["Age"],

                                 cut_points,

                                 labels=label_names)

    if self.model in [0,1,2,6,7,8,9]:

      return None

    elif self.model == 3:

      return self.create_dummies(df,"Age_categories",False)

   

  def process_pclass(self, df):

    if self.model in [0,1,3,4,5,6,7,8,9]:

      return pd.DataFrame(df["Pclass"],columns=["Pclass"])

    elif self.model == 2:

      return self.create_dummies(df,"Pclass",False)

    else:

      return None

        

  #Transformer method we wrote for this transformer 

  def transform(self, X , y = None ):

    df = X.copy()



    age = self.process_age(df)  

    pclass = self.process_pclass(df)

    

    if self.model in [0,1,2,4,5,6,7,8,9]:

      return pclass

    elif self.model == 3:

      return pd.concat([pclass,age],axis=1)

    else:

      return None
# for validation purposes only

select = FeatureSelector(train.drop(labels=["Survived"],axis=1).select_dtypes(include=["int64","float64"]).columns).transform(train)



# change model to 0,1,2,3, ..., 7

model = NumericalTransformer(model=9)

df = model.transform(select)
# global varibles

seed = 42

num_folds = 10

scoring = {'Accuracy': make_scorer(accuracy_score)}
# load the datasets

train = pd.read_csv("/kaggle/input/titanic/train.csv")



# split-out train/validation and test dataset

X_train, X_test, y_train, y_test = train_test_split(train.drop(labels="Survived",axis=1),

                                                    train["Survived"],

                                                    test_size=0.20,

                                                    random_state=seed,

                                                    shuffle=True,

                                                    stratify=train["Survived"])
# Categrical features to pass down the categorical pipeline 

categorical_features = X_train.select_dtypes(include=["object"]).columns



# Numerical features to pass down the numerical pipeline 

numerical_features = X_train.select_dtypes(include=["int64","float64"]).columns



# Defining the steps in the categorical pipeline 

categorical_pipeline = Pipeline(steps = [('cat_selector', FeatureSelector(categorical_features)),

                                         ('cat_transformer', CategoricalTransformer(model=9))

                                         ]

                                )

# Defining the steps in the numerical pipeline     

numerical_pipeline = Pipeline(steps = [('num_selector', FeatureSelector(numerical_features)),

                                       ('num_transformer', NumericalTransformer(model=9)) 

                                       ]

                              )



# Combining numerical and categorical piepline into one full big pipeline horizontally 

# using FeatureUnion

full_pipeline_preprocessing = FeatureUnion(transformer_list = [('categorical_pipeline', categorical_pipeline),

                                                               ('numerical_pipeline', numerical_pipeline)

                                                               ]

                                           )
# for validate purposes

new_data = full_pipeline_preprocessing.fit_transform(X_train)

new_data_df = pd.DataFrame(new_data,)#columns=cat_cols_final.tolist() + num_cols_final.tolist())

new_data_df.head()
"""

    model: 

      - 0: Sex column (categorized), Pclass (raw)

      - 1: Sex column (get_dummies(drop_first=True)), Pclass (raw)

      - 2: Sex column (get_dummies(drop_first=True)), Pclass (get_dummies(drop_first=False))

      - 3: Sex column (get_dummies(drop_first=True)), Pclass (raw), Age (get_dummies(drop_first=False))

      - 4: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (categorized)

      - 5: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))

      - 6: New Sex column (get_dummies(drop_first=False)), Pclass (raw), Embarked (get_dummies(drop_first=False))

      - 7: New Sex column (get_dummies(drop_first=False)+drop(Sex_woman)), Pclass (raw), Embarked (get_dummies(drop_first=False))

      - 8: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size

      - 9: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size, Embarked (get_dummies(drop_first=False))

"""



# The full pipeline as a step in another pipeline with an estimator as the final step

pipe = Pipeline(steps = [('full_pipeline', full_pipeline_preprocessing),

                         #("fs",SelectKBest()),

                         ("clf",XGBClassifier())])



# create a dictionary with the hyperparameters

search_space = [

                {"clf":[RandomForestClassifier()],

                 "clf__n_estimators": [100],

                 "clf__criterion": ["entropy"],

                 "clf__max_leaf_nodes": [64],

                 "clf__random_state": [seed]

                 },

                {"clf":[LogisticRegression()],

                 "clf__solver": ["liblinear"]

                 },

                {"clf":[XGBClassifier()],

                 "clf__n_estimators": [50,100],

                 "clf__max_depth": [4],

                 "clf__learning_rate": [0.001, 0.01,0.1],

                 "clf__random_state": [seed],

                 "clf__subsample": [1.0],

                 "clf__colsample_bytree": [1.0],

                 "full_pipeline__numerical_pipeline__num_transformer__model":[9],

                 "full_pipeline__categorical_pipeline__cat_transformer__model":[9]

                 }

                ]



# create grid search

kfold = StratifiedKFold(n_splits=num_folds,random_state=seed)



# return_train_score=True

# official documentation: "computing the scores on the training set can be

# computationally expensive and is not strictly required to

# select the parameters that yield the best generalization performance".

grid = GridSearchCV(estimator=pipe, 

                    param_grid=search_space,

                    cv=kfold,

                    scoring=scoring,

                    return_train_score=True,

                    n_jobs=-1,

                    refit="Accuracy")



tmp = time.time()



# fit grid search

best_model = grid.fit(X_train,y_train)



print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))
print("Best: %f using %s" % (best_model.best_score_,best_model.best_params_))
result = pd.DataFrame(best_model.cv_results_)

result.head()
result_acc = result[['mean_train_Accuracy', 'std_train_Accuracy','mean_test_Accuracy', 'std_test_Accuracy','rank_test_Accuracy']].copy()

result_acc["std_ratio"] = result_acc.std_test_Accuracy/result_acc.std_train_Accuracy

result_acc.sort_values(by="rank_test_Accuracy",ascending=True)
# best model

predict_first = best_model.best_estimator_.predict(X_test)

print(accuracy_score(y_test, predict_first))