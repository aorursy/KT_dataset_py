import pandas as pd

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

#from google.colab import files

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

import numpy as np


test = pd.read_csv("../input/test.csv")

titanic_public = pd.read_csv("../input/titanic_public.csv")

train = pd.read_csv("../input/train.csv")
aux = train.copy()



# fill missing values with -0.5

aux["Age"] = aux["Age"].fillna(-0.5)



# divide age column into a range of values

cut_points = [-1,0,5,12,18,35,60,100]

label_names = ["Missing","Infant","Child",

               "Teenager","Young Adult","Adult","Senior"]

aux["Age_categories"] = pd.cut(aux["Age"],

                                 cut_points,

                                 labels=label_names)

def create_dummies(df,column_name):

    # drop_first = True to avoid colinearity

    dummies = pd.get_dummies(df[column_name],

                             prefix=column_name,

                             drop_first=True)

    df = pd.concat([df,dummies],axis=1)

    return df



aux = create_dummies(aux,"Pclass")

aux = create_dummies(aux,"Age_categories")

aux = create_dummies(aux,"Sex")
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

    self.model = model



  #Return self nothing else to do here    

  def fit( self, X, y = None ):

    return self 



  def create_dummies(self, df, column_name, drop_first_col):

    """Create Dummy Columns from a single Column

    """

    dummies = pd.get_dummies(df[column_name],prefix=column_name, drop_first=drop_first_col)

    return dummies

# extracting title information from the passenger's name

  def process_titles(self, df):

    df["Titles"] = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

    titles = {

        "Mr" :         0,

        "Mme":         1,

        "Ms":          1,

        "Mrs" :        1,

        "Master" :     2,

        "Mlle":        3,

        "Miss" :       3,

        "Capt":        4,

        "Col":         4,

        "Major":       4,

        "Dr":          4,

        "Rev":         4,

        "Jonkheer":    5,

        "Don":         5,

        "Sir" :        5,

        "Countess":    5,

        "Dona":        5,

        "Lady" :       5

    } 



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

    tit = self.process_titles(df)



    if self.model == 0:

      return  pd.concat([sex,tit],axis=1)  

    elif self.model in [1,2,3]:

      return sex

    elif self.model == 4:

      return pd.concat([sex,family_size,tit],axis=1)

    elif self.model in [4,5,6,7]:

      return pd.concat([sex,embarked,tit],axis=1)

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

cat_cols_final = df_cat.columns

df_cat.head()
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

num_cols_final = df.columns

pd.concat([df.select_dtypes(include=["int64","uint8"]),

                       df_cat.select_dtypes(include=["int64","uint8"]),

                       train.Survived],axis=1).corr()["Survived"].abs().sort_values()

df.head()
# global varibles

seed = 42

num_folds = 10

scoring = {'Accuracy': make_scorer(accuracy_score)}

# load the datasets

train = pd.read_csv("train.csv")



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

                                           )# for validate purposes

new_data = full_pipeline_preprocessing.fit_transform(X_train)

new_data_df = pd.DataFrame(new_data,)#columns=cat_cols_final.tolist() + num_cols_final.tolist())

new_data_df.head()




# The full pipeline as a step in another pipeline with an estimator as the final step

pipe = Pipeline(steps = [('full_pipeline', full_pipeline_preprocessing),

                         #("fs",SelectKBest()),

                         ("clf",XGBClassifier())])



# create a dictionary with the hyperparameters

search_space = [

                {"clf":[RandomForestClassifier()],

                 "clf__n_estimators": [100, 200],

                 "clf__criterion": ["entropy"],

                 "clf__max_leaf_nodes": [64, 128, 256],

                 "clf__random_state": [seed]

                 },

                {"clf":[LogisticRegression()],

                 "clf__solver": ["liblinear"]

                 },

                {"clf":[XGBClassifier()],

                 "clf__n_estimators": [200, 300],

                 "clf__max_depth": [2,3,4],

                 "clf__booster":['gbtree'],

                 "clf__learning_rate": [0.001, 0.01,0.1],

                 "clf__random_state": [seed],

                 "clf__subsample": [1.0],

                 "clf__objective":['reg:squarederror'],

                 "clf__tree_method":["auto"],

                 "clf__colsample_bytree": [1.0],

                 "full_pipeline__numerical_pipeline__num_transformer__model":[0],

                 "full_pipeline__categorical_pipeline__cat_transformer__model":[6]

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
print("Best: %f using %s" % (best_model.best_score_,best_model.best_params_))
# best model

predict_first = best_model.best_estimator_.predict(X_test)

print(accuracy_score(y_test, predict_first))
predict_final = best_model.best_estimator_.predict(test)

holdout_ids = test["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": predict_final}

submission = pd.DataFrame(submission_df)



submission.to_csv("submission.csv",index=False)

files.download('submission.csv')