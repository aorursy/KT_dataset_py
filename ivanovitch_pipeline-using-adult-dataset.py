!pip install --upgrade scikit-learn
import sklearn
sklearn.__version__
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

# Rich visual representation of estimators (new 0.23.2)
from sklearn import set_config
set_config(display='diagram')

# columns used 
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 
           'sex','capital_gain', 'capital_loss', 'hours_per_week',
           'native_country','high_income']
# importing the dataset
income = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                   header=None,
                   names=columns)
income.head()
# There are duplicated rows
income.duplicated().sum()
# Delete duplicated rows
income.drop_duplicates(inplace=True)
income.duplicated().sum()
# Verify if columns[int64] has outliers (with data leakage!!!!!!!)

# data
x = income.select_dtypes("int64")

# identify outlier in the dataset
lof = LocalOutlierFactor()
outlier = lof.fit_predict(x)
mask = outlier != -1

print("Income shape [original]: {}".format(income.shape))
print("Income shape [outlier removal]: {}".format(income.loc[mask,:].shape))

# income with outliner
income_w = income.loc[mask,:].copy()
income_w.head()
# define a categorical encoding for target variable
le = LabelEncoder()

# fit and transoform y_train
income_w["high_income"] = le.fit_transform(income_w.high_income)
le.classes_
income_w.head()
#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self.feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self.feature_names ]
# Handling categorical features 
class CategoricalTransformer( BaseEstimator, TransformerMixin ):
  # Class constructor method that takes one boolean as its argument
  def __init__(self, new_features=True):
    self.new_features = new_features
    self.colnames = None

  #Return self nothing else to do here    
  def fit( self, X, y = None ):
    return self 

  def get_feature_names(self):
        return self.colnames.tolist()

  # Transformer method we wrote for this transformer 
  def transform(self, X , y = None ):
    df = X.copy()

    # customize feature?
    # how can I identify this one? EDA!!!!
    if self.new_features: 
      
      # minimize the cardinality of native_country feature
      df.loc[df['native_country']!=' United-States','native_country'] = 'non_usa' 

      # replace ? with Unknown
      edit_cols = ['native_country','occupation','workclass']
      for col in edit_cols:
        df.loc[df[col] == ' ?', col] = 'unknown'

      # decrease the cardinality of education feature
      hs_grad = [' HS-grad',' 11th',' 10th',' 9th',' 12th']
      elementary = [' 1st-4th',' 5th-6th',' 7th-8th']
      # replace
      df['education'].replace(to_replace = hs_grad,value = 'HS-grad',inplace = True)
      df['education'].replace(to_replace = elementary,value = 'elementary_school',inplace = True)

      # adjust marital_status feature
      married= [' Married-spouse-absent',' Married-civ-spouse',' Married-AF-spouse']
      separated = [' Separated',' Divorced']
      # replace 
      df['marital_status'].replace(to_replace = married ,value = 'Married',inplace = True)
      df['marital_status'].replace(to_replace = separated,value = 'Separated',inplace = True)

      # adjust workclass feature
      self_employed = [' Self-emp-not-inc',' Self-emp-inc']
      govt_employees = [' Local-gov',' State-gov',' Federal-gov']
      # replace elements in list.
      df['workclass'].replace(to_replace = self_employed ,value = 'Self_employed',inplace = True)
      df['workclass'].replace(to_replace = govt_employees,value = 'Govt_employees',inplace = True)

    # update column names
    self.colnames = df.columns      
  
    return df
# 
# for validation purposes
#

#model = FeatureSelector(income_w.select_dtypes("object").columns.to_list())
#df = model.fit_transform(income_w)
#df.head()
# 
# for validation purposes
#

#model = CategoricalTransformer(new_features=True)
#df_cat = model.fit_transform(df)
#df_cat.head()
# check the cardinality before and after transformation
#income_w.select_dtypes("object").apply(pd.Series.nunique)
# check the cardinality before and after transformation
#df_cat.apply(pd.Series.nunique)
# transform numerical features
class NumericalTransformer( BaseEstimator, TransformerMixin ):
  # Class constructor method that takes a model parameter as its argument
  # model 0: minmax
  # model 1: standard
  # model 2: without scaler
  def __init__(self, model = 0):
    self.model = model
    self.colnames = None

  #Return self nothing else to do here    
  def fit( self, X, y = None ):
    return self

  # return columns names after transformation
  def get_feature_names(self):
        return self.colnames 
        
  #Transformer method we wrote for this transformer 
  def transform(self, X , y = None ):
    df = X.copy()
    
    # update columns name
    self.colnames = df.columns.tolist()
    
    # minmax
    if self.model == 0: 
      scaler = MinMaxScaler()
      # transform data
      df = scaler.fit_transform(df)
    elif self.model == 1:
      scaler = StandardScaler()
      # transform data
      df = scaler.fit_transform(df)
    else:
      df = df.values

    return df
# 
# for validation purposes
#

#model = FeatureSelector(income_w.select_dtypes("int64").columns.to_list()[:-1])
#df = model.fit_transform(income_w)
#df.head()
# 
# for validation purposes
#

#model = NumericalTransformer(model=2)
#df_cat = model.fit_transform(df)
#df_cat
# split-out train/validation and test dataset
X_train, X_test, y_train, y_test = train_test_split(income_w.drop(labels="high_income",axis=1),
                                                    income_w["high_income"],
                                                    test_size=0.20,
                                                    random_state=41,
                                                    shuffle=True,
                                                    stratify=income_w["high_income"])
# Categrical features to pass down the categorical pipeline 
categorical_features = X_train.select_dtypes("object").columns.to_list()

# Numerical features to pass down the numerical pipeline 
numerical_features = X_train.select_dtypes("int64").columns.to_list()

# Defining the steps in the categorical pipeline 
categorical_pipeline = Pipeline(steps = [('cat_selector', FeatureSelector(categorical_features)),
                                         ('cat_transformer', CategoricalTransformer()),
                                         ('cat_encoder','passthrough')
                                         #('cat_encoder',OneHotEncoder(sparse=False,drop="first"))
                                         ]
                                )

# Defining the steps in the numerical pipeline     
numerical_pipeline = Pipeline(steps = [('num_selector', FeatureSelector(numerical_features)),
                                       ('num_transformer', NumericalTransformer()) 
                                       ]
                              )

# Combining numerical and categorical piepline into one full big pipeline horizontally 
# using FeatureUnion
full_pipeline_preprocessing = FeatureUnion(transformer_list = [('cat_pipeline', categorical_pipeline),
                                                               ('num_pipeline', numerical_pipeline)
                                                               ]
                                           )
# 
# for validate purposes
#

#new_data = full_pipeline_preprocessing.fit_transform(X_train)
#catnames = full_pipeline_preprocessing.get_params()["cat_pipeline"][2].get_feature_names().tolist()
#numnames = full_pipeline_preprocessing.get_params()["num_pipeline"][1].get_feature_names()
#df = pd.DataFrame(new_data,columns = catnames + numnames)
#df.head()
#df.shape
# global varibles
seed = 15
num_folds = 10
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
# See documentation for more info
# https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py

# The full pipeline 
pipe = Pipeline(steps = [('full_pipeline', full_pipeline_preprocessing),
                         ("fs",SelectKBest()),
                         ("classifier",DecisionTreeClassifier())])
# create a dictionary with the hyperparameters
search_space = [{"classifier":[DecisionTreeClassifier()],
                 "classifier__criterion": ["entropy"],  #["gini","entropy"],
                 "classifier__splitter": ["best"],
                 "fs__k":[15],  # [10,15,20] 
                 "fs__score_func": [chi2],  #[f_classif, mutual_info_classif, chi2],
                 "full_pipeline__cat_pipeline__cat_encoder": [OneHotEncoder(sparse=False,drop="first")], #[OneHotEncoder(sparse=False,drop="first"),
                                                            #OrdinalEncoder()],
                 "full_pipeline__cat_pipeline__cat_transformer__new_features":[True],
                 "full_pipeline__num_pipeline__num_transformer__model": [0]}, #[0,2]},
                {"classifier": [KNeighborsClassifier()],
                "classifier__n_neighbors": [3],
                 "full_pipeline__cat_pipeline__cat_encoder":[OneHotEncoder(sparse=False,drop="first")]}]

# create grid search
kfold = StratifiedKFold(n_splits=num_folds,random_state=seed,shuffle=True)

# see other scoring
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
grid = GridSearchCV(estimator=pipe, 
                    param_grid=search_space,
                    cv=kfold,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1,
                    refit="Accuracy")

# fit grid search
all_models = grid.fit(X_train,y_train)
all_models
print("Best: %f using %s" % (all_models.best_score_,all_models.best_params_))
result = pd.DataFrame(all_models.cv_results_)
result
# Just change column name from "test" to "validation" for not confuse
result_auc = result[['mean_train_AUC', 'std_train_AUC','mean_test_AUC', 'std_test_AUC',"rank_test_AUC"]].copy()
for col in result_auc.columns:
  result_auc.rename(columns={col:col.replace("test","validation")}, inplace=True)
result_auc
# Just change column name from "test" to "validation" for not confuse
result_acc = result[['mean_train_Accuracy', 'std_train_Accuracy','mean_test_Accuracy', 'std_test_Accuracy',"rank_test_Accuracy"]].copy()
for col in result_acc.columns:
  result_acc.rename(columns={col:col.replace("test","validation")}, inplace=True)
result_acc
# final model
predict = all_models.predict(X_test)
# confusion matrix (we change the way to make equal to slides)
#             true label
#               1     0     
# predict  1    TP    FP
#          0    FN    TN
#

confusion_matrix(predict,y_test,
                 labels=[1,0])
print(accuracy_score(y_test, predict))
print(classification_report(y_test,predict))
fig, ax = plt.subplots(1,1,figsize=(7,4))

ConfusionMatrixDisplay(confusion_matrix(predict,y_test,labels=[1,0]),
                       display_labels=[">50k","<=50k"]).plot(values_format=".0f",ax=ax)

ax.set_xlabel("True Label")
ax.set_ylabel("Predicted Label")
plt.show()
# columns used in the model (k columns)
features = all_models.best_estimator_.named_steps['fs']
features.get_support()
# All information is trackable going back in the Pipeline
# categorical columns
features_full = all_models.best_estimator_.named_steps['full_pipeline']
features_cat = features_full.get_params()["cat_pipeline"]
features_cat[2].get_feature_names()
features_full
# numerical columns
features_full.get_params()["num_pipeline"][1].get_feature_names()
all_columns = features_cat[2].get_feature_names().tolist() + features_full.get_params()["num_pipeline"][1].get_feature_names()
all_columns
selected_columns = [value for (value, filter) in zip(all_columns, features.get_support()) if filter]
selected_columns
fig, ax = plt.subplots(1,1,figsize=(12,8))

xticks = [x for x in range(len(features.scores_))] 
ax.bar(xticks, features.scores_)
ax.set_xticks(xticks)
ax.set_xticklabels(all_columns,rotation=90)
#ax.set_xticks(ticks=xticks, labels=all_columns,rotation=90,fontsize=15)
ax.set_title("Feature Importance")
plt.show()
all_models.best_estimator_.named_steps['classifier']
from sklearn.tree import plot_tree # to draw a classification tree
plt.figure(figsize=(15, 7.5))
plot_tree(all_models.best_estimator_.named_steps['classifier'], 
          filled=True, 
          rounded=True, 
          class_names=["<=50k", ">50k"],
          feature_names=all_columns)
# Save the model using joblib
with open('pipe.joblib', 'wb') as file:
  joblib.dump(all_models, file)
# Under the production environment [joblib]
with open('pipe.joblib', 'rb') as file:
  model = joblib.load(file)

# final model
predict = model.predict(X_test)
print(accuracy_score(y_test, predict))
print(classification_report(y_test,predict))