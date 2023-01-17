!pip install pandas_profiling==2.7.1
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

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

import numpy as np

import datetime

from sklearn.model_selection import ParameterGrid

import pandas_profiling
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print("Train size: {0}\nTest size: {1}".format(train.shape,test.shape))
train.head()
train.info()
test.info()
pandas_profiling.ProfileReport(train)
msno.matrix(train,figsize=(10,10))

train.isnull().sum()/train.shape[0]
#Vamos preencher os valores faltantes de 'Embarked' com o valor mais comum:

print(train['Embarked'].describe())

train['Embarked'].fillna('S',inplace=True)
aux = train.copy()

aux["Sex"] = pd.Categorical(aux.Sex).codes

aux["Embarked"] = pd.Categorical(aux.Embarked).codes
plt.figure(figsize = (8,8))

sns.heatmap(aux.corr(),cmap=("RdBu_r"),annot=True,fmt='.2f')

plt.xticks(rotation=45) 

plt.show()
train.Age.describe()
def plot_median(data,**kwargs):

    m = data.median()

    plt.axvline(m, **kwargs,)



g = sns.FacetGrid(train, 

                  col="Pclass", 

                  row="Survived", hue="Sex",margin_titles=True)

g.map(plot_median, 'Age', ls=":")

g.map(sns.kdeplot, "Age", shade=True).add_legend()



plt.xlim(0,100)

plt.show()
aux = train.copy()



# preenche os valores faltantes com -0.5

aux["Age"].fillna(-0.5,inplace=True)



# divide a coluna age em faixas de idade

cut_points = [-1,0,5,12,18,35,60,100]

label_names = ["Missing","Infant","Child",

               "Teenager","Young Adult","Adult","Senior"]

aux["Age_categories"] = pd.cut(aux["Age"],

                                 cut_points,

                                 labels=label_names)
pivot = aux.pivot_table(index="Age_categories",values='Survived')

pivot.plot.bar()

plt.show()
# Create dimensions

age_cat_dim = go.parcats.Dimension(values=aux.Age_categories, label="Age")



gender_dim = go.parcats.Dimension(values=train.Sex, label="Gender")



class_dim = go.parcats.Dimension(

    values=train.Pclass,

    categoryorder='category ascending', label="Class"

)



survival_dim = go.parcats.Dimension(

    values=train.Survived, label="Outcome", categoryarray=[0, 1], 

    ticktext=['faleceu', 'sobreviveu']

)



# Create parcats trace

color = train.Survived

colorscale = [[0, 'lightsteelblue'], [1, 'mediumseagreen']]



fig = go.Figure(data = [go.Parcats(dimensions=[gender_dim,class_dim, age_cat_dim,survival_dim],

        line={'color': color, 'colorscale': colorscale},

        hoveron='color', hoverinfo='count+probability',

        labelfont={'size': 18, 'family': 'Times'},

        tickfont={'size': 16, 'family': 'Times'},bundlecolors=True, 

        arrangement='freeform')])

fig.update_layout(width=800,height=500)



fig.show()
def isChild_or_Woman(sex, age):

  if sex == "female":

    return 1

  elif age <13:

    return 1

  else:

      return 0;

aux["boyorwoman"] = aux.apply(lambda x: isChild_or_Woman(x["Sex"],x["Age"]),axis=1)

aux.head()
# Função para adicionar colunas dummy no dataset

def create_dummies(df,column_name):

    # drop_first = True para evitar colinearidade

    dummies = pd.get_dummies(df[column_name],

                             prefix=column_name,

                             drop_first=True)

    df = pd.concat([df,dummies],axis=1)

    return df
aux = create_dummies(aux,"Pclass")

aux = create_dummies(aux,"Age_categories")

aux = create_dummies(aux,"Sex")

aux.head()
#Extrai colunas passadas como argumento no construtor 

class FeatureSelector(BaseEstimator, TransformerMixin):

  #Construtor 

  def __init__( self, feature_names ):

    self.feature_names = feature_names 

    

  #Retorna self

  def fit( self, X, y = None ):

    return self 

    

  #Executa a transformação

  def transform(self, X, y = None):

    return X[self.feature_names]
#Adiciona algumas colunas categóricas

class CategoricalTransformer(BaseEstimator, TransformerMixin):

  #Retorna Self

  def fit( self, X, y = None ):

    return self 



  def create_dummies(self, df, column_name, drop_first_col):

    """Create Dummy Columns from a single Column

    """

    dummies = pd.get_dummies(df[column_name],prefix=column_name, drop_first=drop_first_col)

    return dummies



  def process_sex(self, df):   

      sex_dummies = self.create_dummies(df,"Sex",True)

      return sex_dummies



  def process_embarked(self, df):

      df["Embarked"].fillna("S",inplace=True)

      embarked_dummies = self.create_dummies(df,"Embarked",False)

      return embarked_dummies



  def process_familyName(self, df):

      df["Family"] = df["Name"].str.extract('([A-Za-z]+)\,',expand=False)

      return pd.DataFrame(df["Family"],columns=["Family"])  



  #Executa a transformação

  def transform(self, X , y = None ):

    df = X.copy()



    # Processa os campos

    sex = self.process_sex(df)

    embarked = self.process_embarked(df)  

    family = self.process_familyName(df)  

    return pd.concat([sex, embarked],axis=1)
categorical_features = ['Name', 'Sex', 'Cabin', 'Embarked']

select = FeatureSelector(categorical_features).transform(train)

model = CategoricalTransformer()

model.transform(select)
#Converte os campos numéricos em categóricos

class NumericalTransformer(BaseEstimator, TransformerMixin):

  #Retorna Self

  def fit( self, X, y = None ):

    return self 



  def create_dummies(self, df, column_name, drop_first_col):

    """Create Dummy Columns from a single Column

    """

    dummies = pd.get_dummies(df[column_name],prefix=column_name, drop_first=drop_first_col)

    return dummies



  def isChild_or_Woman(sex, age):

    if sex == "female":

      return 1

    elif age <13:

      return 1

    else:

      return 0;



  def process_boywoman(self, df):

    df["boyorwoman"] = df.apply(lambda x: isChild_or_Woman(x["Sex"],x["Age"]),axis=1)

    return pd.DataFrame(df["boyorwoman"],columns=["boyorwoman"])          



  def process_age(self,df):

      df["Age"].fillna(-0.5,inplace=True)



      # divide a coluna age em faixas de idade

      cut_points = [-1,0,5,12,18,35,60,100]

      label_names = ["Missing","Infant","Child",

                    "Teenager","Young Adult","Adult","Senior"]

      df["Age_categories"] = pd.cut(df["Age"],

                                cut_points,

                                labels=label_names)

      #return self.create_dummies(df,"Age_categories",False)  // Para testes

      return None

   

  def process_pclass(self, df):

      #return self.create_dummies(df,"Pclass",False)  // Para testes

       return pd.DataFrame(df["Pclass"],columns=["Pclass"])

        

  #Transformer method we wrote for this transformer 

  def transform(self, X , y = None ):

    df = X.copy()



    age = self.process_age(df)  

    pclass = self.process_pclass(df)    

    boywoman = self.process_boywoman(df)

    return pd.concat([pclass,age,boywoman],axis=1)
numerical_features = ['Pclass', 'Age', 'Sex']

select = FeatureSelector(numerical_features).transform(train)

model = NumericalTransformer()

model.transform(select)
# Variáveis Globais

seed = 42

num_folds = 10

scoring = {'Accuracy': make_scorer(accuracy_score)}
# Carregando o dataset

train = pd.read_csv("../input/train.csv")



# cria os dados de validação e treinamento

X_train, X_test, y_train, y_test = train_test_split(train.drop(labels="Survived",axis=1),

                                                    train["Survived"],

                                                    test_size=0.20,

                                                    random_state=seed,

                                                    shuffle=True,

                                                    stratify=train["Survived"])
# Definindo os passos do pipeline de campos categóricos

categorical_pipeline = Pipeline(steps = [('cat_selector', FeatureSelector(categorical_features)),

                                         ('cat_transformer', CategoricalTransformer())

                                         ]

                                )

# Definindo o pipeline de campos numéricos

numerical_pipeline = Pipeline(steps = [('num_selector', FeatureSelector(numerical_features)),

                                       ('num_transformer', NumericalTransformer()) 

                                       ]

                              )



# Combinando os dois num super-pipeline usando FeatureUnion

full_pipeline_preprocessing = FeatureUnion(transformer_list = [('categorical_pipeline', categorical_pipeline),

                                                               ('numerical_pipeline', numerical_pipeline)

                                                               ]

                                           )
new_data = full_pipeline_preprocessing.fit_transform(X_train)

new_data_df = pd.DataFrame(new_data,)

new_data_df.head()
# O pipeline completo é o pipeline com um estimator no final

pipe = Pipeline(steps = [('full_pipeline', full_pipeline_preprocessing),

                         #("fs",SelectKBest()),

                         ("clf",XGBClassifier())])



# Cria um dicionário com os parâmetros

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

                 "clf__colsample_bytree": [1.0]

                 }

                ]



pg = ParameterGrid(search_space)

print("Pipeline will run {qtd} combinations".format(qtd=len(pg)))



# Cria o grid search

kfold = StratifiedKFold(n_splits=num_folds)



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



print('Training: begins at {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))   

time_1 = time.time()

# fit grid search

best_model = grid.fit(X_train,y_train)

print('Training: ends at {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))   

print('Duration: {}s'.format(time.time() - time_1))
print("Best: %f using %s" % (best_model.best_score_,best_model.best_params_))
result = pd.DataFrame(best_model.cv_results_)

result_acc = result[['mean_train_Accuracy', 'std_train_Accuracy','mean_test_Accuracy', 'std_test_Accuracy','rank_test_Accuracy']].copy()

result_acc["std_ratio"] = result_acc.std_test_Accuracy/result_acc.std_train_Accuracy

result_acc.sort_values(by="rank_test_Accuracy",ascending=True)
# Melhor modelo

predict_first = best_model.best_estimator_.predict(X_test)

print(accuracy_score(y_test, predict_first))
predict_final = best_model.best_estimator_.predict(test)
holdout_ids = test["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": predict_final}

submission = pd.DataFrame(submission_df)



submission.to_csv("submission.csv",index=False) 