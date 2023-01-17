import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

df = pd.read_csv('../input/_visagio-hackathon_/database_fires.csv') 

df.drop(['id'], axis=1, inplace=True)
target = df['fires'].copy()

df.drop('fires', axis=1, inplace=True)
class EstadosEmRegiao(BaseEstimator, TransformerMixin):

    def __init__(self, toNumbers=True):

        self.toNumbers = toNumbers

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        def getRegiao(estado):

            if (estado in ['AM', 'RR', 'AC', 'RO', 'PA', 'TO', 'AP']):

                return 1

            elif (estado in ['BA', 'SE', 'AL', 'PE', 'PB', 'RN', 'CE', 'MA', 'PI']):

                return 2

            elif (estado in ['GO', 'MT', 'MS', 'DF']):

                return 3        

            elif (estado in ['PR', 'SC', 'RS']):

                return 4

            elif (estado in ['ES', 'MG', 'SP', 'RJ']):

                return 5

        X['regioes'] = X['estado'].map(getRegiao);

        return X.drop('estado', axis=1)

    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)
#trocar as datas completas pelo mês apenas

import re

class DataEmMes(BaseEstimator, TransformerMixin):

    def __init__(self, toDf=False):

        self.toDf = toDf        

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        mes = X['data'].map(lambda x: int(re.search('/(.+?)/', x).group(1)))

        X['mes'] = mes

        return X.drop('data', axis=1)

    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)
from sklearn.pipeline import Pipeline



cat_features = ['estado','estacao','data']

cat_transformer = Pipeline([

    ('emMes', DataEmMes()),

    ('emRegiao', EstadosEmRegiao()),

    ('oneHot', OneHotEncoder())

])



num_features = ['precipitacao','temp_max','temp_min','insolacao','evaporacao_piche',

 'temp_comp_med','umidade_rel_med','vel_vento_med','altitude']

num_transformer = Pipeline([

    ('imputer', IterativeImputer(max_iter=20)),

    ('scaler', StandardScaler()),

])



preprocessor = ColumnTransformer([

    ('cat', cat_transformer, cat_features),

    ('num', num_transformer, num_features)

])





df_tr = preprocessor.fit_transform(df)
#Separando Df de treino e df de teste

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_tr, target, test_size=0.2, random_state=42)
from sklearn.svm import SVC

classificador  = SVC(kernel = 'rbf', C = 5)

classificador.fit(X_train, y_train)

classificador.score(X_test, y_test)
respostas = pd.read_csv('../input/_visagio-hackathon_/respostas.csv')

respId = respostas['id'].copy()

respostas.drop('id', axis=1, inplace=True)

resp_tr = preprocessor.fit_transform(respostas)

resp_tr
gabarito = classificador.predict(resp_tr)
final = pd.Series(gabarito, index=respId ,dtype= 'int32')

final.to_csv('SVC.csv', header=False)