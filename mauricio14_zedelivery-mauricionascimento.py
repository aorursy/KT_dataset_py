!pip install ppscore

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import ppscore as pps

import optuna



import random



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import KFold, train_test_split, cross_val_score

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import make_scorer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.cluster import KMeans
path = r'../input/walmart-recruiting-store-sales-forecasting/'



train = pd.read_csv(os.path.join(path, 'train.csv.zip'))

test = pd.read_csv(os.path.join(path, 'test.csv.zip'))

stores = pd.read_csv(os.path.join(path, 'stores.csv'))

features = pd.read_csv(os.path.join(path, 'features.csv.zip'))
# Juntando os Dataframes de treino e teste para facilitar a manipulação

train['df_label'] = 'train'

test['df_label'] = 'test'



full_df = pd.concat([train, test], axis=0)
# Agregando ao Dataframe principal variáveis de store

full_df = pd.merge(

    full_df,

    stores,

    how='left',

    on='Store'

)



# Agregando ao Dataframe principal variáveis de features

full_df = pd.merge(

    full_df,

    features,

    how='left',

    on=[

        'Store',

        'Date',

        'IsHoliday'

    ]

)
# Análise dos tipos de dados

full_df.dtypes
# Coluna date está como object

# Necessário transformá-la para datetime

full_df['Date'] = pd.to_datetime(full_df['Date'])
# Visualizando, de forma ordenada, as datas presentes

# no dataframe

full_df['Date'].drop_duplicates().sort_values()
# Visualizando dias da semana presentes no dataset

full_df['Date'].dt.weekday.value_counts()
# Cria variável de semana do ano

full_df['WeekOfYear'] = full_df.Date.dt.isocalendar().week



# Cria variável referente ao ano

full_df['Year'] = full_df.Date.dt.year



# Cria variável referente ao dia do mês

full_df['Day'] = full_df.Date.dt.day



# Cria variável referente à semana corrida

full_df['WeekSeq'] = full_df['WeekOfYear'] + 52*(full_df['Year'] - full_df['Year'].min())
# Visualizando o dataset

full_df.head()
# Visualizando o dataset

display(full_df.head())

display(full_df.tail())
# Verificando estatísticas das variáveis do Dataframe completo

full_df.describe().T
# Separando o dataframe em treino e teste para que possamos

# entender analisar o comportamento da variável target

train = full_df.loc[

    full_df['df_label'] == 'train'

].drop(['df_label'], axis=1)



test = full_df.loc[

    full_df['df_label'] == 'test'

].drop(['df_label'], axis=1)
def EDA(df, col):

    

    '''

    Função para auxiliar na Análise Exploratória através

    de plots.

    

    Para variável target:

        Plota distribuição e BoxPlot da variável além de

        plotar a média e mediana de vendas ao longo das

        semanas do ano

        

    Para variáveis categóricas:

        Plota distribuição da variável target para cada uma das categorias,

        além de BoxPlot da variável target em função das categorias

        

    Para variáveis numéricas:

        Plota distribuição e BoxPlot da variável

    

    '''

    if col != 'Weekly_Sales':

        fig,ax = plt.subplots(1, 2, figsize=(15, 6));

    

    if col in ['IsHoliday', 'Type']:

        

        if col == 'IsHoliday':

            sflier=True

        else:

            sflier=False

        

        for val in df[col].unique():

            ax[0].hist(

                df['Weekly_Sales'].loc[

                    df[col] == val

                ],

                bins=50,

                alpha=0.6

            )

            ax[0].legend(df[col].unique())

        ax[0].set_title(f'Distribuição de Weekly Sales por {col}')

        ax[0].set_xlim([0, 200000])



        sns.boxplot(

            x=col,

            y='Weekly_Sales',

            data=df,

            orient='v',

            showfliers=sflier

        )

        ax[1].set_title(f'Boxplot de {col}')

        

    elif col == 'Weekly_Sales':

    

        ax = {}

        fig = plt.figure(figsize=(15, 12))

        grid = plt.GridSpec(3, 2, hspace=0.4, wspace=0.1)

        ax[0] = plt.subplot(grid[0, 0])

        ax[1] = plt.subplot(grid[0, 1])

        ax[2] = plt.subplot(grid[1, :])





        ax[0].hist(df[col], bins=50);

        ax[0].set_title(f'Histograma de {col}')

        ax[0].set_xlim([0, 200000])



        sns.boxplot(df[col], ax=ax[1], orient='h');

        ax[1].set_title(f'Boxplot de {col}');



        mean_sales = df.groupby('WeekOfYear', as_index=False)[col].mean()

        median_sales = df.groupby('WeekOfYear', as_index=False)[col].median()



        ax[2].plot(mean_sales['WeekOfYear'], mean_sales[col], 'o-')

        ax[2].plot(median_sales['WeekOfYear'], median_sales[col], 'o-')

        ax[2].set_xticks(range(1, 53))

        ax[2].legend(['mean', 'median'])

        ax[2].set_title(f'Distribuição de {col} ao longo das semanas')

        plt.show()

        

    else:

        

        ax[0].hist(df[col], bins=50);

        ax[0].set_title(f'Histograma de {col}')



        sns.boxplot(df[col], ax=ax[1], orient='v');

        ax[1].set_title(f'Boxplot de {col}');
for col in train.columns[3: -4]:

    EDA(train, col)
# Dataset de treino

nulls = pd.DataFrame(

    round(train.isnull().sum()/train.shape[0], 3),

    columns=[

        'Null Percent'

    ]

)

nulls
nulls_test = pd.DataFrame(

    round(test.isnull().sum()/test.shape[0], 3),

    columns=[

        'Null Percent'

    ]

)

nulls_test
# CPI



# Fixando semana de análise

week = 5



# Agrupando CPI por loja e contando valores 

# unicos de CPI

anl_CPI = full_df.loc[

    full_df['WeekSeq'] == week

].groupby(

    'Store',

    as_index=False

).agg(unique_values_CPI=('CPI', 'nunique'))



# Checando se algum dos valores é diferente de 1

(anl_CPI['unique_values_CPI'] != 1).any()
# Unemployment



# Fixando semana de análise

week = 5



# Agrupando Unemployment por loja e contando valores 

# unicos de Unemployment

anl_Unemployment = full_df.loc[

    full_df['WeekSeq'] == week

].groupby(

    'Store',

    as_index=False

).agg(unique_values_Unemployment=('Unemployment', 'nunique'))



# Checando se algum dos valores é diferente de 1

(anl_Unemployment['unique_values_Unemployment'] != 1).any()
def check_for_different_values(df, col):

    '''

    Percorre as semanas de df verificando se, em alguma,

    o indicador col assume diferentes valores para uma mesma loja,

    em alguma delas.

    '''

    

    # Selecionando semanas em que o indicador não é nulo

    weeks_not_null = df.loc[

        ~df[col].isnull(), 'WeekSeq'

    ].unique()

    

    # Para cada semana, checamos se a quantidade de valores

    # unicos do indicador em alguma das lojas é diferente de 1

    for week in weeks_not_null:



        store_week = df.loc[

        (df['WeekSeq'] == week)

        ]



        store_week_grpd = store_week.groupby('Store').agg(

        nunique=(col, 'nunique')

        )



        if (store_week_grpd['nunique'] != 1).any():

            print(

                f'Valores diferentes para {col} na semana {week}'

            )

            break

        

    print(f'{col} não varia de loja para loja!')
cols = ['CPI', 'Unemployment']



# Testa se para CPI e Unemployment os valores variam

# dentro de alguma loja em alguma semana.

for col in cols:

    check_for_different_values(full_df, col)

# Selecionando 5 lojas aleatoriamente

def plot_indicator_over_weeks(df, indicators):

    '''

    Plota gráficos de linha ,para um conjunto de 5 lojas

    escolhidas aleatoriamente, de indicadores selecionados.

    '''

    

    random.seed(95)

    stores = random.sample(

        range(full_df['Store'].min(), full_df['Store'].max()+1),

        5,

    )



    fig, ax = plt.subplots(len(cols), 1, figsize=(15, 7))



    for i in range(len(cols)):



        for store in stores:

            store_df = full_df.loc[

                full_df['Store'] == store

            ]

            store_df = store_df.groupby(['Store', 'WeekSeq'], as_index=False)[cols[i]].mean()



            ax[i].plot(store_df['WeekSeq'], store_df[cols[i]], '.-')

            ax[i].set_title(f'Variação de {cols[i]} ao longo das semanas')



    fig.legend(stores, loc='center right', ncol=1)



    plt.show()

    

    

plot_indicator_over_weeks(full_df, cols)
def input_nulls_rolling_window(df, window_size, cols, centrality_measures):

    '''

    Para os indicadores selecionados, preenche os nulos de cada loja

    com a respectiva medida de centralidade indicada, calculada sobre

    janela móvel de tamanho escolhido.

    '''

    

    for col in cols:

        for store in df['Store'].unique():

            aux = df.loc[

                df['Store'] == store

            ]



            last_week = aux.loc[

                ~aux[col].isnull(),'WeekSeq'

            ].max()

            end_week = aux.loc[

                aux[col].isnull(),'WeekSeq'

            ].max()



            for week in range(last_week+1, end_week+1):



                if centrality_measures[col] == 'mean':

                    RolMean = aux.loc[

                        aux['WeekSeq'].between(week-window_size, week-1)

                    ].groupby('Store')[col].mean()



                    df.loc[

                        (df['Store'] == store)

                        &(df['WeekSeq'] == week),

                        col

                    ] = df.loc[

                        (df['Store'] == store)

                        &(df['WeekSeq'] == week),

                        col

                    ].fillna(RolMean.values[0])





                elif centrality_measures[col] == 'median':



                    RolMedian = aux.loc[

                        aux['WeekSeq'].between(week-window_size, week-1)

                    ].groupby('Store')[col].median()



                    df.loc[

                        (df['Store'] == store)

                        &(df['WeekSeq'] == week),

                        col

                    ] = df.loc[

                        (df['Store'] == store)

                        &(df['WeekSeq'] == week),

                        col

                    ].fillna(RolMedian.values[0])



    return df
# Faz imput de nulos para CPI e Unemployment

full_df = input_nulls_rolling_window(

    df=full_df,

    window_size=52,

    cols=['CPI', 'Unemployment'],

    centrality_measures={

        'CPI' : 'mean',

        'Unemployment' : 'median'

    }

)
# Heatmap de correlações

plt.figure(figsize=(15, 8))

sns.heatmap(

    train.corr(),

    annot=True,

    fmt='.2f',

    cmap='Greens'

);
## PPS

pps.predictors(train, 'Weekly_Sales')
# Descartando variáveis mencionadas acima

full_df.drop([x for x in full_df.columns if 'MarkDown' in x], axis=1, inplace=True)

full_df.drop(['Fuel_Price'], axis=1, inplace=True)
# Visualizando o dataset

full_df.head()
# Separando dataframe nos dataset de treino e teste

train = full_df.loc[

    full_df['df_label'] == 'train'

].drop(['df_label'], axis=1)



test = full_df.loc[

    full_df['df_label'] == 'test'

].drop(['df_label'], axis=1)
# Definindo X e y, descartando variável de data

# já representada por outras variáveis.

X = train.drop(

    [

        'Date',

        'Weekly_Sales'

    ], axis=1

)

y = train['Weekly_Sales']
# Transformando variáveis categóricas em Dummy



cat_features = ['IsHoliday', 'Type']



for col in cat_features:



    enc = OneHotEncoder(sparse=False)

    transf_df = pd.DataFrame(

        enc.fit_transform(X[[col]]),

        columns=[col + '_' + str(x) for x in enc.categories_[0]]

    )

    X.drop(col, axis=1, inplace=True)

    X = pd.concat(

        [

            X,

            transf_df

        ],

        axis=1

    )
def WMAE_Func(y_true, y_pred, **kwargs):

    '''

    Função de erro customizada, que dá um peso maior para

    datas que são feriados.

    '''

    

    df = kwargs['df']

    df = df.loc[

        y_true.index

    ]

    df['Weights'] = 1

    df.loc[

        df['IsHoliday_True'] == 1,

        'Weights'

    ] == 5

    

    weights = df['Weights'].to_numpy()



    wmae = 1/np.sum(weights)*np.sum(weights*np.abs(y_true-y_pred))

    

    return -wmae
# Definindo scorer do scikit learn a partir de função

# de erro customizada

wmae_scorer = make_scorer(WMAE_Func, greater_is_better=False, df=X)
# Definindo objeto de validação cruzada



class TimeSeriesCV:

    '''

    Função para validação cruzada utilizando janelas móveis

    '''

    

    def __init__(self, nsplits, window_size, test_size, start_week):

        self.nsplits = nsplits

        self.start_week = start_week

        self.window_size = window_size

        self.test_size = test_size

        

    def split(self, X, y, groups=None):

        

        start_week = self.start_week

        end_week = X['WeekSeq'].max()

        test_size = self.test_size

        window_size = self.window_size

        nsplits = self.nsplits

        max_train_week = end_week - test_size

        step = (max_train_week - start_week - window_size)//(nsplits-1)



        for i in range(nsplits):



            week_ini_train = start_week + i*step

            week_final_train = week_ini_train + window_size

            week_final_test = week_final_train + test_size

            train_index = X.loc[

                (X['WeekSeq'] >= week_ini_train)

                & (X['WeekSeq'] < week_final_train)

            ].index.values



            test_index = X.loc[

                (X['WeekSeq'] >= week_final_train)

                & (X['WeekSeq'] < week_final_test)

            ].index.values

            

            yield train_index, test_index
# Criando objeto de validação cruzada

tscv = TimeSeriesCV(

    nsplits=6,

    window_size=52,

    test_size=38,

    start_week=X['WeekSeq'].min()

)
# Definindo modelo baseline

baseline = LinearRegression()
# Avaliando modelo baseline com base na metrica WMAE

scores = cross_val_score(

    baseline,

    X,

    y,

    scoring=wmae_scorer,

    cv=tscv

)



print(f'WMAE ao longo das folhas: {scores.round(2)}')

print('\n')

print(f'WMAE médio: {scores.mean().round(2)}')
# Criando variável com a data do Natal do ano em questão

full_df['Christmas'] = full_df['Year'].astype(str) + '-12-25'

full_df['Christmas'] = pd.to_datetime(full_df['Christmas'])
# Contando os dias até o Natal do ano vigente

full_df['Days_to_Christmas'] = (full_df['Christmas'] - full_df['Date']).dt.days
# Mapeando as semanas que contém feriado

weeks_holiday = full_df.loc[

    full_df['IsHoliday'] == True, 'WeekOfYear'

].unique()
# Criando variável com a distância (em semanas) até o próximo feriado

full_df['Weeks_To_Holiday'] = full_df['WeekOfYear'].apply(

    lambda x: min([week - x for week in weeks_holiday if week - x >= -1])

)
# Descartando data do natal

full_df.drop(['Christmas'], axis=1, inplace=True)
# Visualizando o dataset

full_df.head()
# Mapeando as semanas que contém feriado

weeks_holiday = full_df.loc[

    full_df['IsHoliday'] == True, 'WeekOfYear'

].unique()



# Criando dict que mapeia numero da semana à classe de feriado

holiday_dict = {}

i=1

for week in weeks_holiday:

    holiday_dict[week] = i

    i += 1



# Atribuindo aos nulos (semanas que não são feriado) a classe 0

full_df['Holiday_Clf'] = full_df['WeekOfYear'].map(holiday_dict).fillna(0).astype(int)
# Visualizando o dataset

full_df.head()
# Criando variavel com vendas semanais da mesma semana do ano anterior

full_df['y_lag52'] = full_df.groupby(['Store', 'Dept'])[['Weekly_Sales']].shift(52).values



# Para lojas/departamentos que ainda não existiam na referida semana

# do ano anterior (nulos), preenchemos com 0

full_df['y_lag52'] = full_df['y_lag52'].fillna(0)



# Desprezando o primeiro ano, para o qual o valor da variavel será nulo

full_df = full_df.loc[full_df['WeekSeq'] >= full_df['WeekSeq'].min() + 52].reset_index(drop=True)
# Visualizando o dataset

full_df.head()
class KMeansTransformerWeek(BaseEstimator, TransformerMixin):

    '''

    Agrupa e classifica semanas do ano com base na média de vendas.

    '''

    

    def __init__(self, num_clusters):

        self.num_clusters = num_clusters

        self.grped_weeks = pd.DataFrame()

        

    def fit(self, X, y):

        df = X.copy()

        df['Weekly_Sales'] = y.values

        grped_weeks = df.groupby(['WeekOfYear'], as_index=False)['Weekly_Sales'].mean()

        

        kmeans = KMeans(n_clusters=self.num_clusters)

        

        labels = kmeans.fit_predict(grped_weeks[['Weekly_Sales']])

        grped_weeks['Week_Clf'] = labels

        self.grped_weeks = grped_weeks

        

        return self

    

    def transform(self, X):

        

        X_ = X.copy()

        

        grped_weeks = self.grped_weeks

        

        X_ = pd.merge(

            X_,

            grped_weeks[

                [

                    'WeekOfYear',

                    'Week_Clf'

                ]

            ],

            how='left',

            on='WeekOfYear'  

        )

        

        X_['Week_Clf'] = X_['Week_Clf'].fillna(99)



        return X_
class KMeansTransformerDepts(BaseEstimator, TransformerMixin):

    

    def __init__(self, num_clusters):

        self.num_clusters = num_clusters

        self.grped_depts = pd.DataFrame()

        

    def fit(self, X, y):

        df = X.copy()

        df['Weekly_Sales'] = y.values

        

        grped_depts = df.groupby(

            [

                'Store',

                'Dept'

            ], as_index=False

        )['Weekly_Sales'].mean()

    

        kmeans = KMeans(n_clusters=self.num_clusters)

        labels = kmeans.fit_predict(grped_depts[['Weekly_Sales']])

        grped_depts['Dept_Clf'] = labels

        self.grped_depts = grped_depts



        return self

    

    def transform(self, X):



        X_ = X.copy()

        

        grped_depts = self.grped_depts

        

        X_ = pd.merge(

            X_,

            grped_depts[

                [

                    'Store',

                    'Dept',

                    'Dept_Clf'

                ]

            ],

            how='left',

            on=[

                'Store',

                'Dept'

            ]  

        )

        

        X_['Dept_Clf'] = X_['Dept_Clf'].fillna(99)

        return X_
# Definindo X e y

X = full_df.loc[

    full_df['df_label'] == 'train'

]



y = full_df.loc[

    full_df['df_label'] == 'train',

    'Weekly_Sales'

]
# Aqui definimos quais features usaremos e qual o tipo de cada uma delas

# As variáveis que não estão inclusas em num_cols nem em cat_cols serão descartadas

cols = X.columns.tolist()



cat_cols = [

    'Type',

    'Holiday_Clf']



num_cols = [

    'Size',

    'Temperature', 

    'CPI', 

    'Unemployment',

    'Year', 

    'Days_to_Christmas', 

    'Weeks_To_Holiday',

    'y_lag52'

]



# Pegando o identificador posicional de cada uma das colunas selecionadas

num_idx = [cols.index(x) for x in num_cols]

cat_idx = [cols.index(x) for x in cat_cols]



# Adicionamos mais duas posições referentes às colunas que serão

# criadas pelas clusterizações

cat_idx += [len(cols), len(cols)+1]
# Construindo Pipeline para os transformadores baseados em clusterização

pipe_kmeans = Pipeline(

    [

        (

            'week_label', KMeansTransformerWeek(5)

        ),

        (

            'dept_label', KMeansTransformerDepts(12)

        )

    ]

)
# Construindo ColumnTransformer que cria dummies para variáveis categóricas

# e padroniza variáveis numéricas, dropando as demais variáveis.

dataprep = ColumnTransformer(

    [

        (

            'ohe',

            OneHotEncoder(handle_unknown='ignore'),

            cat_idx

        ),

        (

            'sc',

            StandardScaler(),

            num_idx

            

        ),

    ],

    remainder='drop'

)
# Agregando todos os passos em um só pipeline

def create_pipe(model, transformers):

    '''

    Cria Pipeline a partir de um determinado modelo e um

    conjuto de transformadores.

    '''

    

    pipe = Pipeline(

    [

        ('KMeansTransform', transformers['clustering']),

        ('dataprep', transformers['dataprep']),

        ('model', model)

    ]

    )

    

    return pipe
def WMAE_Func2(y_true, y_pred, **kwargs):

    '''

    Função de erro customizada, que dá um peso maior para

    datas que são feriados, considerando variável de classificação

    dos feriados.

    '''

    

    df = kwargs['df']

    df = df.loc[

        y_true.index

    ]

    df['Weights'] = 1

    df.loc[

        df['Holiday_Clf'] > 0,

        'Weights'

    ] == 5

    

    weights = df['Weights'].to_numpy()



    wmae = 1/np.sum(weights)*np.sum(weights*np.abs(y_true-y_pred))

    

    return -wmae





# Recriando objeto referente à métrica de avaliação

wmae_scorer = make_scorer(WMAE_Func2, greater_is_better=False, df=X)
# Criando objeto de validação cruzada

tscv = TimeSeriesCV(

    nsplits=6,

    window_size=52,

    test_size=38,

    start_week=X['WeekSeq'].min()

)
# Definindo dicionário com transformers

transformers = {

    'clustering' : pipe_kmeans,

    'dataprep' : dataprep

}



# Definindo modelos a serem testados

models = [

    LinearRegression(),

    DecisionTreeRegressor(),

    RandomForestRegressor(),

    GradientBoostingRegressor(),

    XGBRegressor()

]



# Testando modelos

for model in models:

    

    print('***********************************')

    print('Teste do modelo:', type(model).__name__)

    # Cria pipeline com o modelo em avaliação

    pipe = create_pipe(model, transformers)

    

    # Calcula métrica de avaliação ao longo das folhas de

    # validação cruzada

    scores = cross_val_score(

    pipe,

    X,

    y,

    cv=tscv,

    scoring=wmae_scorer,

    n_jobs=-1

    )

    

    print('Scores:', scores.round(2))

    print('WMAE médio:', scores.mean().round(2))

    print('\n')
# Criando pipeline com modelo RandomForest

pipe = create_pipe(RandomForestRegressor(), transformers)
def objective(trial):

    

    '''

    Função objetivo para otimização, que define o intervalo de busca

    para os hiperparâmetros selecionados.

    '''

    

    # Definindo intervalos de busca

    nclusters_week = trial.suggest_int('KMeansTransform__week_label__num_clusters', 3, 10)

    nclusters_dept = trial.suggest_int('KMeansTransform__dept_label__num_clusters', 3, 15)

    n_estimators = trial.suggest_int('model__n_estimators', 50, 150)

    max_depth = trial.suggest_int('model__max_depth', 10, 100)

    bootstrap = trial.suggest_categorical('model__bootstrap', [False, True])

    max_features = trial.suggest_categorical('model__max_features', ['auto', 'log2'])

    min_samples_leaf = trial.suggest_int('model__min_samples_leaf', 1, 10)

    min_samples_split = trial.suggest_int('model__min_samples_split', 2, 15)

    

    params = {

        'KMeansTransform__week_label__num_clusters': nclusters_week,

        'KMeansTransform__dept_label__num_clusters': nclusters_dept,

        'model__bootstrap': bootstrap,

        'model__max_depth' : max_depth,

        'model__max_features': max_features,

        'model__min_samples_leaf': min_samples_leaf,

        'model__min_samples_split': min_samples_split

    }

    

    # Aplicando tais intervalos ao pipeline

    pipe.set_params(**params)

    

    # Avaliando o modelo com base no WMAE e utilizando a

    # validação cruzada com janela móvel

    scores = cross_val_score(

    pipe,

    X,

    y,

    cv=tscv,

    scoring=wmae_scorer,

    n_jobs=-1

    )

    

    return scores.mean()
# Realizando a otimização

study = optuna.create_study()

study.optimize(objective, n_trials=10)
# Atribuindo os melhores parâmetros encontrados pela otimização ao modelo

pipe.set_params(**study.best_params);
# Definindo conjunto de treino como sendo as 52 semanas anteriores

# à primeira semana do intervalo de predição.



start_week = full_df.loc[full_df['df_label'] == 'test', 'WeekSeq'].min() - 52



X_train = full_df.loc[

    (full_df['df_label'] == 'train')

    & (full_df['WeekSeq'] >= start_week)

]



y_train = full_df.loc[

    (full_df['df_label'] == 'train')

    & (full_df['WeekSeq'] >= start_week),

    'Weekly_Sales'

]
# Ajustando modelo

pipe.fit(X_train, y_train);
# Definindo X de teste

X_test = full_df.loc[

    full_df['df_label'] == 'test'

]
# Fazendo predição

yhat = pipe.predict(X_test)
# Montando arquivo de submissão



sub = full_df.loc[

    full_df['df_label'] == 'test'

]



sub['Id'] = sub['Store'].astype(str).str.cat(

    [

        sub['Dept'].astype(str),

        sub['Date'].astype(str)

    ],

    sep='_'

)



sub['Weekly_Sales'] = yhat



sub = sub[

    [

        'Id',

        'Weekly_Sales'

    ]

]



sub.to_csv('submission.csv', index=False)