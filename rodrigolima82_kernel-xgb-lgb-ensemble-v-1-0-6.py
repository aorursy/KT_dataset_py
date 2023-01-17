# Importar os principais pacotes

import numpy as np

import pandas as pd

import itertools

import seaborn as sns

sns.set()



import matplotlib.pyplot as plt

%matplotlib inline



import time

import datetime

import gc



# Evitar que aparece os warnings

import warnings

warnings.filterwarnings("ignore")



# Seta algumas opções no Jupyter para exibição dos datasets

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 200)



# Variavel para controlar o treinamento no Kaggle

TRAIN_OFFLINE = False
# Importa os pacotes de algoritmos

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

import lightgbm as lgb

from lightgbm.sklearn import LGBMClassifier

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier



# Importa pacotes do sklearn

from sklearn import preprocessing

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, log_loss

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from sklearn.decomposition import PCA, FastICA, TruncatedSVD

def read_data():

    

    if TRAIN_OFFLINE:

        print('Carregando arquivo dataset_treino.csv....')

        train = pd.read_csv('../dataset/dataset_treino.csv')

        print('dataset_treino.csv tem {} linhas and {} colunas'.format(train.shape[0], train.shape[1]))

        

        print('Carregando arquivo dataset_teste.csv....')

        test = pd.read_csv('../dataset/dataset_teste.csv')

        print('dataset_teste.csv tem {} linhas and {} colunas'.format(test.shape[0], test.shape[1]))

        

    else:

        print('Carregando arquivo dataset_treino.csv....')

        train = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')

        print('dataset_treino.csv tem {} linhas and {} colunas'.format(train.shape[0], train.shape[1]))

        

        print('Carregando arquivo dataset_treino.csv....')

        test = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')

        print('dataset_teste.csv tem {} linhas and {} colunas'.format(test.shape[0], test.shape[1]))

    

    return train, test
# Leitura dos dados

train, test = read_data()
# Juntando os datasets de treino e teste para realizar todos os tratamentos nos dados

df = train.append(test)
# Transformando as features categorias com LabelEncoder

le = LabelEncoder()



for i, col in enumerate(df):

    if df[col].dtype == 'object':

        df[col] = le.fit_transform(np.array(df[col].astype(str)).reshape((-1,)))

        

        

# Realizando tratamento de missing value

for c in df.columns:

    if c != 'ID' and c != 'target':

        df[c].fillna(df[c].mean(),inplace=True)

# Criando novas features atraces do PCA / ICA / GRP e SRP



n_comp = 4



# tSVD

tsvd = TruncatedSVD(n_components=n_comp, random_state=42)

tsvd_results_df = tsvd.fit_transform(df.drop(columns = ['ID','target'], axis = 1))



# PCA

pca = PCA(n_components=n_comp, random_state=42)

pca2_results_df = pca.fit_transform(df.drop(columns = ['ID','target'], axis = 1))



# ICA

ica = FastICA(n_components=n_comp, random_state=42)

ica2_results_df = ica.fit_transform(df.drop(columns = ['ID','target'], axis = 1))



# GRP

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=42)

grp_results_df = grp.fit_transform(df.drop(columns = ['ID','target'], axis = 1))



# SRP

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=42)

srp_results_df = srp.fit_transform(df.drop(columns = ['ID','target'], axis = 1))





# Append dos componentes com o dataset

for i in range(1, n_comp+1):

    df['pca_' + str(i)]  = pca2_results_df[:,i-1]

    df['ica_' + str(i)]  = ica2_results_df[:,i-1]

    df['tsvd_' + str(i)] = tsvd_results_df[:,i-1]

    df['grp_' + str(i)]  = grp_results_df[:,i-1]

    df['srp_' + str(i)]  = srp_results_df[:,i-1] 
# Patronizacao dos dados

scaler = StandardScaler()

for c in df.columns:

    if c != 'ID' and c != 'target':

        df[c] = scaler.fit_transform(df[c].values.reshape(-1, 1))
df.head()
# Separa dataset de treino e teste depois de aplicar Feature Engineering

treino = df[df['target'].notnull()]

teste = df[df['target'].isnull()]



# Separando features preditoras e target

train_x = treino.drop(['ID','target'], axis=1)

train_y = treino['target']



# Removendo ID dataset de teste

test_x = teste.drop(['ID','target'], axis=1)
# Criando uma funcao para criação, execução e validação do modelo

def run_model_xgb(X_tr, y_tr, useTrainCV=True, cv_folds=5, early_stopping_rounds=10):

    

    # Criando o modelo XGB com todas as otimizações

    modelo = XGBClassifier(learning_rate = 0.1, 

                              n_estimators = 1000, 

                              max_depth = 4,

                              min_child_weight = 1, 

                              gamma = 0, 

                              subsample = 0.7, 

                              colsample_bytree = 0.6,

                              reg_alpha = 0.005,

                              objective = 'binary:logistic', 

                              n_jobs = -1,

                              scale_pos_weight = 1, 

                              seed = 42)



    # Utilização do Cross-Validation

    if useTrainCV:

        xgb_param = modelo.get_xgb_params()

        xgtrain = xgb.DMatrix(X_tr, label=y_tr)

        

        print ('Start cross validation')

        cvresult = xgb.cv(xgb_param, 

                          xgtrain, 

                          num_boost_round=modelo.get_params()['n_estimators'], 

                          nfold=cv_folds,

                          metrics=['logloss'],

                          stratified=True,

                          seed=42,

                          verbose_eval=True,

                          early_stopping_rounds=early_stopping_rounds)



        modelo.set_params(n_estimators=cvresult.shape[0])

        best_tree = cvresult.shape[0]

        print('Best number of trees = {}'.format(best_tree))

    

    # Fit do modelo

    modelo.fit(X_tr, y_tr, eval_metric='logloss')

        

    # Predição no dataset de treino

    train_pred = modelo.predict(X_tr)

    train_pred_prob = modelo.predict_proba(X_tr)[:,1]

    

    # Exibir o relatorio do modelo

    print("Log Loss (Treino): %f" % log_loss(y_tr, train_pred_prob))

    print("Log Loss (Test): %f" % cvresult['test-logloss-mean'][best_tree-1])

    

    return modelo
%%time



modeloXGB = run_model_xgb(train_x, train_y)
# Configurações Gerais

STRATIFIED_KFOLD = False

RANDOM_SEED = 737851

NUM_THREADS = 4

NUM_FOLDS = 10

EARLY_STOPPING = 1000



LIGHTGBM_PARAMS = {

    'boosting_type': 'goss',#'gbdt',

    'n_estimators': 10000,

    'learning_rate': 0.005134,

    'num_leaves': 54,

    'max_depth': 10,

    'subsample_for_bin': 240000,

    'reg_alpha': 0.436193,

    'reg_lambda': 0.479169,

    'colsample_bytree': 0.508716,

    'min_split_gain': 0.024766,

    'subsample': 1,

    'is_unbalance': False,

    'silent':-1,

    'verbose':-1

}
def run_model_lgb(data):

    df = data[data['target'].notnull()]

    test = data[data['target'].isnull()]

    del_features = ['ID','target']

    predictors = list(filter(lambda v: v not in del_features, df.columns))

    

    print("Train/valid shape: {}, test shape: {}".format(df.shape, test.shape))



    if not STRATIFIED_KFOLD:

        folds = KFold(n_splits= NUM_FOLDS, shuffle=True, random_state= RANDOM_SEED)

    else:

        folds = StratifiedKFold(n_splits= NUM_FOLDS, shuffle=True, random_state= RANDOM_SEED)



    oof_preds = np.zeros(df.shape[0])

    sub_preds = np.zeros(test.shape[0])

    importance_df = pd.DataFrame()

    eval_results = dict()

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[predictors], df['target'])):

        train_x, train_y = df[predictors].iloc[train_idx], df['target'].iloc[train_idx]

        valid_x, valid_y = df[predictors].iloc[valid_idx], df['target'].iloc[valid_idx]



        params = {'random_state': RANDOM_SEED, 'nthread': NUM_THREADS}

        

        clf = LGBMClassifier(**{**params, **LIGHTGBM_PARAMS})

        

        clf.fit(train_x, train_y, 

                eval_set=[(train_x, train_y), (valid_x, valid_y)],

                eval_metric='logloss', 

                verbose=400, 

                early_stopping_rounds= EARLY_STOPPING)



        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        sub_preds += clf.predict_proba(test[predictors], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits



        print('Fold %2d Log Loss : %.6f' % (n_fold + 1, log_loss(valid_y, oof_preds[valid_idx])))

        del train_x, train_y, valid_x, valid_y

        gc.collect()



    print('Full Log Loss score %.6f' % log_loss(df['target'], oof_preds))

        

    return clf
%%time



modeloLGB = run_model_lgb(df)
# Submission XGB

submissionXGB = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/sample_submission.csv')

submissionXGB['PredictedProb'] = modeloXGB.predict_proba(test_x)[:,1]

plt.hist(submissionXGB.PredictedProb)

plt.show()
# Submission LGB

submissionLGB = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/sample_submission.csv')

submissionLGB['PredictedProb'] = modeloLGB.predict_proba(test_x)[:,1]

plt.hist(submissionLGB.PredictedProb)

plt.show()
submissionEnsemble = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/sample_submission.csv')

submissionEnsemble['PredictedProb'] = submissionXGB['PredictedProb'] * 0.7 + submissionLGB['PredictedProb'] * 0.3

submissionEnsemble.to_csv('submission_ensemble_v1.0.6.csv', index=False)

plt.hist(submissionEnsemble.PredictedProb)

plt.show()