!pip install wordninja
!pip install textblob
# Importação dos pacotes para Analise
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report, log_loss, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import dump, load
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import wordninja
import textblob
from nltk.tokenize.treebank import TreebankWordDetokenizer

import warnings
warnings.simplefilter(action = 'ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Importação dos dataset
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head(2)
test.head(2)
sub.head(2)
print('Tamanho do arquivo de Treino', train.shape)
print('Tamanho do arquivo de Teste', test.shape)
df = train.append(test)
df.reset_index(inplace=True, drop=True)
df.shape
# Separando palavras juntas
df['text_split'] = df['text'].apply(wordninja.split)
df['text_new'] = df['text_split'].apply(TreebankWordDetokenizer().detokenize)
df.head(2)
# Verificando o balanceamento dos Tweets
df['target'].value_counts()
df.shape
# Variáveis importantes
pontuacao = ['.',',','-','+',':',';','&','+','/','!','?','#','%','(',')','  ']
# Palavras para retirar da análise
stop_words = stopwords.words('english')
# Tamanho da validação de teste
test_size = 0.2
random_state = 42
# Parametros do vetor CountVectorizer
ngram_range = (1, 2)
strip_accents = 'ascii'
# Parametros do vetor TfidfTransformer
use_idf = True
# Excluindo da descrição texto após os números, informações julgadas irrelevantes para a classificação.
df['text_new'] = df['text_new'].str.replace('[0-9]+', '', regex=True)

# Excluindo da descrição puntuação, informações julgadas irrelevantes para a classificação.
for x in pontuacao:
  df['text_new'] = df['text_new'].str.replace(x, ' ')
  
df.head(2)
# Função Treinamento, Teste, Resultado 
def train(feature, target, new_feature, new_target):
  cvt = CountVectorizer(ngram_range=ngram_range, strip_accents=strip_accents, stop_words=stop_words)
  tfi = TfidfTransformer(use_idf=use_idf)
  clf = RandomForestClassifier(n_estimators=500)
  #clf = LogisticRegression(multi_class='multinomial')
  #clf = MLPClassifier((10,))

  # Criando pipeline
  clf = Pipeline([('cvt', cvt), ('tfi', tfi), ('clf', clf)])
  
  # Dividindo dataset em treino e teste
  x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=test_size, random_state=random_state)

  # Executando pipeline
  clf.fit(x_train, y_train)

  # Avaliando a performance com predição
  predicted = clf.predict(x_test)
  predicted_proba = clf.predict_proba(x_test)
  print('#---------Indicadores Classificação---------#\n')
  print(classification_report(y_test, predicted))
  print('#----Log Loss----#')
  print(log_loss(y_test, predicted_proba))
  print('\n#----F1 Score----#')
  print(f1_score(y_test, predicted))

  # Predição dos novos dados
  predicted_new = clf.predict(new_feature)

  # Probabilidade das predições
  y_proba = clf.predict_proba(new_feature)
  estimadores = clf.classes_

  # Adicionando a coluna para o novo DF
  df_test[new_target] = predicted_new

  return df_test, y_proba, estimadores
# Função Treinamento, Teste, Resultado XGB
def train_xgb(feature, target, new_feature, new_target):
    cvt = CountVectorizer(ngram_range=ngram_range, strip_accents=strip_accents, stop_words=stop_words)
    tfi = TfidfTransformer(use_idf=use_idf)

    # Criando pipeline
    pip = Pipeline([('cvt', cvt), ('tfi', tfi)])
    feature = pip.fit_transform(feature)

    #print("XGBClassifier\n")
    #print("Parameter optimization\n")
    xgb_model = xgb.XGBClassifier()
    clf = GridSearchCV(xgb_model, {'booster': ['gbtree','gblinear','dart'],
                                   'n_estimators': [25, 50, 75]})
    clf.fit(feature, target)
    #print('Melhores parametros\n', clf.best_score_)
    #print('Melhor Score\n', clf.best_params_)
    predicted = clf.predict(feature)
    predicted_proba = clf.predict_proba(feature)
    print('#---------Indicadores Classificação---------#\n')
    print(classification_report(target, predicted))
    print('#----Log Loss----#')
    print(log_loss(target, predicted_proba))
    print('\n#----F1 Score----#')
    print(f1_score(target, predicted))

    new_feature = df_test['text_new']
    new_feature = pip.transform(new_feature)

    # Predição dos novos dados
    predicted_xgb = clf.predict(new_feature)

    # Probabilidade das predições
    y_proba_xgb = clf.predict_proba(new_feature)
    estimadores_xgb = clf.classes_

    # Adicionando a coluna para o novo DF
    df_test[new_target] = predicted_xgb
    
    return df_test, y_proba_xgb, estimadores_xgb
# Separando df treino de test
df.drop(columns=['keyword','location'], inplace=True)
df_train = df.dropna().copy()
df_test = df.loc[(df.target.isnull())].copy()
# Selecionando apenas o item a ser classificado e o target do DF principal
feature = df_train['text_new']
target = df_train.target
# Dados de Teste
new_feature = df_test['text_new']
new_target = 'target'
# Chamando Função
df_test, y_proba, estimadores = train(feature, target, new_feature, new_target)
df_test['target'] = df_test['target'].apply(int)
df_test.head(2)
# Chamando Função XGB
df_test, y_proba_xgb, estimadores_xgb = train_xgb(feature, target, new_feature, new_target)
df_test['target'] = df_test['target'].apply(int)
df_test.head(2)
# Exportando classificação
submission = df_test[['id','target']]
submission.to_csv('submission.csv', index=False)
submission.head(2)
# Exportando classificação XGB
submission_xgb = df_test[['id','target']]
submission_xgb.to_csv('submission_xgb.csv', index=False)
submission_xgb.head(2)


