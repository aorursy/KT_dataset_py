import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input/fasam-nlp-competition-turma-4"))
import numpy as np

import pandas as pd

import seaborn as sns

import warnings

import matplotlib

import matplotlib.pyplot as plt



%matplotlib inline



sns.set(style="ticks")

warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
# Leitura do Dataset

df = pd.read_csv('../input/fasam-nlp-competition-turma-4/train.csv')

print(df.shape)

df.head()
# Definição de alguns parâmetros dos modelos e tokenização



# Máximo de tokens 

min_word_frequency = 10



# Tamanho do embedding

n_gram_range = (1, 1)
df['full_text'] = df['title'] + "\n" + df['text']

text         = df['full_text'].values

vectorizer   = TfidfVectorizer(min_df=min_word_frequency, ngram_range=n_gram_range)



# Transforma o texto em números

vectorizer.fit(text)

X = vectorizer.transform(text)  
# Categoriza o target "category" -> [0,..., 1] (output: y)

Y_classes = pd.get_dummies(df['category']).columns

Y         = pd.get_dummies(df['category']).values
Y_classes
Y
(X.shape, Y.shape)
# Separa o dataset em dados de treinamento/validação

X_train, X_val, Y_train, Y_val = train_test_split(X,Y, 

                                                      test_size = 0.20, 

                                                      random_state = 42,

                                                      stratify=Y)
model = OneVsRestClassifier(LogisticRegression())

model.fit(X_train, Y_train)
model.score(X_train, Y_train)
# Avaliação do modelo para o dataset de validação

model.score(X_val, Y_val)
# Leitura do Dataset de validação dos resultados

test_df = pd.read_csv('../input/fasam-nlp-competition-turma-4/test.csv')

print(test_df.shape)

test_df.head()
test_df['full_text'] = test_df['title'] + "\n" + test_df['text']
def predict(text):

    '''

    Utiliza o modelo treinado para realizar a predição

    '''

    new_text = vectorizer.transform(text)

    pred     = model.predict(new_text)

    return pred
pred = predict(test_df.full_text)

pred_classes = [Y_classes[np.argmax(c)] for c in pred]

pred_classes[:5]
# Atualizando a categoria dos artigos no dataset de validação

test_df['category'] = pred_classes

test_df.head()
# Criando o arquivo submission.csv contendo os dados para cálculo do ranking no kaggle

# Esse arquivo deve ser enviado para o kaggle

test_df[["article_id", "category"]].to_csv("submission.csv", index=False)