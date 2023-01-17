import fasttext

import unidecode

import string



import pandas as pd



from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/imdb-ptbr/imdb-reviews-pt-br.csv')

df.head()
# Colocando a coluna id como index do dataset:

df.set_index('id', inplace=True) 





# Transformando o target para utilizar o fasttext:

df['target']=['__label__'+ s for s in df['sentiment']]
# Verificando a distribuição da variável resposta

df['target'].value_counts()
# Colocando todas as palavras em caixa baixa:

df['text_pt'] = df['text_pt'].str.lower()

# Removendo acentuação:

df['text_pt'] = df['text_pt'].apply(lambda text: unidecode.unidecode(text))

# Removendo pontuação:

df['text_pt'] = df['text_pt'].str.replace('[{}]'.format(string.punctuation), '')
df['text_pt'].head()
# Utilizando a proporção de 70/30 para bases de treino e teste, respectivamente:



X = df['text_pt']

Y = df['target']



X_train, X_test, Y_train, Y_test = train_test_split(X,

                                                    Y,

                                                    test_size = 0.3,

                                                    random_state = 42)
# Reunindo novamente as bases de treino e teste:



data_train = {'target': Y_train, 'text': X_train}

df_train = pd.DataFrame.from_dict(data_train)

data_test = {'target': Y_test, 'text': X_test}

df_test = pd.DataFrame.from_dict(data_test)
# Salvando as bases de treino e teste em arquivos separados:



df_train.to_csv('df_train.csv', sep='\t', index = False, header = False)

df_test.to_csv('df_test.csv', sep='\t', index = False, header = False)
model = fasttext.train_supervised(input='df_train.csv')
model.test('df_test.csv')