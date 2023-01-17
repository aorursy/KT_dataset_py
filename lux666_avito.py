#importações
import math
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
#Lendo os dados de treinamento e teste e criando os dataframes (estrutura de dados) train e test
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#Preenchendo os valores vazios com 0 para evitar nan
train = train.fillna(0)
test = test.fillna(0)

'''
#Criando as colunas de ano, mês e dia da feature activation date para obtermos colunas numéricas

#Criando as colunas de ano, mês e dia para o dataset de treinamento
years = []
months = []
days = []
for date in train['activation_date']:
    years.append(date[0:4])
    months.append(date[5:7])
    days.append(date[8:10])

#Criando as novas colunas no dataframe
train['activation_year'] = years
train['activation_month'] = months
train['activation_day'] = days

#Criando as colunas de ano, mês e dia para o dataset de teste
years = []
months = []
days = []
for date in test['activation_date']:
    years.append(date[0:4])
    months.append(date[5:7])
    days.append(date[8:10])
#Criando as novas colunas no dataframe
test['activation_year'] = years
test['activation_month'] = months
test['activation_day'] = days'''

#Convertendo as colunas de classes discretas para valores inteiros, para obtermos colunas numéricas
label_encoders = { } #cria lista dict de chave/valor para 
features_to_encode = ['category_name', 'parent_category_name', 'city', 'region', 'user_type']
for feature in features_to_encode:
    label_encoders[feature] = preprocessing.LabelEncoder() #instanciando o objeto label_encoders da classe LabelEnconder
    label_encoders[feature].fit(pd.concat([train[feature], test[feature]])) #gera tabela de correspondência entre dados de uma feature e um número
    train[feature] = label_encoders[feature].transform(train[feature])
    test[feature] = label_encoders[feature].transform(test[feature])

#Transformando a coluna image do dataset. Como funciona: se a coluna está preenchida com valor significa que o anúncio
#possui um código para a imagem, e então o valor do seu atributo image é setado para 1. Caso contrário, seu valor é setado para 0.
#Primeiro trato as images do dataset de treinamento
images = []
for image in train['image']:
    if len(str(image)) <= 1:
        images.append(0)
    else:
        images.append(1)
train['image'] = images
 
#Depois trato as images do dataset de teste
images = []
for image in test['image']:
    if len(str(image)) <= 1:
        images.append(0)
    else:
        images.append(1)
test['image'] = images

#Como ainda foi possível utilizar nenhum modelo baseado em linguagem natural, analisou-se o comprimento
#e o número de palavras contidas no título e na descrição do anúncio, partindo da hipótese de que um anúncio mais bem detalhado
#pode gerar maior interesse de compra. Isso melhorou um pouco a perfórmance do modelo.

#computando o comprimento e o número de palavras do título e da descrição do conjunto de treinamento
length_title = [] #número de caracteres da string que corresponde ao título do anúncio
length_description = [] #número de caracteres da string que corresponde ao comprimento da descrição do anúncio
words_title = [] #quantidade de palavras no título
words_description = [] #quantidade de palavras na descrição
percent_capital_title = [] #porcentagem de capital letters no título
percent_capital_description = [] #porcentagem de capital letters na descrição

#começando a computar o comprimento, o número de palavras e a porcentagem de capital letters no título p/ a base de treinamento
for title in train['title']: 
    length_title.append(len(str(title)))
    words_title.append(len(str(title).split(' ')))
    count_capital = sum(1 for letter in str(title) if letter.isupper())
    if count_capital > 0:
        percent_capital_title.append(int(len(str(title)) / count_capital)) #adiciona no percent_capital_title a (qtdade de caracteres título / qtdade de letras maiúsculas)
    else:
        percent_capital_title.append(0)

#começando a computar o comprimento,o número de palavras e a porcentagem de capital letters na descrição p/ a base de treinamento
for description in train['description']:
    length_description.append(len(str(description)))
    words_description.append(len(str(description).split(' ')))
    count_capital = sum(1 for letter in str(description) if letter.isupper())
    if count_capital > 0:
        percent_capital_description.append(int(len(str(description)) / count_capital))
    else:
        percent_capital_description.append(0)

#criando as novas colunas de comprimento (título e descrição), número de palavras (título e descrição) e porcentagem de capital letters (título e descrição) para o conjunto de treino
train['length_title'] = length_title
train['length_description'] = length_description
train['words_title'] = words_title
train['words_description'] = words_description
train['percent_capital_title'] = percent_capital_title
train['percent_capital_description'] = percent_capital_description

#computando o comprimento e o número de palavras do título e da descrição do conjunto de teste
length_title = []
length_description = []
words_title = []
words_description = []
percent_capital_title = []
percent_capital_description = []

#começando a computar o comprimento,o número de palavras e a porcentagem de capital letters no título p/ a base de teste
for title in test['title']:
    length_title.append(len(str(title)))
    words_title.append(len(str(title).split(' ')))
    count_capital = sum(1 for letter in str(title) if letter.isupper())
    if count_capital > 0:
        percent_capital_title.append(int(len(str(title)) / count_capital))
    else:
        percent_capital_title.append(0)

#começando a computar o comprimento,o número de palavras e a porcentagem de capital letters na descrição p/ a base de teste
for description in test['description']:
    length_description.append(len(str(description)))
    words_description.append(len(str(description).split(' ')))
    count_capital = sum(1 for letter in str(description) if letter.isupper())
    if count_capital > 0:
        percent_capital_description.append(int(len(str(description)) / count_capital))
    else:
        percent_capital_description.append(0)

#criando as novas colunas de comprimento (título e descrição), número de palavras (título e descrição) e porcentagem de capital letters (título e descrição) para o conjunto de teste
test['length_title'] = length_title
test['length_description'] = length_description
test['words_title'] = words_title
test['words_description'] = words_description
test['percent_capital_title'] = percent_capital_title
test['percent_capital_description'] = percent_capital_description

#Escolhendo quais features serão usadas para treinar o modelo (aqui somente as features numéricas podem ser utilizadas,
#porque se trata de uma regressão linear).
#Features possíveis: price, item_seq_number, image_top_1, category_name, parent_category_name, city, region, user_type,
#length_title, length_description, words_title, words_description, percent_capital_title, percent_capital_description.
#O atributo user_type parece prejudicar o algoritmo, por isso não foi utilizado.
#Atributos que parecem não fazer diferença no resultado e por isso foram removidos (bloco supracomentado) do modelo: activation_year, month e day.

features = ['price', 'item_seq_number', 'image_top_1', 'category_name', 'parent_category_name', 'city', 'region', 'image',
            'length_description', 'length_title', 'words_title', 'words_description', 'percent_capital_title', 'percent_capital_description']

x_train = train[features]
x_test = test[features]

#Guardando os targets para treinar o modelo 
y_train = train.deal_probability

#Regra:
#x_train ->>> y_train (deal_probability do conjunto de treinamento. É um dado do problema)
#x_test  ->>> y_test = pred = predições = resposta do desafio = deal_probability

#Instanciando nosso modelo de regressão linear
linear_regression = linear_model.LinearRegression()
#Treinando o modelo
linear_regression.fit(x_train, y_train)
#Fazendo as predições
pred = linear_regression.predict(x_test)

#transforma para zero o pequeno número de casos (exceção) em que a predição tem valor negativo
i = 0
for x in pred:
    if x < 0:
        pred[i] = 0
    i += 1

'''Calculando a acurácia através do mean squared error, que é como o kaggle faz o rankeamento das submissões
#meanSquaredError = np.sum(np.square(y_test - pred))/pred.size
#rootMeanSquaredError = math.sqrt(meanSquaredError)
#print('Full dataset root mean squared error: ', rootMeanSquaredError)'''
#Criando um dataframe do pandas com as colunas pedidas pelo kaggle: item_id e deal_probability
my_submission = pd.DataFrame({'item_id': test['item_id'], 'deal_probability': pred})
#Salvando o dataframe do resultado em CSV para que possa ser submetido ao desafio
my_submission.to_csv('submission.csv', index=False)