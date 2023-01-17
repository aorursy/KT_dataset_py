!pip install emot
!python -m spacy download pt
!pip install scikit-plot
!pip install imblearn
# para importar e manipular dados
import numpy  as np
import pandas as pd

# para expressão regular
import re

# para remover pontuações
import string

# para nlp
import spacy

# para manipular emoji/emoticon
import emot
from emot.emo_unicode import UNICODE_EMO

# para vetorizar
from sklearn.feature_extraction.text import TfidfVectorizer

# para gerar dados sintéticos e balancear a variável a ser predita
from imblearn.over_sampling import SMOTE

# para dispor os dados na mesma escala
from sklearn.preprocessing import MinMaxScaler

# modelo de machine learning
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble        import RandomForestClassifier
from sklearn.naive_bayes     import MultinomialNB
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression

# para avaliar o modelo
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import scikitplot as skplt

# cria o objeto de pré processamento spacy
pln = spacy.load('pt')

# exibir todas as colunas do dataframe
pd.set_option('display.max_columns', None)
# importação da primeira planilha - para treinamento
resenhas_treino = pd.read_excel('../input/satisfacao-apps-bancos/Satisfao com App.xlsx')
resenhas_treino.info()
# importação da segunda planilha - para validação
resenhas_validacao = pd.read_excel('../input/satisfacao-apps-bancos/Satisfao com App.xlsx', sheet_name=1)
resenhas_validacao.info()
# alterar os nomes das variáveis eliminando os espaços vazios
col_names = [col.replace(' ', '_') for col in resenhas_treino.columns]
resenhas_treino.columns        = col_names
resenhas_validacao.columns = col_names
# excluir os regisros com valores ausentes
resenhas_treino.dropna(inplace=True)
resenhas_validacao.dropna(inplace=True)
# alterar as variáveis relativas aos sentimentos atribuídos às resenhas dos usuários em maiúsculas
ls_cols_sentimentos = [col for col in resenhas_treino.columns if 'Elogio' in col or 'Reclamação' in col or 'Classificável' in col]
resenhas_treino[ls_cols_sentimentos]    = resenhas_treino[ls_cols_sentimentos].apply(lambda x: x.str.upper())
resenhas_validacao[ls_cols_sentimentos] = resenhas_validacao[ls_cols_sentimentos].apply(lambda x: x.str.upper())
# unificar a ausência do til
for var in ls_cols_sentimentos:
    resenhas_treino[var]    = resenhas_treino[var].map({'NÃO': 'NAO', 'SIM': 'SIM'})
    resenhas_validacao[var] = resenhas_validacao[var].map({'NÃO': 'NAO', 'SIM': 'SIM'})
# criar variável "comentario_negativo" que é binária com a seguinte regra:
# quando "Reclamação quanto ao app" ou "Reclamação a Instituição" for "SIM" retornar 1
# caso contrário retornar 0

def localizar_txt_negativo(df):
    if df['Reclamação_quanto_ao_app'] == 'SIM' or df['Reclamação_a_Instituição'] == 'SIM':
        return 1
    else:
        return 0

resenhas_treino['comentario_negativo']    = resenhas_treino.apply(localizar_txt_negativo, axis=1)
resenhas_validacao['comentario_negativo'] = resenhas_validacao.apply(localizar_txt_negativo, axis=1)
# checar as proporções (%) da variável comentario_negativo
resenhas_treino['comentario_negativo'].value_counts() / len(resenhas_treino) * 100
# amostra dos dados:
resenhas_treino.tail()
# amostra dos dados:
resenhas_validacao.tail()
# criar lista de comentários
lista_comentarios = resenhas_treino['Comentario'].tolist()
# criar função para armazenar lista de emojis e emoticons
# a depender do texto, pode ser retornada uma lista, o que é tratado a seguir
emojis = []
emotis = []

def localiza_emoji_emoti(texto):  
    emoji = emot.emoji(texto)
    emoti = emot.emoticons(texto)
    
    if emoji['flag'] == True:
        emojis.append(emoji['value'])
        
    try:
        if emoti['flag'] == True:
            emotis.append(emoti['value'])
    except:
        emotis.append('nada')
# aplicando a função na lista de comentários
for txt in lista_comentarios:
    localiza_emoji_emoti(txt)
# removendo duplicidade dos emojis e emoticos
lista_emojis = []
lista_emotis = []

for linha in emojis:
    for emoji in linha:
        if emoji not in lista_emojis:
            lista_emojis.append(emoji)
            
for linha in emotis:
    for emoti in linha:
        if emoti != 'nada' and emoti not in lista_emotis:
            lista_emotis.append(emoti)
            
len(lista_emojis), len(lista_emotis)
# visualizar os primeiros emojis:
lista_emojis[0:6]
# visualizar os primeiros emoticons:
lista_emotis[0:6]
# na lista de emoticons foi capturada parte da string "nada", estes valores podem ser removidos
lista_emotis.remove('n')
lista_emotis.remove('d')
lista_emotis.remove('a')
# criar dois dicionários com as interpretações, um para emojis outro para emoticons
dict_emojis = {
    'exclamation_question_mark': 'ruim',
    'person_pouting': 'ruim',
    'kiss_mark': 'ótimo',
    'upside-down_face': 'ótimo',
    'smiling_face_with_open_mouth_&_smiling_eyes': 'ótimo',
    'love_letter': 'ótimo',
    'rose': 'ótimo',
    'angry_face_with_horns': 'ruim',
    'yellow_heart': 'ótimo',
    'blue_heart': 'ótimo',
    'green_heart': 'ótimo',
    'relieved_face': 'ótimo',
    'trophy': 'ótimo',
    'expressionless_face': 'ruim',
    'slightly_smiling_face': 'ótimo',
    'nauseated_face': 'ruim',
    'face_with_stuck-out_tongue_&_winking_eye': 'ótimo',
    'OK_hand': 'ótimo',
    'neutral_face': 'ruim',
    'person_shrugging': 'ruim',
    'weary_face': 'ruim',
    'heart_with_arrow': 'ótimo',
    'grimacing_face': 'ruim',
    'sleepy_face': 'ruim',
    'pig_face': 'ruim',
    'thinking_face': 'ruim',
    'loudly_crying_face': 'ruim',
    'blossom': 'ótimo',
    'face_with_cold_sweat': 'ruim',
    'crying_cat_face': 'ruim',
    'unamused_face': 'ruim',
    'disappointed_but_relieved_face': 'ruim',
    'smiling_face': 'ótimo',
    'face_screaming_in_fear': 'ruim',
    'face_with_steam_from_nose': 'ruim',
    'broken_heart': 'ruim',
    'see-no-evil_monkey': 'ruim',
    'two_hearts': 'ótimo',
    'growing_heart': 'ótimo',
    'slightly_frowning_face': 'ruim',
    'crying_face': 'ruim',
    'dizzy': 'ruim',
    'smiling_face_with_open_mouth_&_closed_eyes': 'ótimo',
    'victory_hand': 'ótimo',
    'face_with_rolling_eyes': 'ruim',
    'revolving_hearts': 'ótimo',
    'smiling_face_with_open_mouth': 'ótimo',
    'rolling_on_the_floor_laughing': 'ótimo',
    'pensive_face': 'ruim',
    'dizzy_face': 'ruim',
    'angry_face': 'ruim',
    'confused_face': 'ruim',
    'smiling_face_with_open_mouth_&_cold_sweat': 'ótimo',
    'smirking_face': 'ótimo',
    'smiling_face_with_sunglasses': 'ótimo',
    'face_with_tears_of_joy': 'ótimo',
    'white_medium_star': 'ótimo',
    'thumbs_down': 'ruim',
    'red_heart': 'ótimo',
    'clapping_hands': 'ótimo',
    'smiling_face_with_halo': 'ótimo',
    'purple_heart': 'ótimo',
    'smiling_face_with_heart-eyes': 'ótimo',
    'heart_suit': 'ótimo',
    'hugging_face': 'ótimo',
    'glowing_star': 'ótimo',
    'smiling_face_with_smiling_eyes': 'ótimo',
    'grinning_face_with_smiling_eyes': 'ótimo',
    'thumbs_up': 'ótimo',
    'face_blowing_a_kiss': 'ótimo',
    'winking_face': 'ótimo'
}

dict_emotis = {
    'Wink or smirk': 'ótimo',
    'Happy face or smiley': 'ótimo',
    'Tongue sticking out, cheeky, playful or blowing a raspberry': 'ótimo',
    'Frown, sad, andry or pouting': 'ruim',
    'Skeptical, annoyed, undecided, uneasy or hesitant': 'ruim'
}

len(dict_emojis), len(dict_emotis)
# dos 105 emojis, 71 foram traduzidos. Os emoticons foram todos traduzidos
# transformar emoji/emoticon para seu significado literal
# fonte: https://towardsdatascience.com/text-preprocessing-for-data-scientist-3d2419c8199d#:~:text=Text%20preprocessing%20is%20an%20important,learning%20algorithms%20can%20perform%20better.

def traduzir_emoti_emoji(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, UNICODE_EMO[emot])
        text = text.replace(':', ' ')

    lista = text.split(' ')
    for x in range(len(lista)):
        chave = lista[x]
        if chave in dict_emojis:
            lista[x] = dict_emojis[chave]
        if chave in dict_emotis:
            lista[x] = dict_emotis[chave]
            
    texto = ' '
    texto = (texto.join(lista)) 
    texto = texto.strip()
            
    return texto
ls_resenha_uma_palavra = list(set(resenhas_treino[resenhas_treino['Comentario'].fillna(' ').str.split().str.len() == 1]['Comentario'].str.lower().values))
len(ls_resenha_uma_palavra)
# importar stop words em português do Spacy
from spacy.lang.pt.stop_words import STOP_WORDS
stop_words = STOP_WORDS
len(stop_words)
# lista final de stop words
ls_stop_words = [word for word in stop_words if word not in ls_resenha_uma_palavra]
len(ls_stop_words)
pontuacoes = string.punctuation
pontuacoes
# função final de préprocessamento

def preprocessamento(texto):
    # Letras minúsculas
    texto = str(texto).lower()
    
    # converter emoji/emoticon para texto
    texto = traduzir_emoti_emoji(texto)

    # Espaços em branco
    texto = re.sub(r" +", ' ', texto)

    # Lematização
    documento = pln(texto)

    lista = []
    for token in documento:
        lista.append(token.lemma_)
  
    # Stop words, pontuações e espaços em excesso (função strip)
    lista = [palavra for palavra in lista if palavra not in ls_stop_words and palavra not in pontuacoes]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()]).strip()

    return lista
# aplicando pré processamento e armazenando em nova coluna
resenhas_treino['coment_processado']    = resenhas_treino['Comentario'].apply(preprocessamento)
resenhas_validacao['coment_processado'] = resenhas_validacao['Comentario'].apply(preprocessamento)
resenhas_treino.head()
resenhas_treino.tail()
resenhas_validacao.head()
resenhas_validacao.tail()
# transaformar a lista em objeto tipo string
ls_resenhas = resenhas_treino['coment_processado'].tolist()
texto = ' '.join(ls_resenhas)
type(texto)
# carregar o modelo Spacy em português
nlp = spacy.load('pt')
# ... e processar o texto
doc = nlp(texto)
def qtde_palavras(texto):
    texto = str(texto)
    palavras = texto.split()
    return len(palavras)

def qtde_maiusculas(texto):
    texto = str(texto)
    quantidade = 0
    for c in texto:
        if c.isupper():
            quantidade += 1
    return quantidade

def comprimento(texto):
    texto = str(texto)
    return len(texto)

def qtde_exclamacoes(texto):
    texto = str(texto)
    quantidade = 0
    for x in range(len(texto)):
        if '!' in texto[x]:
            quantidade += 1
    return quantidade

def qtde_interrogacoes(texto):
    texto = str(texto)
    quantidade = 0
    for x in range(len(texto)):
        if '?' in texto[x]:
            quantidade += 1
    return quantidade

def qtde_pontuacoes(texto):
    texto = str(texto)
    quantidade = 0
    for x in range(len(texto)):
        if '.' in texto[x] or ',' in texto[x] or ';' in texto[x] or ':' in texto[x]:
            quantidade += 1
    return quantidade

def qtde_simbolos(texto):
    texto = str(texto)
    quantidade = 0
    for x in range(len(texto)):
        if '*' in texto[x] or '&' in texto[x] or '%' in texto[x] or '$' in texto[x]:
            quantidade += 1
    return quantidade

def qtde_palavras_unicas(texto):
    texto = str(texto)
    palavras = texto.split()
    palavras_unicas = set(palavras)
    return len(palavras_unicas)

def qtde_tag_part_of_speech(texto):
    texto = str(texto)
    doc = nlp(texto)
    pos_list = [token.pos_ for token in doc]
    qtde_substantivos = len([w for w in pos_list if w == 'NOUN'])
    qtde_adjetivos    = len([w for w in pos_list if w == 'ADJ'])
    qtde_verbos       = len([w for w in pos_list if w == 'VERB'])
    qtde_adverbios    = len([w for w in pos_list if w == 'ADV'])
    qtde_interjeicoes = len([w for w in pos_list if w == 'INTJ'])
    return[qtde_substantivos, qtde_adjetivos, qtde_verbos, qtde_adverbios, qtde_interjeicoes]
%%time

# criar novas variáeis aplicando as funções acima
resenhas_treino['qtde_palavras']    = resenhas_treino['Comentario'].apply(qtde_palavras)
resenhas_treino['qtde_maiusculas']  = resenhas_treino['Comentario'].apply(qtde_maiusculas)
resenhas_treino['comprimento']      = resenhas_treino['Comentario'].apply(comprimento)
resenhas_treino['maiusc_x_compri']  = resenhas_treino['qtde_maiusculas'] / resenhas_treino['comprimento']
resenhas_treino['qtde_exclamacoes'] = resenhas_treino['Comentario'].apply(qtde_exclamacoes)
resenhas_treino['qtde_interrogacoes'] = resenhas_treino['Comentario'].apply(qtde_interrogacoes)
resenhas_treino['qtde_pontuacoes']    = resenhas_treino['Comentario'].apply(qtde_pontuacoes)
resenhas_treino['qtde_simbolos']      = resenhas_treino['Comentario'].apply(qtde_simbolos)
resenhas_treino['qtde_palavras_unicas'] = resenhas_treino['Comentario'].apply(qtde_palavras_unicas)
resenhas_treino['unicas_x_comprimento'] = resenhas_treino['qtde_palavras_unicas'] / resenhas_treino['comprimento']
resenhas_treino['qtde_substantivos'], resenhas_treino['qtde_adjetivos'], resenhas_treino['qtde_verbos'], resenhas_treino['qtde_adverbios'], resenhas_treino['qtde_interjeicoes'] = zip(*resenhas_treino['Comentario'].apply(lambda comment: qtde_tag_part_of_speech(comment)))

resenhas_treino['substantivos_vs_comprimento']  = resenhas_treino['qtde_substantivos'] / resenhas_treino['comprimento']
resenhas_treino['adjectivos_x_comprimento']     = resenhas_treino['qtde_adjetivos'] / resenhas_treino['comprimento']
resenhas_treino['verbos_x_comprimento']         = resenhas_treino['qtde_verbos'] /resenhas_treino['comprimento']
resenhas_treino['adverbios_x_comprimento']      = resenhas_treino['qtde_adverbios'] /resenhas_treino['comprimento']
resenhas_treino['interjeicoes_x_comprimento']   = resenhas_treino['qtde_interjeicoes'] /resenhas_treino['comprimento']
resenhas_treino['substantivos_x_qtde_palavras'] = resenhas_treino['qtde_substantivos'] / resenhas_treino['qtde_palavras']
resenhas_treino['adjectivos_x_qtde_palavras']   = resenhas_treino['qtde_adjetivos'] / resenhas_treino['qtde_palavras']
resenhas_treino['verbos_x_qtde_palavras']       = resenhas_treino['qtde_verbos'] / resenhas_treino['qtde_palavras']
resenhas_treino['adverbios_x_qtde_palavras']    = resenhas_treino['qtde_adverbios'] / resenhas_treino['qtde_palavras']
resenhas_treino['interjeicoes_x_qtde_palavras'] = resenhas_treino['qtde_interjeicoes'] / resenhas_treino['qtde_palavras']
# vetorizador
vect = TfidfVectorizer(stop_words=ls_stop_words, # define as stop-words
                       ngram_range=(1, 2),       # gerar unigramas e bigramas
                       max_features=300          # selecionar as 300 features (variáveis) mais importantes
                      ).fit(resenhas_treino.coment_processado)
# Criar matriz esparsa a partir do vetorizador
X = vect.transform(resenhas_treino.coment_processado)
# Gerar Pandas DataFrame
resenhas_tfidf = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())

# relatório de dtypes
resenhas_tfidf.info()
resenhas_tfidf['qtde_palavras']                = resenhas_treino['qtde_palavras']                 
resenhas_tfidf['qtde_maiusculas']              = resenhas_treino['qtde_maiusculas']               
resenhas_tfidf['comprimento']                  = resenhas_treino['comprimento']                   
resenhas_tfidf['maiusc_x_compri']              = resenhas_treino['maiusc_x_compri']               
resenhas_tfidf['qtde_exclamacoes']             = resenhas_treino['qtde_exclamacoes']              
resenhas_tfidf['qtde_interrogacoes']           = resenhas_treino['qtde_interrogacoes']            
resenhas_tfidf['qtde_pontuacoes']              = resenhas_treino['qtde_pontuacoes']               
resenhas_tfidf['qtde_simbolos']                = resenhas_treino['qtde_simbolos']                 
resenhas_tfidf['qtde_palavras_unicas']         = resenhas_treino['qtde_palavras_unicas']          
resenhas_tfidf['unicas_x_comprimento']         = resenhas_treino['unicas_x_comprimento']          
resenhas_tfidf['qtde_substantivos']            = resenhas_treino['qtde_substantivos']                         
resenhas_tfidf['qtde_adjetivos']               = resenhas_treino['qtde_adjetivos']                    
resenhas_tfidf['qtde_verbos']                  = resenhas_treino['qtde_verbos']                         
resenhas_tfidf['qtde_adverbios']               = resenhas_treino['qtde_adverbios']              
resenhas_tfidf['qtde_interjeicoes']            = resenhas_treino['qtde_interjeicoes']              
resenhas_tfidf['substantivos_vs_comprimento']  = resenhas_treino['substantivos_vs_comprimento']   
resenhas_tfidf['adjectivos_x_comprimento']     = resenhas_treino['adjectivos_x_comprimento']      
resenhas_tfidf['verbos_x_comprimento']         = resenhas_treino['verbos_x_comprimento']          
resenhas_tfidf['interjeicoes_x_comprimento']   = resenhas_treino['adverbios_x_comprimento']
resenhas_tfidf['interjeicoes_x_comprimento']   = resenhas_treino['interjeicoes_x_comprimento']
resenhas_tfidf['substantivos_x_qtde_palavras'] = resenhas_treino['substantivos_x_qtde_palavras']  
resenhas_tfidf['adjectivos_x_qtde_palavras']   = resenhas_treino['adjectivos_x_qtde_palavras']    
resenhas_tfidf['verbos_x_qtde_palavras']       = resenhas_treino['verbos_x_qtde_palavras']       
resenhas_tfidf['adverbios_x_qtde_palavras']    = resenhas_treino['adverbios_x_qtde_palavras']
resenhas_tfidf['interjeicoes_x_qtde_palavras'] = resenhas_treino['interjeicoes_x_qtde_palavras']
# visualizar amostra
resenhas_tfidf.head()
# função para gerar relatório de desempenho
def relatorio_desempenho(y_true_treino, y_prev_treino, y_true_teste, y_prev_teste, y_proba_teste):
    print('======= RELATÓRIO DE DESEMPENHO =======')
    acuracia = round(accuracy_score(y_true_treino, y_prev_treino) * 100, 4)
    f1       = round(f1_score(y_true_treino, y_prev_treino, average='weighted') * 100, 4)
    print('Apurado nos dados de treino')
    print('---------------------------')
    print('Acurácia: {}%'.format(acuracia))
    print('F1 score: {}%'.format(f1))
    
    print('\nApurado nos dados de teste')
    print('----------------------------')
    acuracia = round(accuracy_score(y_true_teste, y_prev_teste) * 100, 4)
    f1       = round(f1_score(y_true_teste, y_prev_teste, average='weighted') * 100, 4)
    print('Acurácia: {}%'.format(acuracia))
    print('F1 score: {}%'.format(f1))

    # Matriz de confusão
    skplt.metrics.plot_confusion_matrix(y_true_teste, y_prev_teste, figsize=(7,7), cmap='Greens')
    
    # 'Curva ROC nos dados de teste'
    skplt.metrics.plot_roc(y_true_teste, y_proba_teste, figsize=(7,7))
    print('\n===== FIM RELATÓRIO DE DESEMPENHO =====')
# dividir os dados em treino e teste
y = resenhas_treino.comentario_negativo
X_train, X_test, y_train, y_test = train_test_split(resenhas_tfidf.fillna(0), y, test_size=0.1, stratify=y, random_state=42)
# colocar os dados na mesma escala
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
# como a variável a ser predita está desbalanceada, criar dados sintéticos para equilibrar

oversample = SMOTE()
X_over, y_over = oversample.fit_resample(X_train, y_train)

# novo verfil dos dados de treino
X_over.shape, y_over.shape
%%time
modelo = MultinomialNB()

param_grid = {
    'alpha': [0.001, 0.005, 0.1, 0.5, 1.0]
}

classificador = RandomizedSearchCV(modelo, 
                               param_distributions=param_grid, 
                               scoring='accuracy', 
                               n_jobs=-1, 
                               cv=5, 
                               refit=True,
                               random_state=42)

search_nb  = classificador.fit(X_over, y_over)
ypred_nb   = classificador.predict(X_over)
y_pred     = classificador.predict(X_test)
y_proba_nb = search_nb.predict_proba(X_test)

# Relatório de desempenho
relatorio_desempenho(y_over, ypred_nb, y_test, y_pred, y_proba_nb)

print(search_nb.best_params_)
print('\n')
%%time
modelo = LogisticRegression()

param_grid = {
    'solver': ['sag', 'saga', 'newton-cg', 'lbfgs'],
    'C': [0.1, 1.0, 100.0, 200.0, 300.0, 400,0, 500.0]
}

classificador = RandomizedSearchCV(modelo, 
                               param_distributions=param_grid, 
                               scoring='accuracy', 
                               n_jobs=-1, 
                               cv=5, 
                               refit=True,
                               random_state=42)

search_lr  = classificador.fit(X_over, y_over)
ypred_lr   = classificador.predict(X_over)
y_pred     = classificador.predict(X_test)
y_proba_lr = search_lr.predict_proba(X_test)

# Relatório de desempenho
relatorio_desempenho(y_over, ypred_lr, y_test, y_pred, y_proba_lr)
print(search_lr.best_params_)
%%time

modelo = RandomForestClassifier()

param_grid = {
    'n_estimators': [700, 800, 900],
    'criterion': ['gini', 'entropy']
}

classificador = RandomizedSearchCV(modelo, 
                               param_distributions=param_grid, 
                               scoring='accuracy', 
                               n_jobs=-1, 
                               cv=5, 
                               refit=True,
                               random_state=42)

search_rf  = classificador.fit(X_over, y_over)
ypred_rf   = classificador.predict(X_over)
y_pred     = classificador.predict(X_test)
y_proba_rf = search_lr.predict_proba(X_test)

# Relatório de desempenho
relatorio_desempenho(y_over, ypred_lr, y_test, y_pred, y_proba_rf)
print(search_rf.best_params_)
%%time
modelo = GradientBoostingClassifier()

param_grid = {
    'n_estimators': [1100, 1200, 1300]
}

classificador = RandomizedSearchCV(modelo, 
                               param_distributions=param_grid, 
                               scoring='accuracy', 
                               n_jobs=-1, 
                               cv=5, 
                               refit=True,
                               random_state=42)

search_gb  = classificador.fit(X_over, y_over)
ypred_gb   = classificador.predict(X_over)
y_pred     = classificador.predict(X_test)
y_proba_gb = search_lr.predict_proba(X_test)

# Relatório de desempenho
relatorio_desempenho(y_over, ypred_lr, y_test, y_pred, y_proba_gb)
print(search_gb.best_params_)
# dividir os dados em treino e teste
dados = resenhas_treino[['coment_processado', 'comentario_negativo']].copy()

y = dados['comentario_negativo']
X = dados['coment_processado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
# colocando os dados em formato resenha-dicionário
ls_treino = []
ls_teste  = []

# dados de treino
for texto, emocao in zip(X_train, y_train):
    if emocao == 1:
        dic = ({'NAO_NEGATIVO': False, 'SIM_NEGATIVO': True})
    elif emocao == 0:
        dic = ({'NAO_NEGATIVO': True, 'SIM_NEGATIVO': False})

    ls_treino.append([texto, dic.copy()])
    

# dados de teste
for texto, emocao in zip(X_test, y_test):
    if emocao == 1:
        dic = ({'NAO_NEGATIVO': False, 'SIM_NEGATIVO': True})
    elif emocao == 0:
        dic = ({'NAO_NEGATIVO': True, 'SIM_NEGATIVO': False})

    ls_teste.append([texto, dic.copy()])
    
    
# tamanho:
len(ls_treino), len(ls_teste)
# visualizar alguns registros
ls_treino[0:5]
# cria o modelo em branco
modelo = spacy.blank('pt')

# criação das classes
categorias = modelo.create_pipe("textcat") # constante: textcat
categorias.add_label("NAO_NEGATIVO")
categorias.add_label("SIM_NEGATIVO")
modelo.add_pipe(categorias)
historico = []
# instancia o modelo de deep learning do spaCy
modelo.begin_training()

for epoca in range(20): # serão usadas 20 épocas
    random.shuffle(ls_treino)
    losses = {}

    for batch in spacy.util.minibatch(ls_treino, 512): # atualiza a loss a cada 512 registros
        textos = [modelo(texto) for texto, entities in batch]
        annotations = [{'cats': entities} for texto, entities in batch]
        modelo.update(textos, annotations, losses=losses)
        historico.append(losses)
        
    if epoca % 4 == 0:
        print(losses)
# armazenar o histórico na loss function em uma lista
historico_loss = []

for i in historico:
    historico_loss.append(i.get('textcat'))
# transformar a lista do histórico da loss function em array do Numpy
historico_loss = np.array(historico_loss)
historico_loss[0]
# visualizar o comportamento da loss function
import matplotlib.pyplot as plt

plt.plot(historico_loss)
plt.title('Progressão do erro')
plt.xlabel('Batches')
plt.ylabel('Erro')
# gerando as previsões
prev_treino = []
prev_teste  = []

for texto in X_train:
    previsao = modelo(texto)
    prev_treino.append(previsao.cats)
    
for texto in X_test:
    previsao = modelo(texto)
    prev_teste.append(previsao.cats)
# criar previsões em formato binário
prev_treino_binario = []
prev_teste_binario  = []

for previsao in prev_treino:
    if previsao['SIM_NEGATIVO'] > previsao['NAO_NEGATIVO']:
        prev_treino_binario.append(1)
    else:
        prev_treino_binario.append(0)

for previsao in prev_teste:
    if previsao['SIM_NEGATIVO'] > previsao['NAO_NEGATIVO']:
        prev_teste_binario.append(1)
    else:
        prev_teste_binario.append(0)

# transforma a lista em array do Numpy
prev_treino_binario = np.array(prev_treino_binario)
prev_teste_binario  = np.array(prev_teste_binario)
# criar a previsão em formato de probabilidades
prev_teste_proba = []

for previsao in prev_teste:
    prev_teste_proba.append([
        round(previsao['NAO_NEGATIVO'], 8), 
        round(previsao['SIM_NEGATIVO'], 8)])

prev_teste_proba = np.array(prev_teste_proba)
relatorio_desempenho(y_train, prev_treino_binario, y_test, prev_teste_binario, prev_teste_proba)
# Criar matriz esparsa a partir do vetorizador nos dados de validação
X = vect.transform(resenhas_validacao.coment_processado)

# Gerar Pandas DataFrame para os dados de validação
resenhas_tfidf = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
resenhas_tfidf.head()
%%time

# criar novas variáeis
resenhas_validacao['qtde_palavras']    = resenhas_validacao['Comentario'].apply(qtde_palavras)
resenhas_validacao['qtde_maiusculas']  = resenhas_validacao['Comentario'].apply(qtde_maiusculas)
resenhas_validacao['comprimento']      = resenhas_validacao['Comentario'].apply(comprimento)
resenhas_validacao['maiusc_x_compri']  = resenhas_validacao['qtde_maiusculas'] / resenhas_validacao['comprimento']
resenhas_validacao['qtde_exclamacoes'] = resenhas_validacao['Comentario'].apply(qtde_exclamacoes)
resenhas_validacao['qtde_interrogacoes'] = resenhas_validacao['Comentario'].apply(qtde_interrogacoes)
resenhas_validacao['qtde_pontuacoes']    = resenhas_validacao['Comentario'].apply(qtde_pontuacoes)
resenhas_validacao['qtde_simbolos']      = resenhas_validacao['Comentario'].apply(qtde_simbolos)
resenhas_validacao['qtde_palavras_unicas'] = resenhas_validacao['Comentario'].apply(qtde_palavras_unicas)
resenhas_validacao['unicas_x_comprimento'] = resenhas_validacao['qtde_palavras_unicas'] / resenhas_validacao['comprimento']
resenhas_validacao['qtde_substantivos'], resenhas_validacao['qtde_adjetivos'], resenhas_validacao['qtde_verbos'], resenhas_validacao['qtde_adverbios'], resenhas_validacao['qtde_interjeicoes'] = zip(*resenhas_validacao['Comentario'].apply(lambda comment: qtde_tag_part_of_speech(comment)))

resenhas_validacao['substantivos_vs_comprimento']  = resenhas_validacao['qtde_substantivos'] / resenhas_validacao['comprimento']
resenhas_validacao['adjectivos_x_comprimento']     = resenhas_validacao['qtde_adjetivos'] / resenhas_validacao['comprimento']
resenhas_validacao['verbos_x_comprimento']         = resenhas_validacao['qtde_verbos'] /resenhas_validacao['comprimento']
resenhas_validacao['adverbios_x_comprimento']      = resenhas_validacao['qtde_adverbios'] /resenhas_validacao['comprimento']
resenhas_validacao['interjeicoes_x_comprimento']   = resenhas_validacao['qtde_interjeicoes'] /resenhas_validacao['comprimento']
resenhas_validacao['substantivos_x_qtde_palavras'] = resenhas_validacao['qtde_substantivos'] / resenhas_validacao['qtde_palavras']
resenhas_validacao['adjectivos_x_qtde_palavras']   = resenhas_validacao['qtde_adjetivos'] / resenhas_validacao['qtde_palavras']
resenhas_validacao['verbos_x_qtde_palavras']       = resenhas_validacao['qtde_verbos'] / resenhas_validacao['qtde_palavras']
resenhas_validacao['adverbios_x_qtde_palavras']    = resenhas_validacao['qtde_adverbios'] / resenhas_validacao['qtde_palavras']
resenhas_validacao['interjeicoes_x_qtde_palavras'] = resenhas_validacao['qtde_interjeicoes'] / resenhas_validacao['qtde_palavras']
# incluir novas variáveis no cojunto de dados tfidf

resenhas_tfidf['qtde_palavras']                = resenhas_validacao['qtde_palavras']                 
resenhas_tfidf['qtde_maiusculas']              = resenhas_validacao['qtde_maiusculas']               
resenhas_tfidf['comprimento']                  = resenhas_validacao['comprimento']                   
resenhas_tfidf['maiusc_x_compri']              = resenhas_validacao['maiusc_x_compri']               
resenhas_tfidf['qtde_exclamacoes']             = resenhas_validacao['qtde_exclamacoes']              
resenhas_tfidf['qtde_interrogacoes']           = resenhas_validacao['qtde_interrogacoes']            
resenhas_tfidf['qtde_pontuacoes']              = resenhas_validacao['qtde_pontuacoes']               
resenhas_tfidf['qtde_simbolos']                = resenhas_validacao['qtde_simbolos']                 
resenhas_tfidf['qtde_palavras_unicas']         = resenhas_validacao['qtde_palavras_unicas']          
resenhas_tfidf['unicas_x_comprimento']         = resenhas_validacao['unicas_x_comprimento']          
resenhas_tfidf['qtde_substantivos']            = resenhas_validacao['qtde_substantivos']                         
resenhas_tfidf['qtde_adjetivos']               = resenhas_validacao['qtde_adjetivos']                    
resenhas_tfidf['qtde_verbos']                  = resenhas_validacao['qtde_verbos']                         
resenhas_tfidf['qtde_adverbios']               = resenhas_validacao['qtde_adverbios']              
resenhas_tfidf['qtde_interjeicoes']            = resenhas_validacao['qtde_interjeicoes']              
resenhas_tfidf['substantivos_vs_comprimento']  = resenhas_validacao['substantivos_vs_comprimento']   
resenhas_tfidf['adjectivos_x_comprimento']     = resenhas_validacao['adjectivos_x_comprimento']      
resenhas_tfidf['verbos_x_comprimento']         = resenhas_validacao['verbos_x_comprimento']          
resenhas_tfidf['interjeicoes_x_comprimento']   = resenhas_validacao['adverbios_x_comprimento']
resenhas_tfidf['interjeicoes_x_comprimento']   = resenhas_validacao['interjeicoes_x_comprimento']
resenhas_tfidf['substantivos_x_qtde_palavras'] = resenhas_validacao['substantivos_x_qtde_palavras']  
resenhas_tfidf['adjectivos_x_qtde_palavras']   = resenhas_validacao['adjectivos_x_qtde_palavras']    
resenhas_tfidf['verbos_x_qtde_palavras']       = resenhas_validacao['verbos_x_qtde_palavras']       
resenhas_tfidf['adverbios_x_qtde_palavras']    = resenhas_validacao['adverbios_x_qtde_palavras']
resenhas_tfidf['interjeicoes_x_qtde_palavras'] = resenhas_validacao['interjeicoes_x_qtde_palavras']
# colocar os dados na mesma escala
validacao = scaler.transform(resenhas_tfidf)
# Fazer a predição
y_pred_binario = search_gb.predict(validacao)
y_pred_proba   = search_gb.predict_proba(validacao)
# função para gerar relatório de desempenho final
def relatorio_desempenho_final(y_true_teste, y_prev_teste, y_proba_teste):
    print('======= RELATÓRIO DE DESEMPENHO FINAL =======')  
    print('---------------------------------------------')
    acuracia = round(accuracy_score(y_true_teste, y_prev_teste) * 100, 4)
    f1       = round(f1_score(y_true_teste, y_prev_teste, average='weighted') * 100, 4)
    print('Acurácia: {}%'.format(acuracia))
    print('F1 score: {}%'.format(f1))

    # Matriz de confusão
    skplt.metrics.plot_confusion_matrix(y_true_teste, y_prev_teste, figsize=(7,7), cmap='Greens')
    
    # 'Curva ROC nos dados de teste'
    skplt.metrics.plot_roc(y_true_teste, y_proba_teste, figsize=(7,7))
    print('\n===== FIM RELATÓRIO DE DESEMPENHO FINAL =====')
relatorio_desempenho_final(resenhas_validacao.comentario_negativo, y_pred_binario, y_pred_proba)
# vetorizador
vect = TfidfVectorizer(stop_words=ls_stop_words, # define as stop-words
                       ngram_range=(1, 2),       # gerar unigramas e bigramas
                       max_features=300          # selecionar as 300 features (variáveis) mais importantes
                      ).fit(resenhas_treino.coment_processado)
# Criar matriz esparsa a partir do vetorizador
X = vect.transform(resenhas_treino.coment_processado)
# Gerar Pandas DataFrame
resenhas_tfidf = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
# anexar os dados
resenhas_tfidf['qtde_palavras']                = resenhas_treino['qtde_palavras']                 
resenhas_tfidf['qtde_maiusculas']              = resenhas_treino['qtde_maiusculas']               
resenhas_tfidf['comprimento']                  = resenhas_treino['comprimento']                   
resenhas_tfidf['maiusc_x_compri']              = resenhas_treino['maiusc_x_compri']               
resenhas_tfidf['qtde_exclamacoes']             = resenhas_treino['qtde_exclamacoes']              
resenhas_tfidf['qtde_interrogacoes']           = resenhas_treino['qtde_interrogacoes']            
resenhas_tfidf['qtde_pontuacoes']              = resenhas_treino['qtde_pontuacoes']               
resenhas_tfidf['qtde_simbolos']                = resenhas_treino['qtde_simbolos']                 
resenhas_tfidf['qtde_palavras_unicas']         = resenhas_treino['qtde_palavras_unicas']          
resenhas_tfidf['unicas_x_comprimento']         = resenhas_treino['unicas_x_comprimento']          
resenhas_tfidf['qtde_substantivos']            = resenhas_treino['qtde_substantivos']                         
resenhas_tfidf['qtde_adjetivos']               = resenhas_treino['qtde_adjetivos']                    
resenhas_tfidf['qtde_verbos']                  = resenhas_treino['qtde_verbos']                         
resenhas_tfidf['qtde_adverbios']               = resenhas_treino['qtde_adverbios']              
resenhas_tfidf['qtde_interjeicoes']            = resenhas_treino['qtde_interjeicoes']              
resenhas_tfidf['substantivos_vs_comprimento']  = resenhas_treino['substantivos_vs_comprimento']   
resenhas_tfidf['adjectivos_x_comprimento']     = resenhas_treino['adjectivos_x_comprimento']      
resenhas_tfidf['verbos_x_comprimento']         = resenhas_treino['verbos_x_comprimento']          
resenhas_tfidf['interjeicoes_x_comprimento']   = resenhas_treino['adverbios_x_comprimento']
resenhas_tfidf['interjeicoes_x_comprimento']   = resenhas_treino['interjeicoes_x_comprimento']
resenhas_tfidf['substantivos_x_qtde_palavras'] = resenhas_treino['substantivos_x_qtde_palavras']  
resenhas_tfidf['adjectivos_x_qtde_palavras']   = resenhas_treino['adjectivos_x_qtde_palavras']    
resenhas_tfidf['verbos_x_qtde_palavras']       = resenhas_treino['verbos_x_qtde_palavras']       
resenhas_tfidf['adverbios_x_qtde_palavras']    = resenhas_treino['adverbios_x_qtde_palavras']
resenhas_tfidf['interjeicoes_x_qtde_palavras'] = resenhas_treino['interjeicoes_x_qtde_palavras']
# aplicar normalizador e oversample
X = scaler.transform(resenhas_tfidf.fillna(0))
y = resenhas_treino.comentario_negativo

oversample = SMOTE()
X_over, y_over = oversample.fit_resample(X, y)
# treinar o modelo
modelo = GradientBoostingClassifier(n_estimators=1300)
modelo.fit(X_over, y_over)
# Fazer a predição
y_pred_binario = search_gb.predict(validacao)
y_pred_proba   = search_gb.predict_proba(validacao)
# checar o resultado
relatorio_desempenho_final(resenhas_validacao.comentario_negativo, y_pred_binario, y_pred_proba)
# Resenha recuperada na Google Play em 17/09/2020 para o app do Banco do Brasil
# foi atribuído 5 estrelas
resenha = 'Muito útil e eficiente, esse App! Parabéns Banco do Brasil!!!!👍👍👍'
# processar o texto com Spacy
documento = pln(resenha)
# Exibir entidades da resenha
from spacy import displacy
displacy.render(documento, style = 'ent', jupyter = True)
# Exibir dependências
displacy.render(documento, style='dep', jupyter=True, options={'distance': 90})
d = {'Comentario': [resenha]}
resenha_df = pd.DataFrame(data=d)
resenha_df.head()
# aplicar a função de préprocessamento
resenha_df['coment_processado'] = resenha_df['Comentario'].apply(preprocessamento)
# vetorizar os dados
X = vect.transform(resenha_df.coment_processado)
# Gerar Pandas DataFrame
resenhas_tfidf = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
# criar novas variáeis
resenha_df['qtde_palavras']    = resenha_df['Comentario'].apply(qtde_palavras)
resenha_df['qtde_maiusculas']  = resenha_df['Comentario'].apply(qtde_maiusculas)
resenha_df['comprimento']      = resenha_df['Comentario'].apply(comprimento)
resenha_df['maiusc_x_compri']  = resenha_df['qtde_maiusculas'] / resenha_df['comprimento']
resenha_df['qtde_exclamacoes'] = resenha_df['Comentario'].apply(qtde_exclamacoes)
resenha_df['qtde_interrogacoes'] = resenha_df['Comentario'].apply(qtde_interrogacoes)
resenha_df['qtde_pontuacoes']    = resenha_df['Comentario'].apply(qtde_pontuacoes)
resenha_df['qtde_simbolos']      = resenha_df['Comentario'].apply(qtde_simbolos)
resenha_df['qtde_palavras_unicas'] = resenha_df['Comentario'].apply(qtde_palavras_unicas)
resenha_df['unicas_x_comprimento'] = resenha_df['qtde_palavras_unicas'] / resenha_df['comprimento']
resenha_df['qtde_substantivos'], resenha_df['qtde_adjetivos'], resenha_df['qtde_verbos'], resenha_df['qtde_adverbios'], resenha_df['qtde_interjeicoes'] = zip(*resenha_df['Comentario'].apply(lambda comment: qtde_tag_part_of_speech(comment)))

resenha_df['substantivos_vs_comprimento']  = resenha_df['qtde_substantivos'] / resenha_df['comprimento']
resenha_df['adjectivos_x_comprimento']     = resenha_df['qtde_adjetivos'] / resenha_df['comprimento']
resenha_df['verbos_x_comprimento']         = resenha_df['qtde_verbos'] /resenha_df['comprimento']
resenha_df['adverbios_x_comprimento']      = resenha_df['qtde_adverbios'] /resenha_df['comprimento']
resenha_df['interjeicoes_x_comprimento']   = resenha_df['qtde_interjeicoes'] /resenha_df['comprimento']
resenha_df['substantivos_x_qtde_palavras'] = resenha_df['qtde_substantivos'] / resenha_df['qtde_palavras']
resenha_df['adjectivos_x_qtde_palavras']   = resenha_df['qtde_adjetivos'] / resenha_df['qtde_palavras']
resenha_df['verbos_x_qtde_palavras']       = resenha_df['qtde_verbos'] / resenha_df['qtde_palavras']
resenha_df['adverbios_x_qtde_palavras']    = resenha_df['qtde_adverbios'] / resenha_df['qtde_palavras']
resenha_df['interjeicoes_x_qtde_palavras'] = resenha_df['qtde_interjeicoes'] / resenha_df['qtde_palavras']
# anexar os dados
resenhas_tfidf['qtde_palavras']                = resenha_df['qtde_palavras']                 
resenhas_tfidf['qtde_maiusculas']              = resenha_df['qtde_maiusculas']               
resenhas_tfidf['comprimento']                  = resenha_df['comprimento']                   
resenhas_tfidf['maiusc_x_compri']              = resenha_df['maiusc_x_compri']               
resenhas_tfidf['qtde_exclamacoes']             = resenha_df['qtde_exclamacoes']              
resenhas_tfidf['qtde_interrogacoes']           = resenha_df['qtde_interrogacoes']            
resenhas_tfidf['qtde_pontuacoes']              = resenha_df['qtde_pontuacoes']               
resenhas_tfidf['qtde_simbolos']                = resenha_df['qtde_simbolos']                 
resenhas_tfidf['qtde_palavras_unicas']         = resenha_df['qtde_palavras_unicas']          
resenhas_tfidf['unicas_x_comprimento']         = resenha_df['unicas_x_comprimento']          
resenhas_tfidf['qtde_substantivos']            = resenha_df['qtde_substantivos']                         
resenhas_tfidf['qtde_adjetivos']               = resenha_df['qtde_adjetivos']                    
resenhas_tfidf['qtde_verbos']                  = resenha_df['qtde_verbos']                         
resenhas_tfidf['qtde_adverbios']               = resenha_df['qtde_adverbios']              
resenhas_tfidf['qtde_interjeicoes']            = resenha_df['qtde_interjeicoes']              
resenhas_tfidf['substantivos_vs_comprimento']  = resenha_df['substantivos_vs_comprimento']   
resenhas_tfidf['adjectivos_x_comprimento']     = resenha_df['adjectivos_x_comprimento']      
resenhas_tfidf['verbos_x_comprimento']         = resenha_df['verbos_x_comprimento']          
resenhas_tfidf['interjeicoes_x_comprimento']   = resenha_df['adverbios_x_comprimento']
resenhas_tfidf['interjeicoes_x_comprimento']   = resenha_df['interjeicoes_x_comprimento']
resenhas_tfidf['substantivos_x_qtde_palavras'] = resenha_df['substantivos_x_qtde_palavras']  
resenhas_tfidf['adjectivos_x_qtde_palavras']   = resenha_df['adjectivos_x_qtde_palavras']    
resenhas_tfidf['verbos_x_qtde_palavras']       = resenha_df['verbos_x_qtde_palavras']       
resenhas_tfidf['adverbios_x_qtde_palavras']    = resenha_df['adverbios_x_qtde_palavras']
resenhas_tfidf['interjeicoes_x_qtde_palavras'] = resenha_df['interjeicoes_x_qtde_palavras']
# normalizar os dados
X = scaler.transform(resenhas_tfidf.fillna(0))
# probabilidade de ser crítica
round(search_gb.predict_proba(X)[0][1], 4) * 100