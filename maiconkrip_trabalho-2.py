import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import io
#abrir Tweets.csv diretamente de uma pasta no seu computador.
#Executar a celula e usar o "choose files" para selecionar o arquivo "Tweets.csv"
pd.options.display.max_colwidth = 255
df = pd.read_csv('../input/airline-sentiment/Tweets.csv')
#visualizar tweets
df.head(2)
# Exemplo de comentários. 
comments = df['text']
comments.head(30)
#Importar modelo do USE diretamente do tensorflow hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

#função que recebe os tweets pré-processados e retorna os embeddings
def embed(input):
    return model(input)
# Analise dos dados
print('Head: ',df.columns)
# Dimensões do dataset
print("\nShape: ", df.shape)
# Descrição do dataset
print('\nDescrição:')
print(df.describe())
# Distribuição do dataset
import seaborn as sns
sns.countplot(x='airline_sentiment', data=df)
# Balançeamento para que os plot dos clusters fique melhor. 
df['airline_sentiment'].value_counts()

df_neg = df.query("airline_sentiment == 'negative'").head(1000).copy()
df_neu = df.query("airline_sentiment == 'neutral'").head(1000).copy()
df_pos = df.query("airline_sentiment == 'positive'").head(1000).copy()

df_balanced = pd.concat([df_neg,df_neu,df_pos], ignore_index=True)
df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

comments = df_balanced['text'].copy()

sns.countplot(x='airline_sentiment', data=df_balanced)

#Para usar o dataset inteiro, descomentar:
#df_balanced = df.copy()
df = df_balanced.copy()
# Para usar a função de pré processamento, observar os parâmetros. 

# Parâmetros:
# data = dataframe
# html = remove caracteres hmtl (s/n)
# emoji = remove emojis
# punct = remove pontuação (s/n)
# lower =  tranforma para caixa baixa (s/n)
# stopw = remove stopwords (s/n)
# lng = linguagem para stopwords (padrão english)
# token = realiza tokenize (s/n)
# stm = realiza Stemming (s/n)
# lmz = realiza Lemmatizing (s/n)

#retorno dataframe limpo

def limpa_dataframe(data, column, hmtl='s', emoji = 's', punct = 's', lower = 's', stopw = 's', lng = 'english', token = 's', stm = 's', lmz = 's'):
    
    import nltk
    from nltk import word_tokenize
    from nltk.stem.porter import PorterStemmer        
    from nltk.stem import WordNetLemmatizer
    import bs4
    import string
    
    wn = nltk.WordNetLemmatizer()    
    ps = nltk.PorterStemmer()
    stopword = nltk.corpus.stopwords.words(lng)

    #Removendo Tag HTML
    if (hmtl =='s'):
        data[column] = data[column].apply(lambda x: bs4.BeautifulSoup(x, 'lxml').get_text())
       
    # Remove Emojis
    def deEmojify(inputString):
        import unicodedata
        from unidecode import unidecode
        
        returnString = ""

        for character in inputString:
            try:
                character.encode("ascii")
                returnString += character
            except UnicodeEncodeError:
                replaced = unidecode(str(character))
                if replaced != '':
                    returnString += replaced
                else:
                    try:
                         returnString += "[" + unicodedata.name(character) + "]"
                    except ValueError:
                         returnString += "[x]"

        return returnString
    if(emoji=='s'):
        data[column] = data[column].apply(lambda x: deEmojify(x))

    
    # Removendo a pontuação
    def remove_punct(text):
        text_nopunct = "".join([char for char in text if char not in string.punctuation])
        return text_nopunct
    
    if (punct == 's'):
        data[column] = data[column].apply(lambda x: remove_punct(x))
    
    #Caixa baixa
    if (lower == 's'):
        data[column] = [token.lower() for token in data[column]]
    
    #Tokenização
    if (token == 's'):
        data[column] = [word_tokenize(word) for word in data[column]]
    
    #StopWords
    def remove_stopwords(tokenized_list):
        text = [word for word in tokenized_list if word not in stopword]
        return text
    
    if(stopw == 's'):
        data[column] = data[column].apply(lambda x: remove_stopwords(x))
    
    #Steeming   
    def stemming(tokenized_text):
        text = [ps.stem(word) for word in tokenized_text]
        return text
    
    if (stm == 's'):
        data[column] = data[column].apply(lambda x: stemming(x))

    
    #Lemmatizing
    def lemmatizing(tokenized_text):
        text = [wn.lemmatize(word) for word in tokenized_text]
        return text
    
    if (lmz == 's'):
        data[column] = data[column].apply(lambda x: lemmatizing(x))
    
    data[column] = [' '.join(word) for word in data[column]]


    return data
#Cria um dataset so com os dados limpos. 
df_clean = df_balanced.copy()
df_clean  = limpa_dataframe(df_clean, column = 'text', hmtl='s', emoji = 's', punct = 's', stopw = 's', lng = 'english', token = 's', stm = 's', lmz = 's')
df_clean['text'].head(30)
# Cria o embbeding
messages = df_clean['text']

message_embeddings = embed(messages)

# Converte em duas dimensões para plotar. TNSE
from sklearn.manifold import TSNE

tsne = TSNE(n_components = 2, perplexity = 5)
tsne_result = tsne.fit_transform(message_embeddings)
tsne_df = pd.DataFrame ({'X':tsne_result[:,0],
                         'Y':tsne_result[:,1]})

sns.scatterplot(x = 'X', y = 'Y',
                data = tsne_df
               )
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster

#Determina o X
#X = message_embeddings
# Só consegui seguir usando um dataset com duas dimensões. Acima disso não foi possível. 
X = tsne_result

#Conforme dito em aula
Z = linkage(X, 'average', 'cosine')

# Clusteriza
# Não sei se é isso que foi falado em aula em usar o -1. 
threshold = 1-0.9

c = fcluster(Z, threshold, criterion="distance")    

# Clusters encontrados. 
myset = set(c.tolist())
my_list = list(set(myset))
print('Clusters Encontrados')
print(len(my_list))
# Plot do resultado da clusterização. 

plt.figure(figsize=(10, 8))
plt.scatter(X[:,0], X[:,1], c=c, cmap='prism')  
plt.show()
# Dendrogram da clusterização
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=100,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()
# K-Means para referência. Me parece que serve pra essa necessidade. 
from sklearn.cluster import KMeans
clf_k  = KMeans(n_clusters= 3, init='k-means++', max_iter=300,n_init=5, random_state=0)
pred_y_k = clf_k.fit_predict(X)


sns.scatterplot(x = 'X', y = 'Y',
                hue = pred_y_k,
                palette = ['red','orange','blue'],
                data = tsne_df
               )
#EM - GaussianMixture para referência. Não consegui fazer direito. DBSCAN também não consegui fazer funcionar. 

from sklearn.mixture import GaussianMixture
clf_em = GaussianMixture(n_components=5, init_params='random', covariance_type='full')
pred_y_em = clf_em.fit_predict(X)

# Clusters encontrados. 
myset = set(pred_y_em.tolist())
my_list = list(set(myset))
print(len(my_list))

sns.scatterplot(x = 'X', y = 'Y',
                hue = pred_y_em,
                data = tsne_df
               )
#cria um df com os comentarios e seu cluster para comparação. 
export = pd.DataFrame()
export['Comentario'] = df['text']
export['cluster'] = c
print(len(df))
sns.countplot(x='cluster', data=export)

# Estou pegando os comentarios originais, com emojos, tags, etc. 
#Aqui vejo positivos e neutros em maioria. Me parece serem na maioria dúvidas. 
pd.options.display.max_colwidth = 255
print(export.query("cluster == 1")['Comentario'].head(10))
# Aqui não entendi as relaçãoes. Ao meu ver parecem ser em maioria neutras para ruins. Tem a ver com experiencias de voo. 
print(export.query("cluster == 2")['Comentario'].head(10))
# São geralmente positivos. Parece ter relação com tarifas, preços e e atendimento. 
print(export.query("cluster == 3")['Comentario'].head(10))