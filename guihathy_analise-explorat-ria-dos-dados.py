%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
df = pd.read_csv("../input/olist_classified_public_dataset.csv")
df.head(6)
#Quantidade de linhas e colunas do DataFrame
df.shape
# 3584 linhas e 34 colunas
df["product_category_name"].describe()
#Todas as categorias de produtos
valueCategoria = df["product_category_name"].value_counts()
valueCategoria
print(valueCategoria[0:6])
print("Somatorio: %s" % valueCategoria[0:6].sum())
valueCategoria[0:6].sum()/valueCategoria.sum()
MEAN = df["review_score"].mean()
print("Média da população %s" % MEAN)
std = df["review_score"].std()
print("Desvio Padrão da população: %s" % std)
df.groupby(['product_category_name'])["review_score"].describe().sort_values(['count'], ascending=False)
dfFiltrado = df[(df.product_category_name == "cama_mesa_banho") | 
                (df.product_category_name == "moveis_decoracao")|
                (df.product_category_name == "beleza_saude")|
                (df.product_category_name == "esporte_lazer")|
                (df.product_category_name == "informatica_acessorios")|
                (df.product_category_name == "utilidades_domesticas")]


dfFiltrado.groupby(['product_category_name'])["most_voted_class"].value_counts(normalize=True)
dfInformatica = df[df["product_category_name"] == "informatica_acessorios"]
dfInformatica.head(2)
## Pegando as categorias problemas_de_entrega e problemas_de_qualidade

dfInformatica[(dfInformatica["most_voted_class"] == "problemas_de_entrega") |
              (dfInformatica["most_voted_class"] == "problemas_de_qualidade")]["most_voted_subclass"].value_counts()
print(df[df["most_voted_class"]== "satisfeito_com_pedido"]["product_photos_qty"].describe())
print(dfInformatica[(dfInformatica["most_voted_class"] == "problemas_de_qualidade") | (dfInformatica["most_voted_class"] == "diferente_do_anunciado")]["product_photos_qty"].describe())
df.corr()
df[["review_score", "product_photos_qty", "product_description_lenght", "order_products_value", "order_freight_value", "order_items_qty"]].corr()
# Classificações e suas frequencias
freqClass = df["most_voted_class"].value_counts()
freqClass
print("%s" % ((1983 / freqClass.sum())* 100))
print("%s" % ((950 / freqClass.sum())* 100))
print("%s" % ((480 / freqClass.sum())* 100))
## Plotando as frequencias
sns.barplot(freqClass.index, freqClass.values)
plt.show()
# Criando um novo data frame apenas com os problemas de qualidade
dfQuali = df[df["most_voted_class"] == "problemas_de_qualidade"]
freqSubClassQual = dfQuali["most_voted_subclass"].value_counts()
freqSubClassQual
plt.figure()
subcategorias = ["diferente_do_anunciado", "baixa_qualidade", "devolucao", "outro_pedido"]
for subcategoria in subcategorias:
    dfGenerico = dfQuali[dfQuali["most_voted_subclass"] == subcategoria]
    review = (" ".join(dfGenerico["review_comment_message"])).lower().replace("produtoq", "produto")
    review
    # Removendo a pontuação e Tokenizando o texto
    tokenizer = RegexpTokenizer(r'\w+')
    cleanText = tokenizer.tokenize(review)

    # Removendo as StopWords
    stopWords = stopwords.words('portuguese')
    stopWords.remove("não")
    filtered_words = [word for word in cleanText if word not in stopWords]

    # Criando Bigramas
    filtered_words = nltk.ngrams(filtered_words, 2)

    filtered_words

    freq = nltk.FreqDist(filtered_words)
    plt.figure(figsize=(10, 5))
    plt.title(subcategoria)
    freq.plot(30)
# Criando um novo data frame apenas com os problemas de entrega
dfEntrega = df[df["most_voted_class"] == "problemas_de_entrega"]
freqSubClassEntre = dfEntrega["most_voted_subclass"].value_counts()
freqSubClassEntre
# Frequencia dos estados que mais tiveram problemas com a entrega
freqAtraso = dfEntrega[dfEntrega["most_voted_subclass"] == "atrasado"]["customer_state"].value_counts()
freqAtraso
print((134/freqAtraso.sum()) * 100)
print((108/freqAtraso.sum()) * 100)
print(((134/freqAtraso.sum()) * 100) + ((108/freqAtraso.sum()) * 100))
# Cidades de São Paulo com o maior numero de atrasos
dfEntrega[dfEntrega["customer_state"] == "SP"]["customer_city"].value_counts()
X = df[["order_products_value", "order_freight_value"]]
y = df["review_score"]
# Separando a nossa base em validação e treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y)
mlp = MLPClassifier(hidden_layer_sizes=(13),max_iter=5000)
mlp.fit(X_train,y_train)
MLPClassifier()
predictions = mlp.predict(X_test)
# Criando a matriz de confusão
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))