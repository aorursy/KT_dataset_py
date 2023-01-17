import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
data = pd.read_excel('../input/geo.xlsx')
data_p = data[data.comment_class==1]

data_n = data[data.comment_class==-1]

data.head()
data_n.info()
%matplotlib inline

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'

from pylab import rcParams
# отобразим плохие и хорошие голоса на карте

rcParams['figure.figsize'] = (5, 5)



fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,5))           # quantity rows and columns (and size)



ax1.scatter(x=data.x, y=data.y, alpha=0.05)

ax2.scatter(x=data_p.x, y=data_p.y, alpha=0.05, color='g')

ax3.scatter(x=data_n.x, y=data_n.y, alpha=0.05, color='r')



titles = ["All", "Positive", "Negative"]

colors = ["b", "g", "r"]

all_data = [data, data_p, data_n]



for number, title in zip (fig.axes, titles):

    number.set_title(title, fontsize=16)
from sklearn.cluster import KMeans, DBSCAN
data_prob = data.drop('comment_class', axis=1)

km = KMeans(n_clusters=8)

km.fit(data_prob)

clusters = km.predict(data_prob)
for i in range(8):

    print (i,'cluster =',len(data_prob[clusters==i]))
rcParams['figure.figsize'] = (10, 10)



#            РАБОЧИЙ, НО БОЛЕЕ ДЛИННЫЙ ВАРИАНТ

# for i,color in zip(range(8),{'blue','red','green','black','orange','yellow','brown','orchid','lime'}):

#     x_i = data_prob.x[clusters==i]

#     y_i = data_prob.y[clusters==i]

#     plt.plot(x_i, y_i, 'ro', alpha=0.1, c=color)

    



plt.scatter(data_prob.x, data_prob.y, c=clusters, cmap='autumn', s=60)
k_inertia = []

ks = range(1,11)



for k in ks:

    clf_km = KMeans(n_clusters=k)

    clusters_km = clf_km.fit_predict(data_prob, )

    k_inertia.append(clf_km.inertia_/100)
rcParams['figure.figsize'] = (12,6)

plt.plot(ks, k_inertia)
diff = np.diff(k_inertia)           # np.diff - вычислить N-ю дискретную разность по заданной оси
plt.plot(ks[1:], diff)
diff_r = diff[1:] / diff[:-1]
plt.plot(ks[1:-1], diff_r)
k_opt = ks[np.argmin(diff_r)+1]

k_opt
''' Закоментированно, так как слишком долго грузит. Выдает: 0.56

from sklearn.metrics import silhouette_score

silhouette_score(data_prob, clusters)'''
from sklearn.cluster import KMeans
data_p.head(3)
data_p_pr = data_p.drop('comment_class', axis=1)

data_n_pr = data_n.drop('comment_class', axis=1)
kmeans = KMeans(n_clusters=4)

kmeans.fit(data_p_pr)

clusters_p = kmeans.predict(data_p_pr)
rcParams['figure.figsize'] = (7,7)

    

plt.scatter(data_p_pr.x, data_p_pr.y, c=clusters_p, alpha=0.1, cmap='jet', s=60)
kn_n = KMeans(n_clusters=8)

clusters_n = kn_n.fit_predict(data_n_pr)
rcParams['figure.figsize'] = (7,7)

plt.scatter(data_n_pr.x, data_n_pr.y, c=clusters_n, alpha=0.1, cmap='gnuplot', s=60)
from sklearn.cluster import KMeans, DBSCAN
eps_ = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0])

min_samples_= np.array([5, 10, 50, 100, 300, 500, 1000, 2000])
# Так как на всей выборки считать слишком долго, возьмем лишь часть

choice = np.random.choice(data_prob.index, size=30000, replace=False).tolist()

data_x = data_prob[data_prob.index.isin(choice)]
'''# посмотрим в каких диапазонах стоит подбирать значения

for k in eps_:

    fig, axes = plt.subplots(2,5 , figsize=(15,7))

    for ax, j in zip(fig.axes ,min_samples_):

        DB = DBSCAN(eps=k, min_samples=j, n_jobs=-1 )

        DB_clusters = DB.fit_predict(data_x)

        ax.scatter(data_x.x, data_x.y, c=DB_clusters, cmap='autumn', s=60)'''
choice = np.random.choice(data_prob.index, size=50000, replace=False).tolist()

data_50 = data_prob[data_prob.index.isin(choice)]
#               В данном примере со стандартицацией выходит хуже

# from sklearn.preprocessing import StandardScaler

# scal = StandardScaler()

# data_50_sc = scal.fit_transform(data_50)
db_classic = DBSCAN(eps=0.02, min_samples=500, n_jobs=-1 )
db_clusters = db_classic.fit_predict(data_50)
rcParams['figure.figsize'] = (7,7)

for i,color in zip(range(10),{'blue','red','green','black','orange','yellow','brown','orchid','lime'}):

    x_i = data_50.x[db_clusters==i]

    y_i = data_50.y[db_clusters==i]

    plt.plot(x_i, y_i, 'ro', c=color)

    plt.plot(x_i, y_i, 'ro', c=color)

    

    x_0 = data_50.x[db_clusters==-1]

    y_0 = data_50.y[db_clusters==-1]

    plt.plot(x_0, y_0, 'ro', c='grey')

    
df = pd.read_excel('../input/geo_comment.xlsx')
df_n = df[df['comment_class']==-1]

df_n = df.drop(['x', 'y', 'comment_class'], axis=1)
import re

import string

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import nltk

# from pymorphy2 import MorphAnalyzer - в kaggle не установлен
df_n.head(3)
# Обработоем полученное описание:

#                             - уберем все знаки препинания

#                             - уберем стоп-слова

#                             - нормализуем и сделаем стемминг
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("russian")

PortSt = PorterStemmer()

nltk.download('stopwords')

nltk.download('punkt')
chrs_to_delete = string.punctuation + u'»' + u'«' + u'—' + u'“' + u'„' + u'•' + u'#'

translation_table = {ord(c): None for c in chrs_to_delete if c != u'-'}

# units = MorphAnalyzer.DEFAULT_UNITS

# morph = MorphAnalyzer(result_type=None, units=units)

PortSt = PorterStemmer()

stopw = set(

    [w for w in stopwords.words(['russian', 'english'])]

    + [u'это', u'году', u'года', u'также', u'етот',

       u'которые', u'который', u'которая', u'поэтому',

       u'весь', u'свой', u'мочь', u'eтот', u'например',

       u'какой-то', u'кто-то', u'самый', u'очень', u'несколько',

       u'источник', u'стать', u'время', u'пока', u'однако',

       u'около', u'немного', u'кроме', u'гораздо', u'каждый',

       u'первый', u'вполне', u'из-за', u'из-под',

       u'второй', u'нужно', u'нужный', u'просто', u'большой',

       u'хороший', u'хотеть', u'начать', u'должный', u'новый', u'день',

       u'метр', u'получить', u'далее', u'именно', u'апрель',

       u'сообщать', u'разный', u'говорить', u'делать',

       u'появиться', u'2016',

       u'2015', u'получить', u'иметь', u'составить', u'дать', u'читать',

       u'ничто', u'достаточно', u'использовать',

       u'принять', u'практически',

       u'находиться', u'месяц', u'достаточно', u'что-то', u'часто',

       u'хотеть', u'начаться', u'делать', u'событие', u'составлять',

       u'остаться', u'заявить', u'сделать', u'дело',

       u'примерно', u'попасть', u'хотя', u'лишь', u'первое',

       u'больший', u'решить', u'число', u'идти', u'давать', u'вопрос',

       u'сегодня', u'часть', u'высокий', u'главный', u'случай', u'место',

       u'конец', u'работать', u'работа', u'слово', u'важный', u'сказать']

)
# clean_dict = []

clean_dict = {}

counter = 0



for number, doc in zip(range(50000), df_n.comment):

    body = doc

    body = re.sub('\[.*?\]','', body)

    if body != '':

        body_clean = body.translate(translation_table).lower().strip()

        words = word_tokenize(body_clean)

        tokens = []

        # делаем стемминг и нормализацию

        for word in words:

            if re.match('^[a-z0-9-]+$', word) is not None:

                tokens.append(PortSt.stem(word))

            elif word.count('-') > 1:

                tokens.append(word)

            else:

                filtered_tokens = []

                filtered_tokens.append(word)

                for t in filtered_tokens:

                    stems = stemmer.stem(t)

                    tokens.append(stems)

        # убираем стоп слова

        tokens = filter(

            lambda token: token not in stopw, sorted(set(tokens))

        )



        # убираем слова маленькой длины

        tokens = filter(lambda token: len(token) > 3, tokens)

    else:

        tokens = []

    counter += 1

    if counter % 500 == 0:

        print("{0} docs processed".format(counter))

    if counter == 50000: # для простоты расчета возьмем выборку из 50 000

        break

    clean_dict[number] = tokens

#     clean_dict.append(tokens)



clean_dict = {key: list(val) for key, val in clean_dict.items()}
from gensim.corpora import TextCorpus

from gensim.models.ldamodel import LdaModel



class ListTextCorpus(TextCorpus):



    def get_texts(self):

        for doc in self.input:

            yield doc

                

mycorp = ListTextCorpus(input=clean_dict.values())

justlda = LdaModel(

    corpus=mycorp, num_topics=4, passes=30

)
# Получаем описание 4 тематик

print('LdaModel performance')

for i in range(4):

    terms = justlda.get_topic_terms(i)

    print(i, ' '.join(map(lambda x: mycorp.dictionary.get(x[0]), terms)))