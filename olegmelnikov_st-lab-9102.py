# load popular packages

import numpy as np, pandas as pd, os, matplotlib.pyplot as plt,re, regex, string, pprint

import unicodedata, sys, pickle

from pprint import pprint

from time import time

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import Word2Vec, KeyedVectors

import seaborn as sns

# import boto3, os

# s3 = boto3.resource('s3')

pd.set_option('display.max_rows', 40)

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.width', 10000)

np.set_printoptions(linewidth=1000, precision=4, edgeitems=20, suppress=True)

tqdm.pandas(mininterval=3)



class Timer():

    from time import time

    def __init__(self):

        self.is_localPC = os.path.isdir('C:/Users/Oleg Melnikov/Downloads/')

        # self.is_interactive = 'SHLVL' not in os.environ

        self.set()

    def set(self): self.t0 = time()

    def show(self, msg='', reset=True):

        print(f'{time() - self.t0:.1f} {msg}')

        if reset: self.set()



def display_local_variables(msg='', topN=5, lcl=locals().items()):

    def sizeof_fmt(num, suffix='B'): #https://stackoverflow.com/a/1094933/1870254

        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

            if abs(num) < 1024.0: return "%3.1f%s%s" % (num, unit, suffix)

            num /= 1024.0

        return "%.1f%s%s" % (num, 'Yi', suffix)



    if msg!='': print(msg)

    for var, sz in sorted(((k, sys.getsizeof(v)) for k,v in lcl), key= lambda x: -x[1])[:topN]:

        print("{:>30}: {:>8}".format(var,sizeof_fmt(sz)))

# !pip install paramiko

# print(os.listdir("../input"))
# class for abstracting away loading and basic operations on word embeddings (from different providers)

class Emb():

    def __init__(self, emb_dir=''):

        self.emb_dir = emb_dir

        self.X = None

        self.KeyedVectors = False

        self._isLower = None



    def _readEmb_txt2dict(self, filename='glove.6B.50d.txt'):

        # read embeddings from a text file. Store as dictionary

        from tqdm import tqdm

        assert filename[-4:]=='.txt'

        def get_coefs(word, *arr):

            return word, np.asarray(arr, dtype='float32')  # word followed by word vector

        with open(self.emb_dir+filename, 'r', encoding="utf8") as f:

            self.X = dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f, mininterval=1))

        self.KeyedVectors = False

        return self



    def _readEmb_pickle2dict(self, filename='glove.840B.300d.pkl'):

        # read embeddings from a pkl file (fast!). Store as dictionary

        assert filename[-4:] == '.pkl'

        with open(self.emb_dir+filename, 'rb') as f:

            self.X = pickle.load(f)

        self.KeyedVectors = False

        return self



    def _readEmb_bin2KeyedVectors(self, filename='GoogleNews-vectors-negative300.bin'):

        # read embeddings from a binary file. Store as gensim.models.keyedvectors.Word2VecKeyedVectors

        assert filename[-4:]=='.bin'

        from gensim.models import KeyedVectors



        self.X = KeyedVectors.load_word2vec_format(self.emb_dir + filename, binary=True)



        self.KeyedVectors = True

        return self.X



    def readEmb(self, filename):

        # Different type of word embedding file requires different loading procedure

        if filename[-4:]=='.txt':

            self._readEmb_txt2dict(filename)

        elif filename[-4:]=='.pkl':

            self._readEmb_pickle2dict(filename)

        elif filename[-4:]=='.bin':  # binary file from Gensim word2vec (up to 3 million words)

            self._readEmb_bin2KeyedVectors(filename)

        else:

            print('unknown file type. Must be .txt, .pkl, or .bin')

            return None



        self._lowerCount = None  # reset for the freshly loaded embedding

        # self.info()

        return self



    def isLower(self):

        if self._isLower is None:  # this check takes a few sec, so cache the results

            words = set(self.vocab())

            words_lower = set(w.lower() for w in words)

            self._lowerCount = len(words_lower)

            self._isLower = len((words - words_lower)) == 0  # whether embedding uses strictly lower case

        return self._isLower



    def lowerCount(self):  # returns count of strictly lower-cased words

        self.isLower()  # also counts lower case words

        return self._lowerCount



    def valueType(self):

        if self.KeyedVectors:

            print(type(self.X))

            vec = self.X[next(iter(self.X.vocab))]

        else:

            vec = next(iter(self.X.values()))  # get any one vector

        return type(vec[0])



    def shape(self):

        if self.KeyedVectors:

            return (len(self.X.index2word), self.X.vector_size)

        else:

            vec = next(iter(self.X.values()))  # get any one vector

            return (len(self.X), len(vec))



    def vocab(self, numWords=-1, castAs=set, seed=None):

        words = self.X.vocab.keys() if self.KeyedVectors else self.X.keys()

        if numWords >=0:

            np.random.seed(seed)

            words = np.random.choice(list(words), size=numWords)

        return np.array(list(words)) if castAs==np.array else castAs(words)



    def info(self, numWords=10):

        from pprint import pprint

        if self.X is None:

            print('embedding is not loaded')

        else:

            caseMsg = "lower" if self.isLower() else "varying"

            n,m = self.shape()

            print(f'{n} words, {self.lowerCount()} in lower case, {m} embedding size, {self.valueType()} type')

            pprint(f'{numWords} random word vectors:')

            pprint({k: self.X[k] for k in np.random.choice(list(self.vocab()), size=numWords)})



    def to_frame(self, wordsAsRows=True):

        import pandas as pd

        if not self.KeyedVectors:

            return pd.DataFrame.from_dict(self.X, orient=('index' if wordsAsRows else 'columns'))



    def findWords(self, lstToFind, ignoreCase=True, wholeWord=True, isRegEx=False, asDF=True):

        vocab = vocab_orig = self.vocab(castAs=np.array)



        if ignoreCase:

            lstToFind = np.char.lower(lstToFind)

            vocab = np.array([w.lower() for w in vocab_orig])



        if wholeWord and not isRegEx:

            found_lists = [[vocab == w] for w in lstToFind]

            # found = vocab_orig[ np.any([[vocab == w] for w in lstToFind], 0)[0]]

        elif not isRegEx:

            found_lists = [[np.core.defchararray.find(vocab, w) != -1] for w in lstToFind]

            # found = vocab_orig[ np.any( [[np.core.defchararray.find(vocab, w) != -1] for w in lstToFind],0)[0]]

        else:

            import re

            flags = re.IGNORECASE if ignoreCase else 0

            found_lists = [[bool(re.match(w, x, flags=flags)) for x in vocab] for w in lstToFind]

            # found = vocab_orig[np.any([[re.match(w, x, flags=flags) is not None for x in vocab] for w in lstToFind], 0)]

        found = vocab_orig[ np.any( found_lists,0).flatten()]



        if len(found)>0:

            emb = np.concatenate([self.X[w][:, None] for w in found], 1, out=None).T

        else:

            emb = np.array([])

        if asDF:

            return pd.DataFrame(emb, found)

        return found, emb



    def cosine_similarity(self, lstToFind, ignoreCase=True, wholeWord=True, isRegEx=False, k2plot=100, k2annot=25, clustermap=False):

        '''



        :param lstToFind:

        :param ignoreCase:

        :param wholeWord:

        :param k2plot: add a plot if fewer than k words are found (to avoid overplotting

        :param clustermap: whether the heatmap should be orderd by words or clustered by similarity values

        :param k2annot: add annotations if fewer than k words are found (to avoid overplotting

        :return:

        '''

        words, mtx = self.findWords(lstToFind, ignoreCase=ignoreCase, wholeWord=wholeWord, isRegEx=isRegEx, asDF=False)



        if len(mtx)==0:

            print(f'No match found.')

            return None

        dfSim = pd.DataFrame(cosine_similarity(mtx), index=words, columns=words)



        ann = len(words) < k2annot

        plot = len(words)<k2plot

        col="YlGnBu"



        if plot:

            import seaborn as sns

            if clustermap:

                sns.clustermap(dfSim, cmap=col, xticklabels=words, yticklabels=words, annot=ann).fig.suptitle('Cosine similarities')

            else:

                import matplotlib.pyplot as plt

                # cols_lower = [w.lower() for w in dfSim.columns]

                # cols_sorted = [x for _, x in sorted(zip(dfSim.columns, cols_lower))]

                # dfSim['order']=cols_lower

                # dfSim.sort_values('order', inplace=True)

                # dfSim.drop('order',1,inplace=True)

                # dfSim = dfSim.reindex(cols_lower, axis=1)

                # dfSim.sort_values(by=cols_lower, ascending=False)

                # dfSim.sort_index(inplace=True)

                ax = plt.axes()

                sns.heatmap(dfSim, cmap=col, xticklabels=words, yticklabels=words, annot=ann)

                ax.set_title('Cosine similarities')

                plt.show()

        return dfSim



    def rankSynonyms(self, w, synLst):

        if not w in self.X:

            print(f'{w} is not found in word embedding')

        synLst = [w for w in synLst if w in self.X]

        synVecs = [self.X[v] for v in synLst]



        cosSim = lambda x,y: (x @ y) / np.linalg.norm(x)/ np.linalg.norm(y)

        cosSim1 = lambda y: cosSim(self.X[w],y)

        return  pd.DataFrame(map(cosSim1, synVecs), index=synLst, columns=[w]).sort_values(w, ascending=False)
# Lists of common words (tokens).

sectors = ['Energy','Telecommunication', 'Industrial','Industrials',

           'Financial','Financial_Services', 'Consumer_Staples',

            'Utility', 'Utilities', 'Materials', 'Consumer_Discretionary',

           'Telecommunication Services', 'Health_Care', 'Information_Technology']



continents=('North_America','Asia','Europe','Australia','South_America','Africa','SOUTH_AMERICA','NORTH_AMERICA')



countries=('United_Kingdom','United_States','Japan','China','Germany','Switzerland','Spain','Belarus',

           'Philippines','United_Arab_Emirates','Ireland','France','India','Thailand','Ukraine',

           'Taiwan','Netherlands','Russia','Belgium','Australia','Canada','Bahrain','Hong_Kong',

           'Turkey','Saudi_Arabia','Mexico','Greece','Malaysia','South_Korea','Chile','Jordan',

           'Luxembourg','Bermuda','Sweden','Italy','Morocco','Portugal','Brazil','Colombia',

           'Venezuela','Lebanon','Indonesia','Israel','Oman','Peru','South_Africa','Singapore',

           'Denmark','Argentina','Czech_Republic','Vietnam','Qatar','Egypt','Nigeria','Norway',

           'Austria','Finland','Poland','Pakistan','Mongolia','Kuwait','Hungary','Puerto_Rico',

           'Kazakhstan','Togo','Mauritius','Cayman_Islands','Panama','Channel_Island','Liberia','New_Zealand')



industries=('Aerospace_Defense','AEROSPACE_DEFENSE','catalogers_retailers','cataloger_retailer',

            'Consumer_Finances','CONSUMER_FINANCE','CONSUMER_FINANCE_INDUSTRY','THE_CONSUMER_FINANCE',

            'Conglomerates','Regional_Banks','Financial_Services',

            'Thrifts_&_Mortgage_Finance','Pharmaceuticals',

            'Other_Transportation','Computer_Services','Construction_Services','Hotels_&_Motels',

            'Business_&_Personal_Services','Apparel/Accessories','Software_&_Programming',

            'Specialty_Stores','Telecommunications_services','Semiconductors','Diversified_Insurance',

            'Discount_Stores','Electric_Utilities','Managed_Health_Care','Investment_Services',

            'Life_&_Health_Insurance','Real_Estate','Electronics','Specialized_Chemicals','Airline',

            'Auto_&_Truck_Parts','Food_Processing','Diversified_Chemicals','Aluminum','Biotechs',

            'Property_&_Casualty_Insurance','Diversified_Metals_&_Mining','Tobacco',

            'Internet_&_Catalog_Retail','Containers_&_Packaging',#'Consumer_Financial_Services',

            'Diversified_Utilities','Electrical_Equipment','Oil_&_Gas_Operations','Iron_&_Steel',

            'Beverages','Construction_Materials','Major_Banks','Insurance_Brokers','Computer_Hardware',

            'Other_Industrial_Equipment','Rental_&_Leasing','Communications_Equipment',

            'Aerospace_&_Defense','Recreational_Products','Oil_Services_&_Equipment',

            'Medical_Equipment_&_Supplies','Household/Personal_Care','Natural_Gas_Utilities',

            'Apparel/Footwear_Retail','Computer_&_Electronics_Retail','Auto_&_Truck_Manufacturers',

            'Broadcasting_&_Cable','Railroads','Business_Products_&_Supplies','Food_Retail',

            'Heavy_Equipment','Healthcare_Services','Trading_Companies','Restaurants','Casinos_&_Gaming',

            'Printing_&_Publishing','Advertising','Air_Courier','Trucking','Department_Stores',

            'Household_Appliances','Home_Improvement_Retail','Consumer_Electronics','Security_Systems',

            'Paper_&_Paper_Products','Furniture_&_Fixtures','Computer_Storage_Devices',

            'Environmental_&_Waste','Drug_Retail','Precision_Healthcare_Equipment')
emb50=Emb("../input/nlpword2vecembeddingspretrained/").readEmb('glove.6B.50d.txt')
emb50.cosine_similarity(['python','programming'], ignoreCase=True, wholeWord=False, isRegEx=False, k2annot=5)
# loads a 300-dimensional Word2Vec embedding from a text file (from Google). Takes ~1-2 minute

emb300wv=Emb("../input/nlpword2vecembeddingspretrained/").readEmb('GoogleNews-vectors-negative300.bin')
emb300wv.cosine_similarity(['citi[_].*'], ignoreCase=True, wholeWord=False, isRegEx=True)
# emb300wv.rankSynonyms('Chase', sectors).plot.barh(grid=True)

emb300wv.rankSynonyms('Microsoft', ['countries']).plot.barh(grid=True)

# emb300wv.rankSynonyms('Google', countries).plot.barh(grid=True)

# emb300wv.rankSynonyms('JPMOrgan_Chase', industries).plot.barh(grid=True)

# emb300wv.rankSynonyms('Aboitiz', industries).plot.barh(grid=True)

# emb300wv.rankSynonyms('Aboitiz', industries).plot.barh(grid=True)

# emb300wv.rankSynonyms('Agile_Property', industries).plot.barh(grid=True)

# emb300wv.rankSynonyms('Alexion', industries).plot.barh(grid=True)

# emb300wv.rankSynonyms('Alexion_Pharmaceuticals', industries).plot.barh(grid=True)

# emb300wv.rankSynonyms('AmerisourceBergen', industries).plot.barh(grid=True)

# emb300wv.rankSynonyms('Chugoku', industries).plot.barh(grid=True) 

# emb300wv.rankSynonyms('Chugoku', countries).plot.barh(grid=True) 

# emb300wv.rankSynonyms('Unibanco', countries).plot.barh(grid=True) 

# emb300wv.rankSynonyms('Jazz_Pharmaceuticals', countries).plot.barh(grid=True) 

# emb300wv.rankSynonyms('Jeronimo_Martins', countries).plot.barh(grid=True)



# emb300wv.rankSynonyms('JetBlue_Airways', countries).plot.barh(grid=True) 

# emb300wv.rankSynonyms('JetBlue', countries).plot.barh(grid=True) 

# emb300wv.rankSynonyms('JetBlue_Airways_Corp.', countries).plot.barh(grid=True) 

# emb300wv.rankSynonyms('Jetblue', countries).plot.barh(grid=True) 

# emb300wv.rankSynonyms('Jetblue_Airways_Corp', countries).plot.barh(grid=True) 

# emb300wv.rankSynonyms('JetBlue_Airways_Nasdaq_JBLU', countries).plot.barh(grid=True) 

# emb300wv.rankSynonyms('www.jetblue.com', countries).plot.barh(grid=True)



# emb300wv.rankSynonyms('Putin', countries).plot.barh(grid=True) 

# emb300wv.rankSynonyms('Putin', industries).plot.barh(grid=True) 
sectors