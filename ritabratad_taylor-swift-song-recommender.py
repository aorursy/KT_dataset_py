import pandas as pd

from pandas import DataFrame

import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer #for TFIDF 

def clean_lyric(lyric): #first layer of cleaning(remo)

    if '[' in lyric:

        return ''

    else:

        return lyric
df = pd.read_csv('/kaggle/input/taylorswiftlyrics/final_taylor_swift_lyrics.tsv', sep= '\t')

print(df.loc[1])
df=df.drop(['release_date','index','line_number','album'],axis=1)
org=df #to store a copy
print(df)
spec_chars = ["!",'"',"#","%","&","(",")",

              "*","+",",","-",".","/",":",";","<",

              "=",">","?","@","[","\\","]","^","_",

              "`","{","|","}","~","–"]

for char in spec_chars:

    df['lyric'] = df['lyric'].str.replace(char, ' ')

df['lyric'] = df['lyric'].str.replace("'", '')
base=[]

s_name=df.loc[0][0]

st=""

for i, j in df.iterrows(): 

    if(j[0]==s_name):

        st+=" "+j[1] #adding lines to the string of a song

    else:

        base.append([s_name,st]) #adding the song to array

        s_name=j[0]

        st=""

        

        

        

        
len(base) #No of songs
frame= DataFrame (base,columns=['name','lyrics']) #Converting back to data frame
frame.head()
# Initialize tfidf vectorizer

tfidf = TfidfVectorizer(analyzer='word', stop_words='english')



# Fit and transform 

tfidf_matrix = tfidf.fit_transform(frame['lyrics'])
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarities = cosine_similarity(tfidf_matrix)

similarities = {}

for i in range(len(cosine_similarities)):

    # Now we'll sort each element in cosine_similarities and get the indexes of the songs. 

    similar_indices = cosine_similarities[i].argsort()[:-50:-1] 

    # After that, we'll store in similarities each name of the 50 most similar songs.

    # Except the first one that is the same song.

    similarities[frame['name'].iloc[i]] = [(cosine_similarities[i][x], frame['name'][x]) for x in similar_indices][1:]

class ContentBasedRecommender:

    def __init__(self, matrix):

        self.matrix_similar = matrix



    def _print_message(self, song, recom_song):

        rec_items = len(recom_song)

        

        print(f'The {rec_items} recommended songs for {song} are:')

        for i in range(rec_items):

            print(f"Number {i+1}:")

            print(f"{recom_song[i][1]} with {round(recom_song[i][0], 3)} similarity score") 

            print("--------------------")

        

    def recommend(self, recommendation):

        # Get song to find recommendations for

        song = recommendation['song']

        # Get number of songs to recommend

        number_songs = recommendation['number_songs']

        # Get the number of songs most similars from matrix similarities

        recom_song = self.matrix_similar[song][:number_songs]

        # print each item

        self._print_message(song=song, recom_song=recom_song)



# Instantiate class

recommedations = ContentBasedRecommender(similarities)



#Commenting this cell because it requires dynamic input and cant show output here



#inp= input("Enter the song name:")

#print("")

#print("Which one of these songs you want :")

#for i,j in frame.iterrows():

    #if (j[0].find(inp)>=0):

       # print(i,j[0])

        

#print("")

#num=input("enter the song number ")

#num=int(num)

#num_son= input("enter the number of recommendations needed ")

#num_son= int(num_son)

    

    
num=0 #ID for Lover

num_son = 3 #top 3 songs
# Create dict to pass

recommendation = {

    "song": frame['name'].iloc[num],

    "number_songs":  num_son

}



# Recommend

recommedations.recommend(recommendation)
from gensim.test.utils import common_texts, get_tmpfile

from gensim.models import Word2Vec

path = get_tmpfile("word2vec.model")

model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)

model.save("word2vec.model")
import numpy as np

class DocSim:

    def __init__(self, w2v_model, stopwords=None):

        self.w2v_model = w2v_model

        self.stopwords = stopwords if stopwords is not None else []



    def vectorize(self, doc: str) -> np.ndarray:

        """

        Identify the vector values for each word in the given document

        :param doc:

        :return:

        """

        doc = doc.lower()

        words = [w for w in doc.split(" ") if w not in self.stopwords]

        word_vecs = []

        for word in words:

            try:

                vec = self.w2v_model[word]

                word_vecs.append(vec)

            except KeyError:

                # Ignore, if the word doesn't exist in the vocabulary

                pass



        # Assuming that document vector is the mean of all the word vectors

        # PS: There are other & better ways to do it.

        vector = np.mean(word_vecs, axis=0)

        return vector



    def _cosine_sim(self, vecA, vecB):

        """Find the cosine similarity distance between two vectors."""

        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))

        if np.isnan(np.sum(csim)):

            return 0

        return csim



    def calculate_similarity(self, source_doc, target_docs=None, threshold=0):

        """Calculates & returns similarity scores between given source document & all

        the target documents."""

        if not target_docs:

            return []



        if isinstance(target_docs, type("abc")):

            target_docs = [target_docs]



        source_vec = self.vectorize(source_doc)

        results = []

        for doc in target_docs:

            target_vec = self.vectorize(doc)

            sim_score = self._cosine_sim(source_vec, target_vec)

            if sim_score > threshold:

                results.append({"score": sim_score, "doc": doc})

            # Sort results by score in desc order

            results.sort(key=lambda k: k["score"], reverse=True)



        return results
ds = DocSim(model)
testY= frame.name[1][0]

testX= frame.name[1][1]



trainX=[]

trainY=[]



for i,j in frame.iterrows():

    if (i!=1):

        trainX.append(j[1])

        trainY.append(j[0])
sim_scores = ds.calculate_similarity(testX, trainX)
print(sim_scores)
trainX[0]
type(testX)
source_doc = "how to delete an invoice"

target_docs = ['delete a invoice', 'how do i remove an invoice', "purge an invoice"]



sim_scores = ds.calculate_similarity(source_doc, target_docs)

print(sim_scores)

common_texts[2]