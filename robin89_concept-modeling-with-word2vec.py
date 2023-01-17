%matplotlib inline

import numpy as np

import pandas as pd 

import nltk

import string

import operator

from nltk.stem import WordNetLemmatizer

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering

from sklearn.manifold import TSNE

from gensim.models import word2vec

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from nltk.corpus import wordnet as wn

from nltk.probability import FreqDist

import re
class RAKE_tagged():

    '''

    This class applies the RAKE (Rapid Automatic Keyword Extraction) algorithm after filtering and stemming the possible candidates.

    it can be used for any language by passing the taggers, tokenizers and stemmers to the initialisation

    '''

    def __init__(self, no_of_keywords, stopwords='auto', pos =['N'], tokenizer='auto', tagging='auto', stemming='auto'):

        self.pos = pos

        self.nkey = no_of_keywords

        self.stopwords = stopwords

        self.tokenizer = tokenizer

        self.tagging = tagging

        self.lemma = stemming

        self.finaltext = []





        pass

    

    def isPunct(self, word):

        return len(word) == 1 and word in string.punctuation



    def isNumeric(self, word):

        try:

            float(word) if '.' in word else int(word)

            return True

        except ValueError:

            return False

            

    def manage_params(self):

        if type(self.stopwords) == list:

            pass

        elif self.stopwords == "auto":

            self.stopwords = set(nltk.corpus.stopwords.words())

        else:

            self.stopwords = []

            

        if self.tokenizer == 'auto':

            self.tokenizer = nltk.word_tokenize

        else:

            pass

        

        if self.tagging == 'auto':

            self.tagging = nltk.pos_tag

        else:

            pass

        

        if self.lemma == 'auto':

            lemmatizer = WordNetLemmatizer()

            self.lemma = lemmatizer.lemmatize

        else:

            pass



    def candidate_keywords(self, document_sents):

        '''

        Finding the possible candidates for keywords, by removing stopwords and filtering by POS tag

        '''

        phrase_list = []

        all_words = []



        for sentence in document_sents:

            words = map(lambda x: "#" if x in self.stopwords else x, self.tagging(self.tokenizer(sentence.lower())))

            

            # create lemmatized corpus for word2vec (lists-of-words in lists-of-sentences in list-of-corpus)

            sent = []

            for w in words:

                all_words.append(w)

                if w[0] != "#" or self.isPunct(w[0]) == False:

                    try:

                        sent.append(self.lemma(w[0], w[1][0].lower()))

                    except:

                        sent.append(self.lemma(w[0]))

            self.finaltext.append(sent)

        

            

        

        # back to the RAKE Algorithm 

        phrase = []

        candidates =[]



        for (word, tag) in all_words:

            if word == "#" or self.isPunct(word):

                if len(phrase) > 0:

                    phrase_list.append(phrase)

                    phrase = []

            else:

                if self.pos != None:

                    phrase.append(word)

                    for t in self.pos:

                        if tag.startswith(t):

                            candidates.append(self.lemma(word, t[0].lower()))

                                #print(self.lemma(word, t.lower()))

                        else:

                            pass

                else:

                    phrase.append(word)

                    candidates.append(word)

        return phrase_list, candidates

    

    def calculate_word_scores(self, phrase_list, candidates):

        word_freq = nltk.FreqDist()

        word_degree = nltk.FreqDist()

        for phrase in phrase_list:

          degree = len(list(filter(lambda x: not self.isNumeric(x), phrase))) - 1

          for word in phrase:

            word_freq[word] += 1

            word_degree[word] += degree

        for word in word_freq.keys():

          word_degree[word] = word_degree[word] + word_freq[word]

        word_scores = {}

        for word in word_freq.keys():

            if word in candidates:

                word_scores[word] = word_degree[word] / word_freq[word]

        return word_scores

    

    def calculate_phrase_scores(self, phrase_list, word_scores, candidates):

        phrase_scores = {}

        for phrase in phrase_list:

          phrase_score = 0

          for word in phrase:

              if word in candidates:

                phrase_score += word_scores[word]

          phrase_scores[" ".join(phrase)] = phrase_score

        return phrase_scores

    

    def fit(self):

        #might be useful for piping

        pass



    def transform(self, corpus, output_type="w"):

        keyword_results = []

        self.manage_params()

        for document in corpus:

            sentences = nltk.sent_tokenize(document)

            phrase_list, candidate_list = self.candidate_keywords(sentences)

            word_scores = self.calculate_word_scores(phrase_list, candidate_list)

            phrase_scores = self.calculate_phrase_scores(phrase_list, word_scores, candidate_list)

            if output_type == "s":

                sorted_scores = sorted(phrase_scores.items(),key=operator.itemgetter(1), reverse=True)

            else:

                sorted_scores = sorted(word_scores.items(),key=operator.itemgetter(1), reverse=True)           

            keyword_results.append([k[0] for k in sorted_scores[0:self.nkey]])

        return keyword_results
class Computing_Vectors():

    

    def __init__(self):



        pass

    

    def fit(self, corpus):

        self.model = word2vec.Word2Vec(corpus, size=100, window=5, min_count=1, workers=4)

        

    

    def transform(self, keywords):

        self.arrays = []

        

        self.model = self.model.wv

        found = 0

        notfound = 0

        for kwl in keywords:

            for keyw in kwl:

                try:

                    self.arrays.append((self.model[keyw], keyw))

                    found += 1

                except:

                    notfound += 1

                    pass



        return self.arrays

    

    def visualize(self):

        x_arr =np.array([i[0] for i in self.arrays])

        tsne = TSNE(n_components=2)

        X_tsne = tsne.fit_transform(x_arr)

        

        plt.figure(figsize=(200, 200), dpi=100)

        max_x = np.amax(X_tsne, axis=0)[0]

        max_y = np.amax(X_tsne, axis=0)[1]

        plt.xlim((-max_x,max_x))

        plt.ylim((-max_y,max_y))



        

        for row_id in range(0, len(self.arrays)):

            target_word = self.arrays[row_id][1]

            x = X_tsne[row_id, 0]

            y = X_tsne[row_id, 1]

            plt.annotate(target_word, (x,y))



        plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

        plt.savefig("word2vec.png")

        plt.show()

 
class Clustering():

    def __init__(self,  w2v, model):

        self.model = model

        self.w2v = w2v

        self.w2vecarrays = np.array([i[0] for i in w2v])

        pass

    

    def fit_transform(self, number_of_clusters):

        cluster = KMeans(n_clusters=number_of_clusters, random_state=0).fit(self.w2vecarrays)

        #cluster = DBSCAN(eps=0.01, min_samples=3).fit(self.w2vecarrays)

        #cluster = SpectralClustering(n_clusters=number_of_clusters, eigen_solver='arpack',affinity="nearest_neighbors")

        #cluster = AgglomerativeClustering(n_clusters=number_of_clusters, linkage='ward')

        self.labels = cluster.labels_

        counts = np.bincount(self.labels[self.labels>=0])

               

        self.concepts = {}

        

        for row_id in range(0, len(self.labels)):

            word = self.w2v[row_id][1]

            label = self.labels[row_id]

            if label in self.concepts:

                self.concepts[label].append(word)

            else:

                self.concepts[label] = [word]

                

        return self.concepts

    

    def visualize(self):

        tsne = TSNE(n_components=2)

        X_tsne = tsne.fit_transform(self.w2vecarrays)

        

        for concept in sorted(self.concepts):

            try:

                txt = " ".join(self.concepts[concept])

                wordcloud = WordCloud(background_color="white",max_font_size=40, relative_scaling=.5).generate(txt)

                plt.figure()

                plt.imshow(wordcloud)       

                plt.title(concept)

                plt.axis("off")

                plt.show()

            except ValueError:

                print(self.concepts[concept])


from nltk.corpus import inaugural



sample = [{'ID': fileid, 'Text': inaugural.raw(fileid)} for fileid in inaugural.fileids()]

df = pd.DataFrame(sample)



df= pd.read_csv('../input/stage2_test_text.csv', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])

#df.head()



df_txt = df['Text'][:2000]
myRE1 = RAKE_tagged(10, stopwords='auto', pos=["N", "VBP", "R"])

myRE = RAKE_tagged(10, stopwords='auto', pos=["N"])

summary = myRE1.transform(df_txt, output_type="s")

summaries =["; ".join(s) for s in summary]

keywords = myRE.transform(summaries, output_type="w")
CV = Computing_Vectors()

CV.fit(myRE1.finaltext)

arr = CV.transform(keywords)

CV.visualize()
CL = Clustering(arr, CV.model)

concepts = CL.fit_transform(25)

CL.visualize()