import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import re
import umap
import pickle
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
plt.style.use('ggplot')
import pickle
from glove import Corpus, Glove

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
'''Import and download if not installed already.
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')'''
# Mutes all the tensorflow warnings. Don't run tensorflow_shutup if you want warnings.
def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    
    Source:
    https://stackoverflow.com/a/54950981
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass

tensorflow_shutup()
class Data_Processor:
    def __init__(self, data_root_path, csv):
        '''
        
        PARAMETRS
        ---------
        data_root_path: (str) The root path of data
        csv: (str) The path to the paper corpus in csv format.
        '''
        self.data_root_path = data_root_path
        self.csv = data_root_path + "/" + csv
        
    def file_reader(self, file_path):
        '''
        Reads and converts json files into dataframe while removing empty 
        Code from: https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool
        
        PARAMETER
        ---------
        file_path: (str)
        '''
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            try:
                # Abstract
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
                # Body text
                for entry in content['body_text']:
                    self.body_text.append(entry['text'])
                self.abstract = '\n'.join(self.abstract)
                self.body_text = '\n'.join(self.body_text)
                # Extend Here
                #
                #
                self.notFound = False

                # Drop if the abtract text or body text is empty!
                if(len(self.abstract)==0 or len(self.body_text)==0): 
                    self.notFound = True
            except:
                self.notFound = True
                
    
    def __create_df_covid__(self, save_to_path):
        '''
        Converts the raw file into a useable dataframe and saves to csv
        
        PARAMETERS
        ----------
        save_to_path: (str)
        '''
        # Used code from https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool
        all_json = glob.glob(self.data_root_path + '/**/*.json', recursive=True)
        
        dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}
        for idx, entry in enumerate(all_json):
            if idx % (len(all_json) // 10) == 0:
                print('Processing index: ' + str(idx) + ' of '  + str(len(all_json)))
            content = self.file_reader(entry)
            if(not self.notFound):
                dict_['paper_id'].append(self.paper_id)
                dict_['abstract'].append(self.abstract)
                dict_['body_text'].append(self.body_text)
        # Our own:
        self._df_covid_ = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])
        self._df_covid_['abstract_word_count'] = self._df_covid_['abstract'].apply(lambda x: len(x.strip().split()))
        self._df_covid_['body_word_count'] = self._df_covid_['body_text'].apply(lambda x: len(x.strip().split()))
        self._df_covid_.drop_duplicates(['abstract', 'body_text'], inplace=True)
        self._df_covid_.describe(include='all')
        self._df_covid_.to_csv(save_to_path)
    
    def load_df_covid(self, covid_path):
        '''
        Loads the covid dataframe from csv file
        
        PARAMETERS
        ----------
        covid_path: (str)
        '''
        self._df_covid_ = pd.read_csv(covid_path)
    
    def __split_data__(self):
        '''
        Split the processed covid dataframe into text format and train/test sets.
        '''
        
        # Split into train/test.
        seed = 69420
        covid_data_copy = self._df_covid_.copy()
        train_set = covid_data_copy.sample(frac=0.8, random_state=seed)
        test_set = covid_data_copy.drop(train_set.index)
        
        ### Get texts
        abstract_text_train = train_set['abstract']
        abstract_text_test = test_set['abstract']
        body_text_train = train_set['body_text']
        body_text_test = test_set['body_text']
        
        # Remove line break and lower. TODO Consider spacing symbols, replacing numbers with '#' and removing references.
        abstract_text_train = abstract_text_train.apply(lambda x: re.sub('\n','',x)).apply(lambda x: x.lower())
        abstract_text_test = abstract_text_test.apply(lambda x: re.sub('\n','',x)).apply(lambda x: x.lower())
        body_text_train = body_text_train.apply(lambda x: re.sub('\n','',x)).apply(lambda x: x.lower())
        body_text_test = body_text_test.apply(lambda x: re.sub('\n','',x)).apply(lambda x: x.lower())
        
        # Put each abstract and body-text on a line
        label_data_train = "\n".join(abstract_text_train.values)
        label_data_test = "\n".join(abstract_text_test.values)
        training_data_train = "\n".join(body_text_train.values)
        training_data_test = "\n".join(body_text_test.values)
        
        with open("label_data_train.txt", "w") as f:
            f.write(label_data_train)

        with open("training_data_train.txt", "w") as f:
            f.write(training_data_train)

        with open("label_data_test.txt", "w") as f:
            f.write(label_data_test)

        with open("training_data_test.txt", "w") as f:
            f.write(training_data_test)
    
    def concatenate_abstracts(self, cluster_label_path):
        '''
        Concatenates abstracts in a cluster together for each cluster.
        
        PARAMETERS
        ----------
        cluster_label_path: (str)'''
        header = ["label"]
        y = pd.read_csv(cluster_label_path, names=header)
        unique_clusters = y['label'].unique()
        abstracts = []
        
        for cluster in unique_clusters:  
            abstract_index = [idx for idx, el in enumerate(y['label']) if el == cluster]
            abstracts.append(self._df_covid_.iloc[abstract_index]['abstract'].values)                    
        
        self.concatenated_abstracts = []
        for idx, cluster in enumerate(abstracts):
            self.concatenated_abstracts.append(" ".join(cluster))
        
        self.concatenated_abstracts = [re.sub('\n','', x.lower()) for x in self.concatenated_abstracts]
        
        #concatenated_abstracts.apply(lambda x: re.sub('\n','',x)).apply(lambda x: x.lower())
        self.concatenated_abstracts ="\n".join(self.concatenated_abstracts)
        
        with open("concatenated_abstracts.txt", "w") as f:
            f.write(self.concatenated_abstracts)
class Cluster_Data:
    def __init__(self, data_processor):
        '''
        PARAMETERS
        ----------
        data_processor: (Data_Processor) 
        '''
        self._df_covid_ = data_processor._df_covid_
    
    def __more_data_processing__(self):
        '''
        Remove punctuation and div. Converte to lowercase.
        '''
        
        # remove punctuation, remove citation marks and remove linebreaks on body_text of the papers
        self._df_covid_['body_text'] = self._df_covid_['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
        self._df_covid_['body_text'] = self._df_covid_['body_text'].apply(lambda x: re.sub('\[[1-9]+\]','',x))
        self._df_covid_['body_text'] = self._df_covid_['body_text'].apply(lambda x: re.sub('\n','',x))
        
        # remove punctuation, remove citation marks and remove linebreaks on abstract of the papers
        self._df_covid_['abstract'] = self._df_covid_['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
        self._df_covid_['abstract'] = self._df_covid_['abstract'].apply(lambda x: re.sub('\[[1-9]+\]','',x))
        self._df_covid_['abstract'] = self._df_covid_['abstract'].apply(lambda x: re.sub('\n','',x))
        
        # convert to lower text
        self._df_covid_['body_text'] = self._df_covid_['body_text'].apply(lambda x: x.lower())
        self._df_covid_['abstract'] = self._df_covid_['abstract'].apply(lambda x: x.lower())
        
    def create_2_gram(self):
        '''
        Create a 2gram from the papers body-text.
    
        '''
        self.__more_data_processing__()
        df_body_text = self._df_covid_['body_text']
        # Reduce articles to lists of words and add to words.
        words = []
        for ii in range(0,len(df_body_text)):
            words.append(str(df_body_text.iloc[ii]).split(" "))

        # Remove empty strings
        filtered_words = []
        for word in words:
            word = list(filter(None, word))
            filtered_words.append(word)

        # Create 2-gram using filtered words.
        self.n_gram_all = []
        for word in filtered_words:
            # get n-grams for the instance
            n_gram = []
            for i in range(len(word)-2+1):
                n_gram.append("".join(word[i:i+2]))
            self.n_gram_all.append(n_gram)   
    
    
    def vectorize(self, file_path, n_features = 2**12):
        '''
        Vectorize a ngram using HashingVectorizer
        Code from: https://www.kaggle.com/maksimeren/covid-19-literature-clustering
        
        PARAMETERS
        ----------
        file_path: (str) path and name of file to save
        n_features: (int) number of features for each vector representation
        '''
        
        self.create_2_gram()
        # hash vectorizer instance
        hvec = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=n_features)
        
        # features matrix X
        self.X = hvec.fit_transform(self.n_gram_all)
        
        with open(file_path, "wb") as fp:   #Pickling
            pickle.dump(self.X, fp)
    
    def load_vectorize(self, file_path):
        '''
        Loads the saved vector file
        
        PARAMETERS
        ----------
        file_path: (str) path to vector file
        '''
        with open(file_path, "rb") as fp:   # Unpickling
            self.X = pickle.load(fp)
    
    def perform_umap(self, save_path):
        '''
        Perform 2D umap embedding of the vectorized ngram and save it to a file.
        
        PARAMETERS
        ----------
        save_path: (str) The path to save the 2D umap embedding of the vectorized ngram to.
        '''
        
        # run umap on the vectorized 2gram
        model = umap.UMAP()
        X_array = self.X.toarray()
        self.X_embedding_umap = model.fit_transform(X_array)
        
        # save the embedding
        X_embedding_df_umap = pd.DataFrame(X_embedding_umap)
        X_embedding_df_umap.to_csv(save_path, index=False, header=False)
    
    def load_umap(self, file_path):
        '''
        Loads the saved umap file
        
        PARAMETERS
        ----------
        file_path: (str)
        '''
        header = ['comp1','comp2']
        self.X_embedded_umap = pd.read_csv(file_path, names = header)
    
    def perform_hdbscan(self, save_path, min_cluster_size = 3, use_embedding = False):
        '''
        Performs clustering using hdbscan.
        
        PARAMETERS
        ----------
        save_path: (str) The path to save the hdbscan cluster label to.
        '''
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
        if(use_embedding):
            clusterer.fit(self.X_embedded_umap)
        else:
            clusterer.fit(self.X)
        self.y = clusterer.labels_
        y_df = pd.DataFrame(self.y)
        y_df.to_csv(save_path, index=False, header=False)
    
    def load_hdbscan(self, file_path):
        '''
        Loads a csv file containing the clustering labels of each paper into a pandas dataframe.
        
        PARAMETERS
        ----------
        file_path: (str) The file path to the hdbscan clustering labels
        '''
        header =['cluster_label']
        y_df = pd.read_csv(file_path, names = header)
        self.y = y_df['cluster_label']
    
    def plot(self):
        '''
        Plots the 2D UMAP embedding of Covid-19 papers with HDBSCAN clustering
        Code from: https://www.kaggle.com/maksimeren/covid-19-literature-clustering


        '''
        # sns settings
        sns.set(rc={'figure.figsize':(15,15)})

        # colors
        palette = sns.color_palette("bright", len(set(self.y)))

        # plot
        sns.scatterplot(self.X_embedded_umap['comp1'], self.X_embedded_umap['comp2'], hue=self.y, legend='full', palette=palette)
        plt.title("UMAP embedding with HDBSCAN clustering of Covid-19 papers")
        plt.savefig("plots/umap_covid19_label_HDBSCAN.png")
        plt.show()
    
class Create_Glove:
    def __init__(self, data_processor):
        '''
        PARAMETERS
        ----------
        data_processor: (Data_Processor) 
        '''
        self._df_covid_ = data_processor._df_covid_
        print('''Import and download if not installed already.
                 import nltk
                 nltk.download('stopwords')
                 nltk.download('wordnet')''')
    
    def createLines(column, file_path):
        '''
        Prepare the data for Glove
        
        PARAMETERS
        ----------
        column: (DataFrame) the dataframe to be made a corpus of
        file_path: (str) path and name of file to save
        '''
        texts = column.apply(lambda x: re.sub('\[[1-9]+\]','',x)).apply(lambda x: re.sub('\n','',x)).apply(lambda x: re.sub('\s+',' ',x)).apply(lambda x: re.sub(r'\s+(?:\.)','',x))

        large_text = ''
        for text in texts:
            large_text += text
        lines = re.split(r'\.\s*', large_text)

        stop_words=set(stopwords.words('english'))
        lines_without_stopwords=[]

        for line in lines:
            temp_line=[]
            splitted = line.split(' ')
            for word in splitted:
                if word not in stop_words:
                    temp_line.append(word)

            string = ' '
            lines_without_stopwords.append(string.join(temp_line))


        lines_without_comma = []
        for line in lines_without_stopwords:    
            new_line = line.replace(',', '')
            lines_without_comma.append(new_line)

        wordnet_lemmatizer = WordNetLemmatizer()
        lines_with_lemmas=[]

        for line in lines_without_comma:
            temp_line=[]
            splitted = line.split(' ')
            for word in splitted:
                temp_line.append(wordnet_lemmatizer.lemmatize(word))
            string = ' '
            lines_with_lemmas.append(string.join(temp_line))

        # Tokenize
        self.new_lines=[]
        for line in lines_with_lemmas:
            new_line = line.split(' ')
            self.new_lines.append(new_line)
        
        with open(file_path, "wb") as fp:   #Pickling
            pickle.dump(self.new_lines, fp)
        
    def load_lines(self, file_path):
        '''
        Load the prepared data for Glove
        
        PARAMTERS
        ---------
        file_path: (str) path and name of file to save
        '''
        with open(file_path, "rb") as fp:   # Unpickling
            self.new_lines = pickle.load(fp)
    
    def train_glove(self, column, lines_path, glove_path, no_components = 300, learning_rate = 0.05, epochs = 30, no_threads = 4):
        '''
        Train and save the glove.
        Code from: https://medium.com/analytics-vidhya/word-vectorization-using-glove-76919685ee0b
        
        PARAMTERS
        ---------
        column: (DataFrame) the dataframe to be made a corpus of
        lines_path: (str) the file path of the preprocessed data
        glove_path: (str) path and name of glove file to save
        no_components: (int) the number of components in each vector
        learning_rate: (float) the rate of learning
        epochs: (int): Number of epochs
        no_threads: (int) The number of threads to be utilized
        '''
        if not self.new_lines:
            self.load_lines(lines_path)
        else:
            self.create_lines(column, lines_path)
        # creating a corpus object
        corpus = Corpus() 
        #training the corpus to generate the co occurence matrix which is used in GloVe
        corpus.fit(self.new_lines, window=10)
        #creating a Glove object which will use the matrix created in the above lines to create embeddings
        #We can set the learning rate as it uses Gradient Descent and number of components
        self.glove = Glove(no_components=no_components, learning_rate=learning_rate)

        self.glove.fit(corpus.matrix, epochs=epochs, no_threads=no_threads, verbose=True)
        self.glove.add_dictionary(corpus.dictionary)
        self.glove.save(glove_path)
        
    def load_glove_model(self, glove_path):
        '''
        Load the trained glove model
        
        PARAMETERS
        ----------
        glove_path: (str) 
        path and name of file to load
        '''
        self.glove = Glove.load(glove_path)
    
    def extract_word_embedding():
        '''
        Creates text file of word embedding out of glove model
        '''
        dic = self.glove_model.__dict__
        inverse_dictionary = dic['inverse_dictionary']
        word_vectors = dic['word_vectors']

        file = open("glove_covid19.txt", "w")
        num_words = len(inverse_dictionary)
        print("Total of " + str(num_words) + " words, each with an emedding of size: " + str(len(word_vectors[0])))
        for i in range(num_words):
            if (i % 100000 == 0):
                print("Created " + str(i) + "/" + str(num_words) + " lines.")
            word = inverse_dictionary[i]
            word_weights = word_vectors[i]

            line = str(word) + " "
            for weight in word_weights:
                line += str(weight) + " "
            file.write(line)
            file.write("\n")
        file.close()
root_path = '/home/bsh/Ex3/'
covid_data_file = 'metadata.csv'
data_processor = Data_Processor(root_path, covid_data_file)
covid_data_file = 'covid_data.csv'
if (os.path.exists(covid_data_file)):
    data_processor.load_df_covid(covid_data_file)
else:
    data_processor.__create_df_covid__(covid_data_file)
data_processor.split_data()
data_clusterer = Cluster_Data(data_processor)
vector_file = 'covid_vectorized.txt'
if(os.path.exists(vector_file)):
    data_clusterer.load_vectorize(vector_file)
else:
    data_clusterer.vectorize(vector_file)
umap_file = 'X_embedding_umap.csv'
if(os.path.exists(umap_file)):
    data_clusterer.load_umap(umap_file)
else:
    data_clusterer.perform_umap(umap_file)
hdbscan_file = 'y_HDBSCAN_3.csv'
if(os.path.exists(hdbscan_file)):
    data_clusterer.load_hdbscan(hdbscan_file)
else:
    data_clusterer.perform_hdbscan(hdbscan_file, use_embedding=False)
data_clusterer.plot()
embedded_hdbscan_file = 'y_HDBSCAN_51_embedded.csv'
if(os.path.exists(embedded_hdbscan_file)):
    data_clusterer.load_hdbscan(embedded_hdbscan_file)
else:
    data_clusterer.perform_hdbscan(embedded_hdbscan_file, min_cluster_size =51, use_embedding=True)
data_clusterer.plot()
glove_creator = Create_Glove(data_processor)
column = data_processor._df_covid_['body_text']
lines_path = 'body_text_Corpus_lines.txt'
glove_model_path =  'body_text_glove.model'

if(os.path.exists(glove_model_path)):
    glove_creator.train_glove(column = column, lines_path = lines_path, glove_path = glove_model_path)
else:
    glove_creator.load(glove_model_path)
glove_creator.extract_word_embedding()
embedding_size = 300
num_epochs = 100
batch_size = 362

%run train.py --glove --embedding_size embedding_size --num_epochs num_epochs --batch_size batch_size  --with_model
data_processor.concatenate_abstracts(embedded_hdbscan_file)
cluster_label_path = '' # TODO: Give path
data_processor.concatenate_abstracts(cluster_label_path)
%run test.py