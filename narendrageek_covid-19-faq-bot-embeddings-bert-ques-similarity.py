import os, re, io

import pandas as pd

import numpy as np

import requests



## stopwords

from gensim.parsing.preprocessing import remove_stopwords

## lemma functionality provide by NLTK

from nltk.stem import WordNetLemmatizer

## make sure you downloaded model for lemmatization

#nltk.download('wordnet')

from nltk import word_tokenize

## make sure you downloaded model for tokenization

#nltk.download('punkt')

import spacy

nlp = spacy.load('en')



## TF_IDF for BOW

from gensim.models import TfidfModel

from gensim.corpora import Dictionary

## cosine similarity

from sklearn.metrics.pairwise import cosine_similarity
## The data is taken from https://www.un.org/sites/un2.un.org/files/new_dhmosh_covid-19_faq.pdf

## it has FAQ based question and answering for COVID-19



def download_pdf_url(dataset_url, file_name):

    response = requests.get(dataset_url)

    pdf_content_output = None

    with io.BytesIO(response.content) as open_pdf_file:

        with open(file_name,'w') as obj:

            obj.write(str(open_pdf_file))
dataset_url = 'https://www.un.org/sites/un2.un.org/files/new_dhmosh_covid-19_faq.pdf'

## download pdf from URL and save the pdf file

download_pdf_url(dataset_url, 'new_dhmosh_covid-19_faq.pdf')
## QA will be stored as .csv file

def extract_QA_from_text_file(INPUT_DIR, text_file_name):

    output_file_name = 'covid_19faq.csv'

    with open(os.path.join(INPUT_DIR, text_file_name), 'r', encoding='latin') as obj:

        text = obj.read()



    text = text.strip()

    ## extract the question by following pattern

    pattern = '\n+\s*\d+[.](.*?)\?'

    question_pattern = re.compile(pattern,re.MULTILINE|re.IGNORECASE|re.DOTALL)  

    matched_QA_positions = [(m.start(0),m.end(0)) for m in question_pattern.finditer(text)]

    print(f"Available no of question is {len(matched_QA_positions)}")

    ## store question and answer pair

    questions = {}

    ## iterate every matched QA 

    for index in range(len(matched_QA_positions)):

        ## get the start and end position

        faq_start_pos = matched_QA_positions[index][0]

        faq_end_pos = matched_QA_positions[index][1]

        

        if index == len(matched_QA_positions) - 1:

            next_faq_start_pos = -1

        else:

            next_faq_start_pos = matched_QA_positions[index+1][0]



        ## get the question from start and end position from original text      

        question = text[faq_start_pos:faq_end_pos]

        if next_faq_start_pos == -1:

            answer = text[faq_end_pos:]

        else:

            answer = text[faq_end_pos:next_faq_start_pos]

        ## replace multiple new lines to space in questions and answers

        question = re.sub("\n+"," ",question.strip())

        answer = re.sub("\n+"," ",answer.strip())

        questions[question] = answer

        

    ## create dataframe from key-value pair

    faq_df = pd.DataFrame.from_dict(questions, orient='index', columns=["answers"])

    faq_df["questions"] = faq_df.index

    faq_df.reset_index(inplace=True)  

    faq_df[["questions", "answers"]].to_csv(os.path.join(INPUT_DIR, output_file_name),index = False)

    print(f"COVID QA file {output_file_name} created")
## Converted PDF to .txt file using pdftools in R

## create a question-answer pair in csv



#extract_QA_from_text_file(INPUT_DIR, 'new_dhmosh_covid-19_faq.txt')



QA_df = pd.read_excel(os.path.join("../input/covid19-frequent-asked-questions", "COVID19_FAQ.xlsx"))

QA_df.head(10)
## Data Preprocessing

class TextPreprocessor():

    def __init__(self, data_df, column_name=None):

        self.data_df = data_df  

        if not column_name and type(colum_name) == str:

            raise Exception("column name is mandatory. Make sure type is string format")

        self.column = column_name

        self.convert_lowercase()    

        self.applied_stopword = False

        self.processed_column_name = f"processed_{self.column}"

        

    def convert_lowercase(self):

        ## fill empty values into empty

        self.data_df.fillna('',inplace=True)

        ## reduce all the columns to lowercase

        self.data_df = self.data_df.apply(lambda column: column.astype(str).str.lower(), axis=0)    



    def remove_question_no(self):

        ## remove question no        

        self.data_df[self.column] = self.data_df[self.column].apply(lambda row: re.sub(r'^\d+[.]',' ', row))    

        

    def remove_symbols(self):

        ## remove unwanted character          

        self.data_df[self.column] = self.data_df[self.column].apply(lambda row: re.sub(r'[^A-Za-z0-9\s]', ' ', row))    



    def remove_stopwords(self):

        ## remove stopwords and create a new column 

        for idx, question in enumerate(self.data_df[self.column]):      

            self.data_df.loc[idx, self.processed_column_name] = remove_stopwords(question)        



    def apply_lemmatization(self, perform_stopword):

        ## get the root words to reduce inflection of words 

        lemmatizer = WordNetLemmatizer()    

        ## get the column name to perform lemma operation whether stopwords removed text or not

        if perform_stopword:

            column_name = self.processed_column_name

        else:

            column_name = self.column

        ## iterate every question, perform tokenize and lemma

        for idx, question in enumerate(self.data_df[column_name]):



            lemmatized_sentence = []

            ## use spacy for lemmatization

            doc = nlp(question.strip())

            for word in doc:       

                lemmatized_sentence.append(word.lemma_)      

                ## update to the same column

                self.data_df.loc[idx, self.processed_column_name] = " ".join(lemmatized_sentence)



    def process(self, perform_stopword = True):

        self.remove_question_no()

        self.remove_symbols()

        if perform_stopword:

            self.remove_stopwords()

        self.apply_lemmatization(perform_stopword)    

        return self.data_df
## pre-process training question data

text_preprocessor = TextPreprocessor(QA_df.copy(), column_name="questions")

processed_QA_df = text_preprocessor.process(perform_stopword=True)

processed_QA_df.head(10)
class TF_IDF():

    def __init__(self):

        self.dictionary = None    

        self.model = None

        self.bow_corpus = None



    def create_tf_idf_model(self, data_df, column_name):

        ## create sentence token list

        sentence_token_list = [sentence.split(" ") for sentence in data_df[column_name]]



        ## dataset vocabulary

        self.dictionary = Dictionary(sentence_token_list) 



        ## bow representation of dataset

        self.bow_corpus = [self.dictionary.doc2bow(sentence_tokens) for sentence_tokens in sentence_token_list]



        ## compute TF-IDF score for corpus

        self.model = TfidfModel(self.bow_corpus)



        ## representation of question and respective TF-IDF value

        print(f"First 10 question representation of TF-IDF vector")

        for index, sentence in enumerate(data_df[column_name]):

            if index <= 10:

                print(f"{sentence} {self.model[self.bow_corpus[index]]}")

            else:

                break



    def get_vector_for_test_set(self, test_df, column_name):

        ## store tf-idf vector

        testset_tf_idf_vector = []

        sentence_token_list = [sentence.split(" ") for sentence in test_df[column_name]]

        test_bow_corpus = [self.dictionary.doc2bow(sentence_tokens) for sentence_tokens in sentence_token_list]   

        for test_sentence in test_bow_corpus:

            testset_tf_idf_vector.append(self.model[test_sentence])      



        return testset_tf_idf_vector



    def get_training_QA_vectors(self):

        QA_vectors = []

        for sentence_vector in self.bow_corpus:

            QA_vectors.append(self.model[sentence_vector])      

        return QA_vectors



    def get_train_vocabulary(self):

        vocab = []

        for index in self.dictionary:

            vocab.append(self.dictionary[index])

        return vocab
class Embeddings():

    def __init__(self, model_path):

        self.model_path = model_path

        self.model = None

        self.__load_model__()

        

    def __load_model__(self):

        #word_vectors = api.load("glove-wiki-gigaword-100")  

        model_name = 'glove-twitter-25' #'word2vec-google-news-50' #'glove-twitter-25'  

        if not os.path.exists(self.model_path+ model_name):

            print("Downloading model")

            self.model = api.load(model_name)

            self.model.save(self.model_path+ model_name)

        else:

            print("Loading model from Drive")

            self.model = KeyedVectors.load(self.model_path+ model_name)

        

    def get_oov_from_model(self, document_vocabulary):

        ## the below words are not available in our pre-trained model model_name

        print("The below words are not found in our pre-trained model")

        words = []

        for word in set(document_vocabulary):  

            if word not in self.model:

                words.append(word)

        print(words)  



    def get_sentence_embeddings(self, data_df, column_name):

        sentence_embeddings_list = []

        for sentence in data_df[column_name]:      

            sentence_embeddings = np.repeat(0, self.model.vector_size)

            try:

                tokens = sentence.split(" ")

                ## get the word embedding

                for word in tokens:

                    if word in self.model:

                        word_embedding = self.model[word]

                    else:

                        word_embedding = np.repeat(0, self.model.vector_size)          

                    sentence_embeddings = sentence_embeddings + word_embedding

                ## take the average for sentence embeddings

                #sentence_embeddings = sentence_embeddings / len(tokens)

                sentence_embeddings_list.append(sentence_embeddings.reshape(1, -1))

            except Exception as e:

                print(e)

            

        return sentence_embeddings_list
!pip install bert-embedding
from bert_embedding import BertEmbedding
## get bert embeddings

def get_bert_embeddings(sentences):

    bert_embedding = BertEmbedding()

    return bert_embedding(sentences)
tf_idf = TF_IDF()

tf_idf.create_tf_idf_model(processed_QA_df, "processed_questions")

## get the tf-idf reprentation 

question_QA_vectors = tf_idf.get_training_QA_vectors()
## Get the document vocabulary list from TF-IDF

document_vocabulary = tf_idf.get_train_vocabulary()
## Now, Let's try building embedding based

import gensim.downloader as api

from gensim.models import KeyedVectors
## create Embedding object

embedding = Embeddings("")

## look for out of vocabulary COVID QA dataset - pretrained model

embedding.get_oov_from_model(document_vocabulary)

## get the sentence embedding for COVID QA dataset

question_QA_embeddings = embedding.get_sentence_embeddings(processed_QA_df, "processed_questions")
question_QA_bert_embeddings_list = get_bert_embeddings(processed_QA_df["questions"].to_list())
%matplotlib inline

import matplotlib.pyplot as plt

 

from sklearn.manifold import TSNE
def display_closestwords_tsnescatterplot(model, word):

    

    arr = np.empty((0,25), dtype='f')

    word_labels = [word]



    # get close words

    close_words = model.similar_by_word(word)

    

    # add the vector for each of the closest words to the array

    arr = np.append(arr, np.array([model[word]]), axis=0)

    for wrd_score in close_words:

        wrd_vector = model[wrd_score[0]]

        word_labels.append(wrd_score[0])

        arr = np.append(arr, np.array([wrd_vector]), axis=0)

        

    # find tsne coords for 2 dimensions

    tsne = TSNE(n_components=2, random_state=0)

    np.set_printoptions(suppress=True)

    Y = tsne.fit_transform(arr)



    x_coords = Y[:, 0]

    y_coords = Y[:, 1]

    plt.figure(figsize=(20,10))

    # display scatter plot

    plt.scatter(x_coords, y_coords)



    for label, x, y in zip(word_labels, x_coords, y_coords):

        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)

    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)

    plt.show()
display_closestwords_tsnescatterplot(embedding.model, 'prevent')
## helps to retrieve similar question based of input vectors/embeddings for test query

def retrieveSimilarFAQ(train_question_vectors, test_question_vectors, train_QA_df, train_column_name, test_QA_df, test_column_name):

    similar_question_index = []

    for test_index, test_vector in enumerate(test_question_vectors):

        sim, sim_Q_index = -1, -1

        for train_index, train_vector in enumerate(train_question_vectors):

            sim_score = cosine_similarity(train_vector, test_vector)[0][0]

            

            if sim < sim_score:

                sim = sim_score

                sim_Q_index = train_index



        print("######")

        print(f"Query Question: {test_QA_df[test_column_name].iloc[test_index]}")    

        print(f"Retrieved Question: {train_QA_df[train_column_name].iloc[sim_Q_index]}")

        print("######")
test_query_string = ["how does covid-19 spread?", 

                     "What are the symptoms of COVID-19?",

                "Should I wear a mask to protect myself from covid-19",              

                "Is there a vaccine for COVID-19",

                "can the virus transmit through air?",

                "can the virus spread through air?"]



test_QA_df = pd.DataFrame(test_query_string, columns=["test_questions"])              

## pre-process testing QA data

text_preprocessor = TextPreprocessor(test_QA_df, column_name="test_questions")

query_QA_df = text_preprocessor.process(perform_stopword=True)
## TF-IDF vector represetation

query_QA_vectors = tf_idf.get_vector_for_test_set(query_QA_df, "processed_test_questions")

query_QA_df.head()

      
retrieveSimilarFAQ(question_QA_vectors, query_QA_vectors, processed_QA_df, "questions", query_QA_df, "test_questions")
## get the sentence embedding for COVID QA query

query_QA_embeddings = embedding.get_sentence_embeddings(query_QA_df, "processed_test_questions")



retrieveSimilarFAQ(question_QA_embeddings, query_QA_embeddings, processed_QA_df, "questions", query_QA_df, "test_questions")
query_QA_bert_embeddings_list = get_bert_embeddings(test_QA_df["test_questions"].to_list())
## store QA bert embeddings in list

question_QA_bert_embeddings = []

for embeddings in question_QA_bert_embeddings_list:

    question_QA_bert_embeddings.append(embeddings[1])



## store query string bert embeddings in list

query_QA_bert_embeddings = []

for embeddings in query_QA_bert_embeddings_list:

    query_QA_bert_embeddings.append(embeddings[1])
retrieveSimilarFAQ(question_QA_bert_embeddings, query_QA_bert_embeddings, processed_QA_df, "questions", query_QA_df, "test_questions")