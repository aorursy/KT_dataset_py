import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import json #json reader to python dict
import os #create system directory

import re  # For preprocessing
import spacy  # For preprocessing

import multiprocessing # For get number of cores

from time import time  # To time our operations
from collections import defaultdict  # For word frequency


from gensim.models.phrases import Phrases, Phraser #For create relevant phrases
from gensim.models import Word2Vec #Our model type


#import logging  # Setting up the loggings to monitor gensim
#logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
HOME_DIR = os.path.realpath('..')
INPUT_DIR = os.path.join(HOME_DIR, 'input')
OUTPUT_DIR = os.path.join(HOME_DIR, 'output')

DATA_DIR = os.path.join(INPUT_DIR, 'CORD-19-research-challenge')
MODELS_DIR = os.path.join(INPUT_DIR, 'models')
RANDOM_VEC_DIR = os.path.join(INPUT_DIR, 'random-vectors-similarity')
def cleaned_metadata_df(metadata):
    """Prepare metadata to work"""
    
    print('Shape original:' , metadata.shape)
    
    result_df = metadata.copy()
    result_df = result_df.drop_duplicates()
    result_df = result_df[result_df.has_pdf_parse]
    
    print('Shape after cleaning:' ,result_df.shape)
    return result_df    
def cleaning(spacy_doc):
    """Lemmatizes and removes stopwords 
        doc needs to be a spacy Doc object """
    txt = [token.lemma_ for token in spacy_doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)
def Word2Vec_Preprocessing(df, text_col_name):
    """ Prepare column from pd.DataFrame with text to Word2Vec model """
    nlp = spacy.load('en', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
    brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df[text_col_name])

    t = time()
    txt = [] 
    for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1):
        txt.append(cleaning(doc))
    #print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

    df_clean = pd.DataFrame({'clean': txt})
    df_clean = df_clean.dropna().drop_duplicates()
    return df_clean
def create_sentences(df, col_name):
    """Build sentences to Word2Vec model """
    t=time()
    sent = [row.split() for row in df[col_name]]
    phrases = Phrases(sent, min_count=30, progress_per=100)
    bigram = Phraser(phrases)
    
    #print('Time to create sentences: {} mins'.format(round((time() - t) / 60, 2)))
    return bigram[sent]
def create_word_frequency_dict(sentences):
    """ Counting of token in sentences """
    t=time()
    word_freq = defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1
    #print('Time to create_word_frequency_dict: {} mins'.format(round((time() - t) / 60, 2)))
    return word_freq
def cosine_similarity(vec_1, vec_2):
    """ Correlation between Word2Vec vectors """
    return np.dot(vec_1, vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2))
def Initialization_Word2Vec_model():
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=20,
                     window=5,
                     size=150,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores)
    return w2v_model
def Build_Word2Vec_vocab(w2v_model, sentences,update=True):
    t = time()
    w2v_model.build_vocab(sentences, progress_per=10000, update=update)
    #print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    return w2v_model
def Train_Word2Vec_model(w2v_model, sentences):
    t = time()
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=20, report_delay=1)
    #print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    
    #w2v_model.init_sims(replace=True)
    
    return w2v_model
def add_full_text_to_metadata(metadata_df):
    t=time()
    len_metadata_df = len(metadata_df)
    for _, row in metadata_df.iterrows():
        sha_cde = str(row['sha']).split('; ')
        full_text_file = row['full_text_file']

        full_text = ''
        
        if _ % 10000 == 0: print(f'{_} full_texts was added!')
        
        for sha in sha_cde: 
            path_full_text = os.path.join(DATA_DIR, f'{full_text_file}/{full_text_file}/pdf_json/{sha}.json'.format(full_text_file, sha))
            try:
                with open(path_full_text) as json_data:
                        data = json.load(json_data)
                full_text = full_text + ' '.join([page['text'] for page in data['body_text']])
            except:
                pass
        metadata_df.loc[row.name, 'full_text'] = str(metadata_df.loc[row.name, 'abstract']) + ' ' + full_text

    print('Time to add full_text: {} mins'.format(round((time() - t) / 60, 2)))
    return metadata_df
def chunk_Word2Vec_Preprocessing(metadata_df, chunk_size = 50):
    part_idx = 0 
    part_size = int(np.round(len(metadata_df)/chunk_size + 1))
    df_clean = dict()
    for idx in range(1,chunk_size+1):
        df_clean[idx] = Word2Vec_Preprocessing(metadata_df.iloc[part_idx:part_idx+part_size,:], 'full_text')
        part_idx = part_idx+part_size
        print('End of cleaning part {}'.format(idx))

    df_clean_all = pd.concat(list(df_clean.values()))
    df_clean_all.to_pickle(os.path.join(OUTPUT_DIR, '/kaggle/working/df_clean_all.pkl'))
    
    return df_clean_all
def Build_sentence_vector(words):
    result_vector = [0]*len(w2v_model.wv.get_vector(words[0]))
    for word in words:
        result_vector = (result_vector + w2v_model.wv.get_vector(word))
    
    return result_vector/np.linalg.norm(result_vector)
def generate_article_vectors(from_column='full_text'):

    w2v_vocab_set = set(w2v_model.wv.vocab.keys())
    article_vectors = dict()

    for _, row in metadata_df.iterrows():

        article_words_set = str(row[from_column]).split()
        words = [word for word in article_words_set if word in (w2v_vocab_set)]
        if len(words):
            article_vectors[row.name] = Build_sentence_vector(words)
            
    return article_vectors
def generate_question_vector(effort_idx):
    question_words = Word2Vec_Preprocessing(efforts_df.iloc[effort_idx:effort_idx+1],'txt').iloc[0,0].split()
    w2v_vocab_set = set(w2v_model.wv.vocab.keys())
    words = list(w2v_vocab_set.intersection(question_words))
    return Build_sentence_vector(words)
def generate_articles_indexies(question_vector, article_vectors):
    similarity_values = dict()
    for key in article_vectors.keys():
        similarity_values[key] = cosine_similarity(question_vector,article_vectors.get(key))

    articles_indexies = sorted(similarity_values, key= similarity_values.get, reverse=True)
    print(articles_indexies[:10])

    return similarity_values, articles_indexies
def find_part_of_article_to_efforts(example_full_text_df, question_vector, effort_idx, article_index, step=150, prc_of_article=0.04):
    w2v_vocab_set = set(w2v_model.wv.vocab.keys())
    spliting_words = example_full_text_df['clean'][0].split()
    spliting_words = [word for word in spliting_words if word in w2v_vocab_set]
    similarity_lst = []
    simil_value = -1
    for idx, word in enumerate(spliting_words):

        part_sentence_vec = Build_sentence_vector(spliting_words[idx:idx+step])
        new_simil_value = cosine_similarity(question_vector, part_sentence_vec)
        similarity_lst.append(new_simil_value)
        if new_simil_value > simil_value:
            simil_idx = idx
            simil_value = new_simil_value

    full_text_rate = simil_idx/len(spliting_words) - prc_of_article/2, simil_idx/len(spliting_words) + prc_of_article/2
    if full_text_rate[0] < 0: full_text_rate = 0, full_text_rate[1] 
    begin_sign = int(round(full_text_rate[0] * len(metadata_df.loc[article_index, 'full_text']),0))
    end_sign = int(round(full_text_rate[1] * len(metadata_df.loc[article_index, 'full_text']),0))

    begin_part_sent, end_part_sent = int(full_text_rate[0]*len(similarity_lst)), int(full_text_rate[1]*len(similarity_lst))
    begin_part_sent, end_part_sent = np.maximum(0, begin_part_sent), np.minimum(len(similarity_lst), end_part_sent)
        
    plt.figure(figsize=(15, 5))
    plt.title('The most similar part of article')
    plt.plot(similarity_lst);
    plt.plot(range(begin_part_sent, end_part_sent),similarity_lst[begin_part_sent:end_part_sent])
    plt.show();

    print('...', metadata_df.loc[article_index, 'full_text'][begin_sign:end_sign], '...')

    return None
def get_effort_article(article_vectors, effort_idx, nr_of_similar_article = 0):
    
    question_vector = generate_question_vector(effort_idx)
    similarity_values, articles_indexies = generate_articles_indexies(question_vector, article_vectors)
    article_index = articles_indexies[nr_of_similar_article]
    print('--------------------------------------------------------------------')
    print(f'TITLE [{article_index}]: ', metadata_df.loc[article_index,'title'])
    print('--------------------------------------------------------------------')
    print('---------------------------')
    print('Publish time: {}'.format(metadata_df.loc[article_index, 'publish_time']))
    print('---------------------------')
    
    print('--------------------------------------------------------------------')
    print('efforts ', efforts_df['txt'][effort_idx])
    print('--------------------------------------------------------------------')
    
    print('similarity effort to full text is: ', similarity_values.get(article_index) ,'\n')
    example_full_text_df  = Word2Vec_Preprocessing(metadata_df.loc[article_index:article_index,:],'full_text')
    find_part_of_article_to_efforts(example_full_text_df, question_vector, effort_idx, article_index)
    return None
metadata = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'),)
metadata_df = cleaned_metadata_df(metadata)
# #w2v_model = Initialization_Word2Vec_model()
# row_cnt = 27174
# w2v_model = Word2Vec.load(os.path.join(INPUT_DIR, 'models/w2v_model_on_all_abstract_full_text_after_{}_rows.w2v'.format(row_cnt)))
# chunk_size = int(len(metadata_df)/100 + 1)
# for idx in range(0,100): 
    
#     chunk_df = add_full_text_to_metadata(metadata_df.iloc[row_cnt:row_cnt+chunk_size,:])
#     df_clean = chunk_Word2Vec_Preprocessing(chunk_df, 1)
#     sentences = create_sentences(df_clean, 'clean')

#     #word_freq = create_word_frequency_dict(sentences)
#     #sorted(word_freq, key=word_freq.get, reverse=True)[:10]

#     update_w2v_vocab = len(w2v_model.wv.vocab) != 0
#     w2v_model = Build_Word2Vec_vocab(w2v_model, sentences, update_w2v_vocab)
#     w2v_model = Train_Word2Vec_model(w2v_model, sentences)

#     row_cnt = row_cnt+chunk_size
#     w2v_model.save(os.path.join(OUTPUT_DIR, '/kaggle/working/w2v_model_on_all_abstract_full_text_after_{}_rows.w2v'.format(row_cnt)))
#     print('-----------------------------------')
#     print('| Model after {} rows was built   |'.format(row_cnt))
#     print('-----------------------------------')
    
#     remove_path = os.path.join(OUTPUT_DIR,'/kaggle/working/w2v_model_on_all_abstract_full_text_after_{}_rows.w2v'.format(row_cnt - 3*chunk_size))
#     if os.path.exists(remove_path):
#         os.remove(remove_path)
    
# w2v_model.init_sims(replace=True)
### Load a model after 2 days training. Model was saved and now I get there from INPUT_DIR
w2v_model = Word2Vec.load(os.path.join(MODELS_DIR, 'w2v_model_on_all_abstract_full_text_after_34377_rows.w2v'))
w2v_model.init_sims(replace=True)
#Time to add full_text ~ 6:30 min
metadata_df = add_full_text_to_metadata(metadata_df)
efforts_df = pd.DataFrame({'txt':[
                          "articulate and translate existing ethical principles and standards to salient issues in COVID-2019"
                         ,"embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight"
                         ,"support sustained education, access, and capacity building in the area of ethics"
                         ,"establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences."
                         ,"develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)"
                         ,"identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed."
                         ,"identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.                        "
                        ]})
article_vectors = generate_article_vectors()
get_effort_article(article_vectors, effort_idx=0, nr_of_similar_article=8)
get_effort_article(article_vectors, effort_idx=1)
get_effort_article(article_vectors, effort_idx=2, nr_of_similar_article=5)
get_effort_article(article_vectors, effort_idx=3, nr_of_similar_article=4)
get_effort_article(article_vectors, effort_idx=4)
get_effort_article(article_vectors, effort_idx=5)
## Model characteristic 
model_vocab_len = len(w2v_model.wv.vocab.keys())
model_full_text_train_len = metadata.shape[0]

question_vector = generate_question_vector(2)
similarity_values, articles_indexies = generate_articles_indexies(question_vector, article_vectors)

### Random vectors test
if os.path.exists(os.path.join(RANDOM_VEC_DIR, 'random_vectors_similarity.pkl')):
    random_check = pd.read_pickle(os.path.join(RANDOM_VEC_DIR, 'random_vectors_similarity.pkl'))

else:
    random_check = []
    for i in range(0,model_full_text_train_len):
        random_abstract = [list(w2v_model.wv.vocab.keys())[int(idx)] for idx in np.round(np.random.uniform(1, model_vocab_len-1, 200))]
        random_vector = Build_sentence_vector(random_abstract)
        random_check.append(cosine_similarity(question_vector, random_vector))
    
    #break after 24820 iteration
    pd.Series(random_check).to_pickle('random_vectors_similarity.pkl')

plt.figure(figsize = (15, 5))
plt.hist(list(similarity_values.values()), bins=50);
plt.hist(random_check, bins = 50);
plt.legend(['similarity_values', 'random_check']);
plt.show();