data_path = '/kaggle/input/CORD-19-research-challenge/'#'../../data/raw/'



import pandas as pd

df = pd.read_csv(data_path+"metadata.csv")
!pip install langdetect

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0





def get_language(text):

    text = str(text)

    try:

        language = detect(text)

    except:

        language = "error"

        print("This row throws and error:", text)

    return language





df["language"] = df["title"].apply(lambda x: get_language(x))
print(df.shape)

df = df[df.language == 'en']

print(df.shape)
trim_data = df[["cord_uid", "abstract"]]

print(trim_data.shape)
trim_data = trim_data.dropna()

print(trim_data.shape)
import nltk

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

import re

import string



nltk.download('stopwords')

nltk.download('punkt')





def clean(doc, tokenize=False):

    doc = str(doc)

    stop_free = " ".join([i for i in doc.lower().split() if i not in stopwords.words('english')])

    punc_free = ''.join(ch for ch in stop_free if ch not in set(string.punctuation))

    normalized = " ".join(WordNetLemmatizer().lemmatize(word, pos="v") for word in punc_free.split())

    processed = re.sub(r"\d+","",normalized)

    if tokenize is True:

        processed = nltk.word_tokenize(processed)

    return processed
trim_data["processed"] = trim_data["abstract"].apply(clean)

trim_data.head()
df["abstract_processed"] = trim_data["processed"]

del trim_data

df.head()
df['tokens'] = df['abstract_processed'].apply(lambda x: nltk.word_tokenize(str(x)))
covid19_names = {

    'COVID19',

    'COVID-19',

    '2019-nCoV',

    '2019-nCoV.',

    # 'novel coronavirus',  # too ambiguous, may mean SARS-CoV

    'coronavirus disease 2019',

    'Corona Virus Disease 2019',

    '2019-novel Coronavirus',

    'SARS-CoV-2',

    'covid-19', 

    'covid 19',

    'covid-2019',

    '2019 novel coronavirus', 

    'corona virus disease 2019',

    'coronavirus disease 19',

    'coronavirus 2019',

    '2019-ncov',

    'ncov-2019', 

    'wuhan virus',

    'wuhan coronavirus',

    'wuhan pneumonia',

    #'NCIP', commented to fix priNCIPal problem

    'sars-cov-2',

    'sars-cov2'

}





# detect if text contains covid-19 terms

def has_covid19(text):

    for name in covid19_names:

        if text and str(name).lower() in str(text).lower():

            return True

    return False





df['title_has_covid19'] = df.title.apply(has_covid19)

df['abstract_has_covid19'] = df.abstract.apply(has_covid19)



df['has_covid19'] = df['title_has_covid19'] | df['abstract_has_covid19']



del df['title_has_covid19']

del df['abstract_has_covid19']
df.groupby('has_covid19').size()
from gensim.models import Word2Vec

EMBEDDING_DIM = 50

word2vec = Word2Vec(sentences=df['tokens'], size=EMBEDDING_DIM, window=5, workers=4, min_count=1)
!pip install rank-bm25

from rank_bm25 import BM25Okapi

bm25 = BM25Okapi(df['tokens'])
import json

import glob



def create_bodytext_dataframe(dataframe):



    df_body = pd.DataFrame(columns=['sha', 'text', 'tokens'])



    for index, row in dataframe.iterrows():

        file_name = row['sha']

        file_list = glob.glob(data_path+'**/**/**/'+str(file_name)+'.json')

        if len(file_list) > 0:

            file_path = file_list[0]

        else:

            #print('File '+str(file_name)+' not found... :(')

            continue



        with open(file_path) as json_data:

            data = json.load(json_data)

            body_list = [bt['text'] for bt in data['body_text']]



            # each json has a series of segments, maybe paper pages...

            for json_segment in body_list:

                # split segments into sentences [this can be improved: (fig. 11) will be splitted]

                sentences = json_segment.split(". ")

                for sentence in sentences:

                    df_body = df_body.append({

                        'sha': file_name,

                        'text': sentence,

                        'tokens': clean(sentence, tokenize=True)

                    }, ignore_index=True)



    return df_body
def search(query, threshold=0.7, N=5, covid19_only=True):

    tokenized = clean(query, tokenize=True)

    

    #remove verbs from tokens

    verb_tags = ['VB' ,'VBD', 'VBG', 'VBN', 'VBZ'] #'VBP'

    tags = nltk.pos_tag(tokenized)

    for tuple_ in tags:

        if tuple_[1] in verb_tags:

            tokenized.remove(tuple_[0])



    keywords_ = []

    # add to keywords if tokens are in word2vec dictionary

    for token in tokenized:

        if token in word2vec.wv.vocab:

            keywords_.append(token)



    keywords = keywords_[:]

    # search for keywords related to query

    for kw in keywords_:

        most_similar = word2vec.wv.most_similar(positive=[kw])

        for word in most_similar:

            if word[1] > threshold:

                keywords.append(word[0])



    # rank papers based on keywords similarity

    doc_scores = bm25.get_scores(keywords)



    # add scores to df

    df['score'] = doc_scores



    # create dataframe with Top N results (filter covid-19 terms)

    if covid19_only is False:

        df_result = df.sort_values(by=['score'])[::-1].head(N)

    else:

        df_result = df[df['has_covid19'] == True].sort_values(by=['score'])[::-1].head(N)



    # create dataframe with the bodies text

    df_bodies = create_bodytext_dataframe(df_result)



    # create a local bm25 model with the tokens from body text

    local_bm25 = BM25Okapi(df_bodies['tokens'])

    sentence_scores = local_bm25.get_scores(keywords)

    df_bodies['score'] = sentence_scores



    # add top X sentences in each paper in df_results as highlight1, 2 and 3

    X = 3

    for index, row in df_result.iterrows():

        best_sentences = df_bodies[df_bodies['sha'] == row['sha']].sort_values(by=['score'], ascending=False)['text'][:X]

        for idx, text in enumerate(best_sentences):

            df_result.at[index, 'highlight'+str(idx+1)] = text



    return df_result, query, keywords
from IPython.core.display import display, HTML





def result_display(result, query, keywords):

    display(HTML(

        '<h2 style="color:#ff6600">'+query+'</h2>' +

        '<p><b>Keywords: </b>'+(','.join(keywords))+'</p>'

    ))



    for index, row in result.iterrows():

        

        # display title and abstract

        display(HTML(

            '<h3 style="color:#ffa64d">' + row['title'] + '</h3>' +

            '<p><b>' + str(row['publish_time']) + '</b><i> ' + str(row['journal']) +'</i></p>'+

            '<p>' + str(row['abstract']) + '</p>'

        ))

        

        # display highlights

        if pd.isnull(row['highlight1']):

            display(HTML('<p><b>No highlights to show :(</b></p>'))

        else:

            display(HTML(

                '<p><b>Highlights:</b></p>' +

                '<ul>' +

                  '<li>'+str(row['highlight1'])+'</li>' +

                  '<li>'+str(row['highlight2'])+'</li>' +

                  '<li>'+str(row['highlight3'])+'</li>' +

                '</ul>'

            ))

        

        # display paper link

        display(HTML(

            '<p><a href='+str(row['url'])+'>Link to paper</a></p>' +

            '<br>'

        ))
query_list = [

    "What do we know about virus genetics, origin, and evolution?",

    "Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.",

    "Animal host(s) and any evidence of continued spill-over to humans"

]



for query_ in query_list:

    result, query, keywords = search(query_)

    result_display(result, query, keywords)