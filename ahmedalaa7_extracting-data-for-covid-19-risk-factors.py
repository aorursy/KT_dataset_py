!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz

import numpy as np 
import pandas as pd
import scispacy
import spacy
import en_core_sci_lg
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import numpy as np
import pandas as pd

root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path)
meta_df.head()
covid_research_papers = meta_df[meta_df['abstract'].astype(str).str.contains('COVID-19|SARS-CoV-2|2019-nCov|SARS Coronavirus 2|2019 Novel Coronavirus')]
covid_abstract = covid_research_papers.abstract
covid_abstract.shape
nlp = en_core_sci_lg.load()

#Tokenizing and simple preprocessing of the documents to remove stop words, stemming and lemmatization of the words.

def spacy_tokenizer(sentence):
    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]

vectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer, min_df=2)
data_vectorized = vectorizer.fit_transform(tqdm(covid_abstract.values.astype('U')))
data_vectorized.shape
# most frequent words
word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'sum of tf-idf': np.asarray(data_vectorized.sum(axis=0))[0]})

word_count.sort_values('sum of tf-idf', ascending=False).set_index('word')[:20].sort_values('sum of tf-idf', ascending=True).plot(kind='barh')
def compute_cosine_similarity(doc_features, corpus_features, top_n=10):
    # get document vectors
    doc_features = doc_features.toarray()[0]
    corpus_features = corpus_features.toarray()
    # compute similarites
    similarity = np.dot(doc_features, corpus_features.T)
    # get docs with highest similarity scores
    top_docs = similarity.argsort()[::-1][:top_n]
    top_docs_with_score = [(index, round(similarity[index], 3)) for index in top_docs]
    
    return top_docs_with_score
from IPython.display import display, HTML
import numpy as np
#Find the 10 most releveant papers to a given query and display them
def SearchDocuments(Query):
    query_docs_tfidf = vectorizer.transform(Query) #Vectorizing and calculating tf-idf for the query

    for index, doc in enumerate(Query):
        doc_tfidf = query_docs_tfidf[index]
        #Computing Cosine similarty between the query and the abstracts and get the 10 most relevant
        top_similar_docs = compute_cosine_similarity(doc_tfidf, data_vectorized, top_n=10)
        
        df = pd.DataFrame()
        Score=[]
        for doc_index, sim_score in top_similar_docs :
            #Getting the full data of the 10 most relevant papers and add them to the dataframe
            data =meta_df.loc[meta_df['cord_uid'] == covid_research_papers.cord_uid.values[doc_index]]
            Score.append(str(sim_score))
            df = df.append(data)

        df['Score']=Score
        # Display the relevant papers in a table
        DisplayTable(df)
        
def AnswerSearchQuery(Query):
    query_docs_tfidf = vectorizer.transform(Query) #Vectorizing and calculating tf-idf for the query

    for index, doc in enumerate(Query):
        doc_tfidf = query_docs_tfidf[index]
        #Computing Cosine similarty between the query and the abstracts and get the 10 most relevant
        top_similar_docs = compute_cosine_similarity(doc_tfidf, data_vectorized, top_n=1)
        result = covid_abstract.values[top_similar_docs[0][0]].split('Results: ')
        if(len(result)==1):
            print(covid_abstract.values[top_similar_docs[0][0]])
        else:
            print(result[1])

#Displaying the dataframe in a table and styling
def DisplayTable(df):
    df = df.replace(np.nan, '', regex=True)
    df['Title'] = df['title'] + '#' + df['url']
    df =df[['Title','publish_time','abstract','Score']]
    dfStyler =df.style.format({'Title': make_clickable_both,'text-align': 'right'})
    dfStyler = dfStyler.set_properties(**{'text-align': 'left'})
    dfStyler=dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    display(HTML(dfStyler.render()))
#Making the title of the paper in the table as Hyperlink to get access to the full text paper    
def make_clickable_both(val): 
    
    name, url = val.split('#')
    if(url==''):
        return name
    return f'<a href="{url}">{name}</a>'

SearchDocuments(['COVID-19 risk factors'])
SearchDocuments(['Data on potential risks factors'])
SearchDocuments(['Risk factors such as Smoking, pre-existing pulmonary disease'])
SearchDocuments(['Risk factors such as Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities'])
SearchDocuments(['Risk factors for Neonates and pregnant women'])
SearchDocuments(['Risk factors for Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences'])
SearchDocuments(['Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors'])
SearchDocuments(['Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups'])
SearchDocuments(['Susceptibility of populations'])
SearchDocuments(['Public health mitigation measures that could be effective for control'])
SearchDocuments(['antiviral treatment'])
SearchDocuments(['risk factors such as age'])
SearchDocuments(['risk factors such as pollution'])
SearchDocuments(['risk factors such as population density'])
SearchDocuments(['risk factors such as humidity'])
SearchDocuments(['risk factors such as heart risks'])
SearchDocuments(['risk factors such as temperature'])
AnswerSearchQuery(['Risk factors such as Smoking, pre-existing pulmonary disease'])
AnswerSearchQuery(['Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors'])
AnswerSearchQuery(['Risk factors for Neonates and pregnant women'])
AnswerSearchQuery(['COVID-19 risk factors'])
print('Please Write down your own Query')
Query= input()
SearchDocuments([Query])