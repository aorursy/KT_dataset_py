import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize 



import spacy

from spacy.lang.en.stop_words import STOP_WORDS



from sklearn.feature_extraction.text import TfidfVectorizer

from itertools import islice



import string

import seaborn as sns; sns.set()



from IPython.display import HTML, display
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv',usecols=['title','journal','abstract','authors','doi','publish_time','sha'])

df=df.fillna('N/A')

publish_times= ['2019','2020']

df=df[df['publish_time'].str.contains('|'.join(publish_times))]



df["key_search"] = df["abstract"].str.lower()+df["title"].str.lower()

covid_keys = ['cov','covid','cov19','covid-19','covid19','-covid','-cov-2','cov2','ncov']

df = df[df['key_search'].str.contains('|'.join(covid_keys))]



df = df.drop_duplicates(subset='title', keep="first")

print (df.shape)

df.head()
punctuations = string.punctuation

stop_words = list(STOP_WORDS)

stop_words += punctuations

stop_words += ['ll','ve','et', 'al','copyright', 'peer']

stop_words[:10]
def tfidf_calculate(df,stop_words):

    abstract = df['abstract'].values

    vectorizer = TfidfVectorizer(max_features=2 ** 12,stop_words=stop_words)

    tfidf = vectorizer.fit_transform(abstract)

    return tfidf,vectorizer



def print_first(n, iterable):

    return list(islice(iterable, n))



def stemming(keys_query):

    stemmer = PorterStemmer()

    stemmed_keys=[]

    for key_query in keys_query:

        stemmed_keys.append(stemmer.stem(key_query))

    return stemmed_keys



def doc_isrelevant(doc,keys_query):

    stemmed_keys = stemming(keys_query)

    new_doc = doc[doc['abstract'].str.contains('|'.join(stemmed_keys))].copy()

    return new_doc



def doc_relevance_calculate(stemmed_keys,doc,vectorizer):

    total_score=0

    df1 = pd.DataFrame(doc.T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])

    df1 = df1.reset_index().rename(columns={"index":"term"})

    total_score = df1.where(df1["term"].str.contains('|'.join(stemmed_keys))).sum(axis = 0, skipna = True)

    return total_score

 

def top_relevant(query,df,stop_words):

    new_docs = doc_isrelevant(df,query)

    stemmed_keys = stemming(query)

    new_docs["total_score"] = 0.0

    tfidf,vectorizer = tfidf_calculate(new_docs,stop_words)

    i=0

    for row in new_docs.itertuples():

        total_score = doc_relevance_calculate(stemmed_keys,tfidf[i],vectorizer)

        new_docs.loc[row.Index,'total_score'] = total_score.tfidf

        i +=1

    sorted_df = new_docs.sort_values(by=['total_score'], ascending=False)

    return sorted_df



def filter_query(query,stop_words):

    query = query.lower()

    word_tokens = word_tokenize(query) 

    filtered_query = [w for w in word_tokens if not w in stop_words] 

    return filtered_query



def process_query(query,df,stop_words):

    filtered_query = filter_query(query,stop_words)

    result = top_relevant(filtered_query,df,stop_words)

    return result



def print_results(data_relevant,query):

    data_relevant=data_relevant.sort_values(by=['total_score'],ascending = False)

    html_ranks = ""

    for i in range(0,5):

        html_ranks += "<h4 style='color: brown;'>Rank: "+str(i+1)+" (Score: "+str(round(data_relevant.iloc[i]['total_score'],3))+")</h4><br/><h4>"+data_relevant.iloc[i]['title']+"</h4><p id='abstract"+str(i)+"' style='margin-top: 20px;margin-bottom: 20px;'>"+data_relevant.iloc[i]['abstract'][:500]+ "...</p><a onclick='show_more("+str(i)+")'></a><br/>"

    html_scripts = "<script type='text/javascript'>function show_more(number){let abstractContent = document.getElementById('abstract'+number); alert(abstractContent);/*abstractContent.innerHTML = \""+html_ranks+"\"*/;}</script>"

    html_show = "<html>"+html_scripts+"<head></head><body><h1 style='color:blue;'> Query: "+query+"</h1><br/>"+html_ranks+"</body></html>"

    display(HTML(html_show))

    return





def plot_scores(data_relevant):

    #test = result1[:2000]

    data_relevant['score_range'] = pd.cut(data_relevant['total_score'], 10)

    data_relevant=data_relevant.sort_values(by=['score_range'],ascending = False)

    bins_obj = data_relevant['score_range'].value_counts()

    

    ax = sns.barplot(x=bins_obj.index.categories, y=bins_obj)

    ax.set_ylabel('Count')

    ax.set_xlabel('Score Range')

    ax.set_title('Count of relative documents in score ranges')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=70, ha='right')

    return
query1 = "Range of incubation periods for the disease in humans."

result1 = process_query(query1,df,stop_words)

print_results(result1,query1)

plot_scores(result1)
query2 = "Incubation varying across age groups"

result2 = process_query(query2,df,stop_words)

print_results(result2,query2)

plot_scores(result2)
query3 = "How long individuals are contagious after recovery?"

result3 = process_query(query3,df,stop_words)

print_results(result3,query3)

plot_scores(result3)
query4 = "Persistence of virus on surfaces of different materials"

result4 = process_query(query4,df,stop_words)

print_results(result4,query4)

plot_scores(result4)
query5 = "Seasons affecting transmission of virus"

result5 = process_query(query5,df,stop_words)

print_results(result5,query5)

plot_scores(result5)
query6 = "Does immune diseases affect recovery?"

result6 = process_query(query6,df,stop_words)

print_results(result6,query6)

plot_scores(result6)
query7 = "Immunity system response to the disease"

result7 = process_query(query7,df,stop_words)

print_results(result7,query7)

plot_scores(result7)
query8 = "Role of the environment in transmission"

result8 = process_query(query8,df,stop_words)

print_results(result8,query8)

plot_scores(result8)
query9 = "Does wearing personal protective equipment such as gloves and masks reduce disease transmission in healthcare community?"

result9 = process_query(query9,df,stop_words)

print_results(result9,query9)

plot_scores(result9)
query10 = "Natural history of the virus and shedding of it from an infected person."

result10 = process_query(query10,df,stop_words)

print_results(result10,query10)

plot_scores(result10)