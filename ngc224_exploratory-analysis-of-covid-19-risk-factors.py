import os

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

import re



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import preprocessing



from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
def load_from_csv(wkgdir):

    

    bio = pd.read_csv(wkgdir + './biorxiv_clean.csv')

    comm = pd.read_csv(wkgdir + './clean_comm_use.csv')

    noncomm = pd.read_csv(wkgdir + './clean_noncomm_use.csv')

    custom = pd.read_csv(wkgdir + './clean_pmc.csv')

    

    corpus = pd.concat([bio,comm,noncomm,custom], ignore_index=True)

    

    return corpus
corpus = load_from_csv('../input/cord-19-eda-parse-json-and-generate-clean-csv/')
corpus.info()
# Replace missing data with NA

corpus = corpus.fillna('not available')
def clean(col):

    col = col.replace('\n', '')

    col = col.replace('\r', '')    

    col = col.replace('\t', '')

    col = re.sub("\[[0-9]+(,[0-9]+)*\]", "", col)

    col = re.sub("\([0-9]+(,[0-9]+)*\)", "", col)

    col = re.sub("\{[0-9]+(,[0-9]+)*\}", "", col)



    return col

    

corpus['abstract'] = corpus['abstract'].apply(clean)

corpus['text'] = corpus['text'].apply(clean)
corpus.rename(columns={'text':'body_text'}, inplace=True)

corpus['text'] = corpus.title + ' ' + corpus.abstract + ' ' + corpus.body_text 
# Drop astract and body text columns

corpus = corpus.drop(['body_text', 'abstract'], axis=1)
# Target term lists

s1 = ['risk', 'risks', 'risk factor', 'risk factors', 'determinant', 'determinants','susceptibility']

s2 = ['smoking', 'smoker', 'smokers', 'smoke', 'smokes', 'cigarette', 'cigarettes', 'tobacco', 'pre-existing respiratory', 'pre-existing pulmonary', 'asthma', 'copd']

s3 = ['coinfection', 'coinfections', 'coexisting respiratory', 'coexisting infection', 'coexisting infections', 'coexisting viral', 'comorbidity', 'comorbidities']

s4 = ['transmissible', 'transmission', 'transmissibility', 'communicable', 'communicability', 'infectivity', 'virulent', 'virulence', 'pathogenicity','contagious','contagion','contagiosity']

s5 = ['neonates', 'neonate', 'newborn', 'newborns', 'pregnant', 'pregnancy', 'pregnancies', 'gestation']

s6 = ['socio-economic', 'socioeconomic', 'economic', 'social behavior', 'behavioral', 'social factors', 'poverty','homelessness','food insecurity','disparity','disparities']

s7 = ['reproductive number', 'incubation period', 'serial interval', 'environmental factors']

s8 = ['severity', 'severe', 'fatal', 'fatality', 'morbidity', 'mortality']

s9 = ['high-risk patients', 'hospitalized patient', 'high-risk patient', 'symptomatic patients', 'symptomatic hospitalized']

s10 = ['population', 'populations', 'group', 'groups', 'demographics', 'demography', 'demographic', 'ethnic', 'ethnicities', 'ethnicity', 'racial', 'race', 'cultural', 'culture', 'gender']

s11 = ['public', 'health', 'public health']

s12 = ['measures', 'policy', 'proactive', 'assessment', 'partnership', 'priorities', 'practices', 'awareness']

s13 = ['control', 'management', 'mitigation']
# CountVectorizer with select vocabulary



def vocab_vectorizer(feature, vocab):



    '''

    Generates: document term matrix using a select vocabulary

    Args: 

        feature - dataframe column containing text to vectorize

        vocab - the specific list of terms from which to generate the matrix

    Returns: document term matrix as a dataframe 

    '''



    # generate a document term matrix

    CV1 = CountVectorizer(input="content", ngram_range=(1,2), vocabulary = vocab)

    DTM1 = CV1.fit_transform(corpus[feature])



    # add col names

    ColNames=CV1.get_feature_names()



    DF1 = pd.DataFrame(DTM1.toarray(), columns=ColNames)



    # add row names

    Dict1 = {}

    for i in range(0, len(corpus.paper_id)):

        Dict1[i] = corpus.paper_id[i]

    DF1 = DF1.rename(Dict1, axis='index')



    return DF1
term_sets = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13] 

dtm_sets = []



for s in tqdm(term_sets):

    dtm_sets.append(vocab_vectorizer(feature='text', vocab=s))
def word_count (field):

    

    tokens = field.split()

    field_length = len(tokens)

    

    return field_length
doc_lengths = pd.DataFrame( {'word_count' : corpus['text'].apply(word_count)} )
def length_normalize(dtm, doc_lengths):

    

    # Replace missing data with NA

    #dtm = dtm.fillna(0)

    

    for i in range(dtm.shape[1]):

    # Select column by index position using iloc[]

        columnSeriesObj = dtm.iloc[: , i]

        columnSeriesObj2 = columnSeriesObj / doc_lengths.word_count.values

        dtm.iloc[: , i] = columnSeriesObj2



    return dtm





dtm_sets_norm = []



for s in dtm_sets:

    dtm_sets_norm.append(length_normalize(s,doc_lengths))
dtm_sets_norm = []



for s in dtm_sets:

    dtm_sets_norm.append(length_normalize(s,doc_lengths))
def max_set_features(s):

    

    s['max'] = s.max(axis=1)

    

    max_array = np.array(s['max']).reshape(-1,1)

    

    min_max_scaler = preprocessing.MinMaxScaler()

    

    s['min_max'] = min_max_scaler.fit_transform(max_array)

    

    return s

dtm_sets_max = []



for s in dtm_sets_norm:

    dtm_sets_max.append(max_set_features(s))
dtm_dict = { 'paper_id' : corpus['paper_id']

            , 'dtm1' : dtm_sets_max[0]['min_max'].values

            ,'dtm2' : dtm_sets_max[1]['min_max'].values

            ,'dtm3' : dtm_sets_max[2]['min_max'].values

            ,'dtm4' : dtm_sets_max[3]['min_max'].values

            ,'dtm5' : dtm_sets_max[4]['min_max'].values

            ,'dtm6' : dtm_sets_max[5]['min_max'].values

            ,'dtm7' : dtm_sets_max[6]['min_max'].values

            ,'dtm8' : dtm_sets_max[7]['min_max'].values

            ,'dtm9' : dtm_sets_max[8]['min_max'].values

            ,'dtm10' : dtm_sets_max[9]['min_max'].values

            ,'dtm11' : dtm_sets_max[10]['min_max'].values

            ,'dtm12' : dtm_sets_max[11]['min_max'].values

            ,'dtm13' : dtm_sets_max[12]['min_max'].values

           }



dtm_columns=['paper_id', 'dtm1', 'dtm2','dtm3','dtm4', 'dtm5', 'dtm6','dtm7', 'dtm8','dtm9','dtm10', 'dtm11', 'dtm12','dtm13']



dtm_summary = pd.DataFrame(dtm_dict,columns=dtm_columns)



del dtm_dict, dtm_columns
task1a = pd.DataFrame({ 'paper_id' : corpus['paper_id'], 'score' : dtm_summary['dtm1'] * dtm_summary['dtm2']}

                      , columns=['paper_id','score'])



task1b = pd.DataFrame({ 'paper_id' : corpus['paper_id'], 'score' : dtm_summary['dtm1'] * dtm_summary['dtm3'] * dtm_summary['dtm4']}

                      , columns=['paper_id','score'])



task1c = pd.DataFrame({ 'paper_id' : corpus['paper_id'], 'score' : dtm_summary['dtm1'] * dtm_summary['dtm5']}

                      , columns=['paper_id','score'])



task1d = pd.DataFrame({ 'paper_id' : corpus['paper_id'], 'score' : dtm_summary['dtm1'] * dtm_summary['dtm6']}

                      , columns=['paper_id','score'])



task2 = pd.DataFrame({ 'paper_id' : corpus['paper_id'], 'score' : dtm_summary['dtm4'] * dtm_summary['dtm7']}

                      , columns=['paper_id','score'])



task3 = pd.DataFrame({ 'paper_id' : corpus['paper_id'], 'score' : dtm_summary['dtm8'] * dtm_summary['dtm9']}

                      , columns=['paper_id','score'])



task4 = pd.DataFrame({ 'paper_id' : corpus['paper_id'], 'score' : dtm_summary['dtm1'] * dtm_summary['dtm10']}

                      , columns=['paper_id','score'])



task5 = pd.DataFrame({ 'paper_id' : corpus['paper_id'], 'score' : dtm_summary['dtm11'] * dtm_summary['dtm12'] * dtm_summary['dtm13']}

                      , columns=['paper_id','score'])
task1a = task1a.sort_values(by='score',ascending=False)

task1b = task1b.sort_values(by='score',ascending=False)

task1c = task1c.sort_values(by='score',ascending=False)

task1d = task1d.sort_values(by='score',ascending=False)

task2 = task2.sort_values(by='score',ascending=False)

task3 = task3.sort_values(by='score',ascending=False)

task4 = task4.sort_values(by='score',ascending=False)

task5 = task5.sort_values(by='score',ascending=False)
task1a.head()
task1a_top100 = task1a.iloc[:100,:]

task1b_top100 = task1b.iloc[:100,:]

task1c_top100 = task1c.iloc[:100,:]

task1d_top100 = task1d.iloc[:100,:]

task2_top100 = task2.iloc[:100,:]

task3_top100 = task3.iloc[:100,:]

task4_top100 = task4.iloc[:100,:]

task5_top100 = task5.iloc[:100,:]
task1a_top100 = task1a_top100.merge(corpus, how='left', on='paper_id', sort=False )

task1b_top100 = task1b_top100.merge(corpus, how='left', on='paper_id', sort=False )

task1c_top100 = task1c_top100.merge(corpus, how='left', on='paper_id', sort=False )

task1d_top100 = task1d_top100.merge(corpus, how='left', on='paper_id', sort=False )

task2_top100 = task2_top100.merge(corpus, how='left', on='paper_id', sort=False )

task3_top100 = task3_top100.merge(corpus, how='left', on='paper_id', sort=False )

task4_top100 = task4_top100.merge(corpus, how='left', on='paper_id', sort=False )

task5_top100 = task5_top100.merge(corpus, how='left', on='paper_id', sort=False )
task1a_top100.head()
#eliminate any remaining numbers from text

task1a_top100['text'] = task1a_top100['text'].replace('\d+', 'NUM', regex=True)

task1b_top100['text'] = task1b_top100['text'].replace('\d+', 'NUM', regex=True)

task1c_top100['text'] = task1c_top100['text'].replace('\d+', 'NUM', regex=True)

task1d_top100['text'] = task1d_top100['text'].replace('\d+', 'NUM', regex=True)

task2_top100['text'] = task2_top100['text'].replace('\d+', 'NUM', regex=True)

task3_top100['text'] = task3_top100['text'].replace('\d+', 'NUM', regex=True)

task4_top100['text'] = task4_top100['text'].replace('\d+', 'NUM', regex=True)

task5_top100['text'] = task5_top100['text'].replace('\d+', 'NUM', regex=True)
# TfidfVectorizer with select vocabulary



def tfidf_vectorizer(df):



    '''

    Generates: tfidf document term matrix

    Args: dataframe

    Returns: document term matrix as a dataframe 

    '''



    

    # generate a document term matrix

    CV1 = TfidfVectorizer(input="content", ngram_range=(1,2), stop_words = 'english',  max_df=0.3, max_features=100)

    DTM1 = CV1.fit_transform(df['text'])



    # add col names

    ColNames=CV1.get_feature_names()



    DF1 = pd.DataFrame(DTM1.toarray(), columns=ColNames)



    # add row names

    Dict1 = {}

    for i in range(0, len(df.paper_id)):

        Dict1[i] = df.paper_id[i]

    DF1 = DF1.rename(Dict1, axis='index')



    return DF1

    
task1a_top100_tfidf = tfidf_vectorizer(task1a_top100)

task1b_top100_tfidf = tfidf_vectorizer(task1b_top100)

task1c_top100_tfidf = tfidf_vectorizer(task1c_top100)

task1d_top100_tfidf = tfidf_vectorizer(task1d_top100)

task2_top100_tfidf = tfidf_vectorizer(task1d_top100)

task3_top100_tfidf = tfidf_vectorizer(task1d_top100)

task4_top100_tfidf = tfidf_vectorizer(task1d_top100)

task5_top100_tfidf = tfidf_vectorizer(task1d_top100)
task1a_top100_tfidf.head()
task1b_top100_tfidf.head()
def tfidf_pca(df):

    

    pca = PCA(n_components=2).fit(df)

    pca_2d = pca.transform(df)



    for i in range(0, pca_2d.shape[0]):



        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',

            marker='+')

        

    return c1
task1a_tfidf_pca = tfidf_pca(task1a_top100_tfidf)
task1b_tfidf_pca = tfidf_pca(task1b_top100_tfidf)
task1c_tfidf_pca = tfidf_pca(task1c_top100_tfidf)
task1d_tfidf_pca = tfidf_pca(task1d_top100_tfidf)
def tfidf_pca_kmeans(df,num_clusters):

    

    pca = PCA(n_components=2).fit(df)

    pca_2d = pca.transform(df)



    task_tfidf_kmeans = KMeans(n_clusters=num_clusters, random_state=111)

    task_tfidf_kmeans.fit(df)



    c1 = plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=task_tfidf_kmeans.labels_)



    return c1, task_tfidf_kmeans
task1a_tfidf_plot, task1a_tfidf_pca_kmeans = tfidf_pca_kmeans(task1a_top100_tfidf, num_clusters=3)
task1b_tfidf_plot, task1b_tfidf_pca_kmeans = tfidf_pca_kmeans(task1b_top100_tfidf, num_clusters=3)
task1c_tfidf_plot, task1c_tfidf_pca_kmeans = tfidf_pca_kmeans(task1c_top100_tfidf, num_clusters=3)
task1d_tfidf_plot, task1d_tfidf_pca_kmeans = tfidf_pca_kmeans(task1d_top100_tfidf, num_clusters=3)
def tfidf_kmeans(df,num_clusters):



    task_tfidf_kmeans = KMeans(n_clusters=num_clusters, random_state=111)

    task_tfidf_kmeans.fit(df)

    

    task_tfidf_kmeans_labels = task_tfidf_kmeans.labels_

    task_tfidf_kmeans_df = pd.DataFrame([df.index,task_tfidf_kmeans_labels]).T

    

    task_tfidf_kmeans_df.columns = ['paper_id', 'cluster']



    return task_tfidf_kmeans_df
task1a_tfidf_pca_kmeans_df = tfidf_kmeans(task1a_top100_tfidf,3)

task1b_tfidf_pca_kmeans_df = tfidf_kmeans(task1b_top100_tfidf,3)

task1c_tfidf_pca_kmeans_df = tfidf_kmeans(task1c_top100_tfidf,3)

task1d_tfidf_pca_kmeans_df = tfidf_kmeans(task1d_top100_tfidf,3)

task2_tfidf_pca_kmeans_df = tfidf_kmeans(task2_top100_tfidf,3)

task3_tfidf_pca_kmeans_df = tfidf_kmeans(task3_top100_tfidf,3)

task4_tfidf_pca_kmeans_df = tfidf_kmeans(task4_top100_tfidf,3)

task5_tfidf_pca_kmeans_df = tfidf_kmeans(task5_top100_tfidf,3)
task1a_tfidf_pca_kmeans_df.info()
task1a_tfidf_cluster1 = task1a_tfidf_pca_kmeans_df.loc[task1a_tfidf_pca_kmeans_df['cluster']==0]
task1a_tfidf_cluster1[:10]
def prepare_output(task_df, task_tfidf_df, task_pca_kmeans_df):

    

    task_tfidf_df.reset_index(inplace=True)

    task_tfidf_df.rename(columns={'index':'paper_id'}, inplace=True)

    

    task_df = task_df.merge(task_pca_kmeans_df, how='left', on='paper_id', sort=False )

    task_df = task_df.merge(task_tfidf_df, how='left', on='paper_id', sort=False )

    

    return task_df
task1a_top100 = prepare_output(task1a_top100, task1a_top100_tfidf, task1a_tfidf_pca_kmeans_df)

#task1b_top100 = prepare_output(task1b_top100, task1b_top100_tfidf, task1b_tfidf_pca_kmeans_df)

task1c_top100 = prepare_output(task1c_top100, task1c_top100_tfidf, task1c_tfidf_pca_kmeans_df)

task1d_top100 = prepare_output(task1d_top100, task1d_top100_tfidf, task1d_tfidf_pca_kmeans_df)

task2_top100 = prepare_output(task2_top100, task2_top100_tfidf, task2_tfidf_pca_kmeans_df)

task3_top100 = prepare_output(task3_top100, task3_top100_tfidf, task3_tfidf_pca_kmeans_df)

task4_top100 = prepare_output(task4_top100, task4_top100_tfidf, task4_tfidf_pca_kmeans_df)

task5_top100 = prepare_output(task5_top100, task5_top100_tfidf, task5_tfidf_pca_kmeans_df)



task1a_top100.to_csv('task1a_top100.csv', sep=',')

#task1b_top100.to_csv('task1b_top100.csv', sep=',')

task1c_top100.to_csv('task1c_top100.csv', sep=',')

task1d_top100.to_csv('task1d_top100.csv', sep=',')

task2_top100.to_csv('task2_top100.csv', sep=',')

task3_top100.to_csv('task3_top100.csv', sep=',')

task4_top100.to_csv('task4_top100.csv', sep=',')

task5_top100.to_csv('task5_top100.csv', sep=',')
# task1b is run separately from the others because index column has different name

task1b_top100_tfidf.reset_index(inplace=True)

task1b_top100_tfidf.rename(columns={'level_0':'paper_id'}, inplace=True)

task1b_top100 = task1b_top100.merge(task1b_tfidf_pca_kmeans_df, how='left', on='paper_id', sort=False )

task1b_top100 = task1b_top100.merge(task1b_top100_tfidf, how='left', on='paper_id', sort=False )

task1b_top100.to_csv('task1b_top100.csv', sep=',')