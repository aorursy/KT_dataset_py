%%capture 

!pip install pandas==1.0.3

!pip install spacy_langdetect

!pip install nltk

!pip install scispacy

!pip install kaggle

!python -m spacy download en_core_web_lg
# Downloading all datasets



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_key = user_secrets.get_secret("key")



f = open("kaggle.json", "w")

f.write('{"username":"pranjalya","key":"'+secret_key+'"}')

f.close()



!mkdir ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download allen-institute-for-ai/CORD-19-research-challenge -f metadata.csv

!unzip metadata.csv.zip

!rm -rf metadata.csv.zip



PATH = '../input/coronawhy/v6_text/v6_text/'
# Importing standard libraries

import os

import numpy as np

import pandas as pd

import nltk

nltk.download('punkt')

import warnings

warnings.filterwarnings('ignore')
def find_ngrams(dataframe, columnToSearch, keywords):

    '''

    Input : Complete Dataframe, Column to search keywords in, Keywords to search for

    Returns : Reduced dataframe which contains those keywords in given column

    '''

    df_w_ngrams = dataframe[dataframe[columnToSearch].str.contains('|'.join(keywords), case=False) == True]

    return df_w_ngrams



def keywordcounter(sentences, keywords_list):

    '''

    Input : List of sentences, List of keywords

    Returns : Keywords present in sentences, Total count of all keywords present in Input

    '''

    keyword = {}

    sent = " ".join(sentences)

    for pol in keywords_list:

        counter = sent.lower().count(pol)

        if (counter > 0):

            keyword[pol] = counter

    return list(keyword.keys()), sum(keyword.values())



def aggregation(item, keyWordList, RiskFactor):

    '''

    Input : Dataframe of sentences of a paper

    Return : Datframe in Standard Output format

    '''

    dfo = {}

    

    dfo['Risk Factor'] = RiskFactor

    dfo['Title'] = item['title'].iloc[0]

    dfo['Keyword/Ngram'], dfo['No of keyword occurence in Paper'] = keywordcounter(item['sentence'].tolist() + [item['abstract'].iloc[0]], keyWordList)

    dfo['paper_id'] = item['paper_id'].iloc[0]

    

    if (pd.isnull(item['url'].iloc[0])==False):

        dfo['URL'] = item['url'].iloc[0]

    else:

        dfo['URL'] = ''



    dfo['Sentences'] = item[item['section']=='results']['sentence'].tolist()

    

    if (item['authors'].iloc[0].isnull().any()==False):

        dfo['Authors'] = item['authors'].iloc[0].tolist()

    else:

         dfo['Authors'] = ''

        

    dfo['Correlation'] = item['causality_type'].iloc[0]

    dfo['Design Methodology'] = item['methodology'].iloc[0]

    dfo['Sample Size'] = item['sample_size'].iloc[0]

    dfo['Coronavirus'] = item['coronavirus'].iloc[0]

    dfo['Fatality'] = item['fatality'].iloc[0]

    dfo['TAXON'] =item['TAXON'].iloc[0]

    

    return dfo
def extract_features(ngramDf, allSentdataFrame):

    # extracting methodology

    methods_list = ['regression','OLS','logistic','time series','model','modelling','simulation','forecast','forecasting']

    methodology = find_ngrams(allSentdataFrame, 'sentence', methods_list)



    # extracting sample size

    sample_size_list = ['population size','sample size','number of samples','number of observations', 'number of subjects']

    sample_size = find_ngrams(allSentdataFrame, 'sentence', sample_size_list)



    # extracting nature of correlation

    causal_list =['statistically significant','statistical significance','correlation','positively correlated','negatively correlated','correlated','p value','p-value','chi square','chi-square','confidence interval','CI','odds ratio','OR','coefficient']



    causality_type = find_ngrams(allSentdataFrame, 'sentence', causal_list)



    # extracting coronavirus related sentence

    coronavirus_list = ['severe acute respiratory syndrome','sars-cov','sars-like','middle east respiratory syndrome','mers-cov','mers-like','covid-19','sars-cov-2','2019-ncov','sars-2','sarscov-2','novel coronavirus','corona virus','coronaviruses','sars','mers','covid19','covid 19']



    coronavirus = find_ngrams(allSentdataFrame, 'sentence', coronavirus_list)



    # extracting outcome

    disease_stage_list = ['lethal', 'morbid',"death", "fatality", "mortality","lethal", "lethality", "morbidity"]



    fatality = find_ngrams(allSentdataFrame, 'sentence', disease_stage_list)



    df_list = [methodology,sample_size,causality_type,coronavirus,fatality]

    df_list_name = ['methodology','sample_size','causality_type','coronavirus','fatality']

    i=0

    for one_df in df_list:

        one_df.rename(columns={'sentence':df_list_name[i]},inplace=True)

        grouped_one_df = one_df.groupby('paper_id')[df_list_name[i]].sum()

        ngramDf = pd.merge(ngramDf,grouped_one_df,on='paper_id',how='left')

        i=i+1

    return ngramDf
def get_relevant_papers(df, ngrams, risk):

    '''

    Input : 

        df -> Dataframe containing sentences from papers with metadata

        ngrams -> Dictionary with keys as Risk Factors

        risk -> The risk factor to be searched

    Returns : Dataframe containing papers in Output format

    '''

    

    # Extracting relevant papers in seperate dataframes from Result section and Abstract

    result_df = find_ngrams(df, 'sentence', ngrams[risk])

    abstract_df = find_ngrams(df, 'abstract', ngrams[risk])



    print("There are {} sentences containing keywords/ngrams in Result section for {}".format(len(result_df), risk))

    print("There are {} sentences containing keywords/ngrams in Abstract section for {}.".format(len(abstract_df), risk))



    

    # Merging the result section and abstract sentences into single dataframe

    df_r = pd.concat([result_df, abstract_df])

    df_r = df_r.loc[df_r.astype(str).drop_duplicates().index]



    print("Total unique papers in Result section : {}".format(result_df['paper_id'].nunique()))

    print("Total unique papers in Abstract section : {}".format(abstract_df['paper_id'].nunique()))

    print("Total unique papers in total : {}".format(df_r['paper_id'].nunique()))



    

    # Getting all sentences from papers containing the keywords in result section for feature extraction

    df_body_all_sentence = pd.merge(df[['paper_id','sentence']], result_df['paper_id'], on='paper_id', how='right')

    df_body_all_sentence.rename(columns={'sentence_x':'all_sentences','sentence_y':'ngram_sentence'}, inplace=True)



    df_abstract_all_sentence = pd.merge(df[['paper_id','abstract']], abstract_df['paper_id'], on='paper_id', how='right')

    df_abstract_all_sentence.rename(columns={'abstract_x':'all_sentences','abstract_y':'ngram_sentence'}, inplace=True)



    

    # Merging these sentences in single dataframe

    df_all_sentences = pd.concat([df_body_all_sentence, df_abstract_all_sentence])

    df_all_sentences = df_all_sentences.loc[df_all_sentences.astype(str).drop_duplicates().index]



    print("Total unique papers in combined section : {}".format(df_all_sentences['paper_id'].nunique()))



    

    # Extracting features from these sentences

    df_real = extract_features(df_r, df_all_sentences)

    df_real = df_real[['paper_id','language', 'section', 'sentence', 'lemma', 'UMLS', 'sentence_id', 'publish_time', 'authors', 'methodology','sample_size', 'causality_type','coronavirus','fatality','title','abstract','publish_time','authors', 'url', 'TAXON']]





    # Preparing the output in format

    grouped = df_real.groupby('paper_id')

    df_output = pd.DataFrame(columns=['Risk Factor', 'Title','Keyword/Ngram', 'No of keyword occurence in Paper', 'paper_id', 'URL', 'Sentences', 'Authors', 'Correlation', 'Design Methodology', 'Sample Size','Coronavirus','Fatality','TAXON'])



    for key, item in grouped:

        df_output = pd.concat([df_output, pd.DataFrame([aggregation(item, ngrams[risk], risk)])])



    df_output = df_output.reset_index()



    print("There are {} papers for Risk Factor : {}\n\n".format(len(df_output), risk))



    # Cleaning some memory

    del df_output['index']

    del df_r

    del df_real

    del df_all_sentences



    return df_output
# Risk Factors list



riskfactors = ['pollution', 'population density', 'humidity', 'age', 'temperature', 'heart risks']
# N-grams list for each risk factor as dictionary



ngrams = {}



ngrams['population density'] = ['high density areas','high density countries', 'population densities', 'density of population', 'sparsely populated','densely populated', 'density of the population','dense population', 'populated areas','densely inhabited','housing density', 'densely-populated','concentration of people','population pressure','population studies','populated regions', 'populous','high population densities','residential densities']



ngrams['pollution'] = ['air pollution and', 'indoor air pollutants', 'indoor air pollution', 'household air pollution', 'air pollution is', 'between air pollution', 'of air pollution', 'particulate air pollution', 'pollution and the', 'air pollutant data', 'water pollution']



ngrams['humidity'] = ['humidity','monsoon','rainy','vapour','rainfall']



ngrams['heart risks'] = ['congestive heart failure', 'complete atrioventricular block','myocardial ischemia', 'rheumatic heart disease', 'junctional premature complex','pulmonary heart disease', 'myocardial disease', 'sick sinus syndrome', 'hypertensive disorder', 'cardiac arrhythmia', 'supraventricular tachycardia','heart disease', 'cardiac arrest', 'supraventricular premature beats','ventricular premature complex', 'endocardial fibroelastosis','primary pulmonary hypertension', 'mitral valve regurgitation', 'heart failure', 'hypertensive renal disease', 'pulmonic valve stenosis', 'left heart failure', 'primary dilated cardiomyopathy', 'ischemic chest pain']

                      

ngrams['temperature'] = ['heated climate', 'cold temperatures', 'hot weather', 'cold weather', 'tropical climate', 'tropical weather', 'temperate','tropic','sunlight', 'summer','winter','spring','autumn','weather','in the season of','climate', 'local temperature']



ngrams['age'] =  ['persons older than', 'patients older than', 'patients not younger', 'patients not younger', 'above 65 years', 'over 65 years', '65 years old', 'over 65 years','above 60 years', 'over 60 years', '60 years old', 'over 60 years','above 70 years', 'over 70 years', '70 years old', 'over 70 years', 'among the elderly', 'among the aged','60 years and over', '65 years and over', 'aging population', 'older age group','circa 65 years']

# Load metadata and keep only the required columns



metadata = pd.read_csv('metadata.csv')

metadata.rename(columns={'sha':'paper_id'}, inplace = True)

metadata = metadata[['paper_id', 'title', 'abstract', 'publish_time', 'authors', 'url']]

metadata['paper_id'] = metadata['paper_id'].astype("str")

metadata['title'] = metadata['title'].fillna('')

metadata['abstract'] = metadata['abstract'].fillna('')
# Then : Loading result section



firstpass = True



for pkl in os.listdir(PATH):

    df = pd.read_pickle(PATH+pkl, compression='gzip')

    if(firstpass):

        v7_data = pd.read_json('../input/coronawhy/v7_text.json')

        df_result = pd.concat([v7_data[v7_data['section']=='results'], df[df['section']=='results']])

        firstpass = False

    else:

        df_result = pd.concat([df_result, df[df['section']=='results']])



df_result['paper_id'] = df_result['paper_id'].astype("str")



print("No of unique papers in result section : ", df_result['paper_id'].nunique(), " out of ", len(df_result), " rows in dataframe")

print("There are metadata present for ", metadata['paper_id'].nunique(), " papers.")
df = df_result.merge(metadata, how='inner', on='paper_id')



df['paper_id'] = df['paper_id'].astype("str")

df['title'] = df['title'].fillna('')

df['abstract'] = df['abstract'].fillna('')



print("There are ", df['paper_id'].nunique(), " papers available in both metadata and papers extracted.")
# Helper code to search for keywords in a better manner



rx = r"\.(?=\D)"

df['sentence'] = df['sentence'].str.replace(rx,' . ')

df['sentence'] = df['sentence'].str.replace(',',' , ')

df['abstract'] = df['abstract'].str.replace(rx,' . ')

df['abstract'] = df['abstract'].str.replace(',',' , ')
relevant_papers = {}



for risk in riskfactors:

    relevant_papers[risk] = get_relevant_papers(df, ngrams, risk)

    relevant_papers[risk] = relevant_papers[risk].sort_values(['Coronavirus', 'No of keyword occurence in Paper'], ascending=[False, False]).reset_index()

    del relevant_papers[risk]['index']

    relevant_papers[risk].to_csv('{}.csv'.format(risk))
relevant_papers['age'].head(10)
relevant_papers['pollution'].head(10)
relevant_papers['population density'].head(10)
relevant_papers['humidity'].head(10)
relevant_papers['heart risks'].head(10)
relevant_papers['temperature'].head(10)