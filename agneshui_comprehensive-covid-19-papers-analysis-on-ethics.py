# load the file with the top results

import numpy as np

import pandas as pd

top20 = pd.read_csv('../input/ethics/ethic_paper.csv')

pd.set_option('max_colwidth', 150)

top20[['cord_uid','title']].head(20)
# load all the required libraries

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')



# load the raw data

paper = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')
length_of_df = len(paper)

print(f'Before cleaning, there are {length_of_df} entries in the metadata.csv file.')
# cleaning pipline

''' 1. keeping the columns: cord_uid, title, abstract

    2. remove duplication in the title column, because they may have different cord_uid, 

       and not being recognized as duplicated entry

    3. remove NA in the abstract column, because the analysis will use the words from the 

       abstract column

    4. reset the index of the dataframe'''



paper.drop(columns=['sha','source_x','doi','pmcid','pubmed_id','license', 'publish_time',

                    'authors', 'journal','Microsoft Academic Paper ID',

                    'WHO #Covidence', 'has_pdf_parse','has_pmc_xml_parse',

                    'full_text_file', 'url'], 

           inplace = True)



paper.drop_duplicates(subset=['title'], keep = 'first', inplace = True)



paper.dropna(subset = ['abstract'], inplace = True)



paper.reset_index(drop=True)
length_of_df = len(paper)

print(f'After cleaning, there are {length_of_df} entries in dataframe, "paper".')
# need to install en_core_sci_lg, as it is not pre-installed on Kaggle.



!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
# import the natural language processing library and PhraseMatcher

import spacy

import en_core_sci_lg

from spacy.matcher import PhraseMatcher



# load large science tokenizer, tagger, parser NER and word vectors

nlp = en_core_sci_lg.load()



# define the matcher

matcher = PhraseMatcher(nlp.vocab)
# define the phrases for each sub-topics

moral_list = ['health care providers',

              'health care workers',

              'ethical principles',

              'professional moral duty',

              'duty to care',

              'obligation', 

              'moral',

              'duty',

              'decisions',

              'dilemmas']



education_list = ['education',

                  'access',

                  'capacity building',

                  'ethics']

               

measure_list = ['qualitative assessment',

                'public health measures',

                'secondary impacts',

                'prevention',

                'surgical masks',

                'social distancing',

                'school closures']

        

psy_list = ['psychological health',

            'immediate needs',

            'fear',

            'anxiety']



misinfo_list = ['stigma',

                'misinformation',

                'rumor',

                'social media']



covid_list = ['COVID-19',

              'novel coronavirus 2019',

              '2019 novel coronavirus',

              'new coronavirus pneumonia',

              'SARS-CoV-2',

              'coronavirus disease 2019'

             ]



# convert the phrase to a Doc object

phrase_pattern_moral = [nlp(text) for text in moral_list]

phrase_pattern_education = [nlp(text) for text in education_list]

phrase_pattern_measure = [nlp(text) for text in measure_list]

phrase_pattern_psy = [nlp(text) for text in psy_list]

phrase_pattern_misinfo = [nlp(text) for text in misinfo_list]

phrase_pattern_covid = [nlp(text) for text in covid_list]



# pass the phrase_pattern to the matcher

matcher.add('Moral', None, *phrase_pattern_moral)

matcher.add('Education', None, *phrase_pattern_education)

matcher.add('Measure', None, *phrase_pattern_measure)

matcher.add('Psychological', None, *phrase_pattern_psy)

matcher.add('Misinformation', None, *phrase_pattern_misinfo)

matcher.add('COVID', None, *phrase_pattern_covid)
def matchID_to_matchString(match_output):

    '''this function coverts the result of Phrasematcher, match ID into the match string.

       

       input: Phrasematcher output, list of number with match ID, the location of the match

       string in the text. The function will use the first entry of each item in the list, 

       and convert it back to the string.

       

       output: a list of unqiue match string, which are the sub-topics (repeated matches are removed)

    '''

    

    match_string_list = list()

    

    for i in range(0, len(match_output)):

        match_string = nlp.vocab.strings[match_output[i][0]]

        

        if match_string not in match_string_list:

            match_string_list.append(match_string)

            

    return match_string_list





def match_all_keywords(abstract_nlp, match_ouput):

    '''this function covert the match output to individual keywords.

       

       input: tokenized abstract and the match output

       output: a list of keywords found in the abstract

    '''

    

    keywords = list()

    

    for i in range(0, len(match_ouput)):

        start_id, start_end = match_ouput[i][1], match_ouput[i][2]

        keywords.append(abstract_nlp[start_id:start_end].text)

        

    return keywords





def find_unique_keywords(all_keywords):

    '''this function find the unique keywords from the all_keywords list

    

       input: a list of all keywords

       output: a list of unique keywords

    '''

    

    unique_list = list()

    

    for words in all_keywords:

        if words not in unique_list:

            unique_list.append(words)

            

    return unique_list



def one_hot_encoding(df, col_name, topics):

    '''this function is a one_hot_encoding process to convert a list of match_string or unique_keywords into

       columns with 0/1 with the column name representing the sub-topic or keywords.

       

       input: 

             df:the dataframe that the new columns will be added

             col_name: it can be the columns match_string if sub-topic is intented to be unpacked; or 

                       unique_keywords if indivdiual keywords is intented to be unpacked

             word_list:  a list of sub-topics in match_string or a list of unique_keywords,

                        it should be each entry of the column: 'match_string' or 'unique_keywords'

       ouput: the dataframe with new column

       

    '''

    def unpack_topic(word_list, topic):

        

        is_topic = list()

        

        for item in word_list:

            is_topic.append(topic in item)

        return sum(is_topic)



    for topic in topics:

        df[topic] = df[col_name].apply(lambda x: unpack_topic(x, topic))

    

    return df
# additional functions for analysis on individual keywords level



def count_keywords(unique_keywords):

    '''this function count the unique keywords of the whole dataframe'''

    

    keyword_count = dict()

    keywords_list = list()

    

    for row in unique_keywords:

        keywords_list.extend(row)

        

    for word in keywords_list:

        if word in keyword_count:

            keyword_count[word] += 1

        else:

            keyword_count[word] = 1



    return keyword_count



def remove_covid(count_list):

    for word in covid_list:

        count_list.pop(word)

    return count_list
# tokenizer the text in abstract

paper['abstract_nlp'] = paper['abstract'].apply(lambda x: nlp(x))
# match the phrases to the tokenized abstract

paper['match_output'] = paper['abstract_nlp'].apply(lambda x: matcher(x))



# convert the phrasematcher output into strings

paper['match_string'] = paper['match_output'].apply(matchID_to_matchString)



# find all the keywords/phrase in the string output of the phrasematcher

paper['all_keywords'] = paper.apply(lambda x: match_all_keywords(x['abstract_nlp'],x['match_output']), axis = 'columns')



# find the unique keywords, ie. remove duplications

paper['unique_keywords'] = paper['all_keywords'].apply(find_unique_keywords)



# count the number of topics and the number of unique keywords

paper['no_topics'] = paper['match_string'].apply(len)

paper['no_unique_keywords'] = paper['unique_keywords'].apply(len)



# remove unwanted columns

paper.drop(columns = ['abstract_nlp','match_output'], inplace = True)
# one-hot-encoding to covert the data in list to individual columns

sub_topics = ['Moral', 'Education', 'Measure', 'Psychological', 'Misinformation', 'COVID']

one_hot_encoding(paper, 'match_string', sub_topics)
# subset the papers related to COVID-19

covid_paper = paper.loc[paper['COVID'] > 0].copy()
# count number of unique keywords that is present (not counting keywords related to COVID-19)

covid_ethics_list = count_keywords(covid_paper['unique_keywords'])

covid_ethics_list = remove_covid(covid_ethics_list)

keywords_stat = pd.DataFrame.from_dict(covid_ethics_list, orient = 'index', columns = ['counts'])

keywords_stat = keywords_stat.sort_values(by = ['counts'], ascending = False)
# adding each keywords to the dataframe

keywords_list = keywords_stat.index

one_hot_encoding(covid_paper, 'unique_keywords', keywords_list)
# add a column with the number of ethic topics

covid_paper['ethic_topics'] = covid_paper['no_topics'].apply(lambda x: x-1)



# add a column with the number of ethic topics

covid_paper['no_ethics_keywords'] = covid_paper.apply(lambda x: sum(x[keywords_list]), axis = 1)



# sort the dataframe with the number of topic, and the number of unique keywords

covid_paper.sort_values(by = ['ethic_topics', 'no_ethics_keywords'], ascending = False, inplace = True)



# output the COVID-19 papers 

covid_paper.to_csv('covid_paper.csv')
# output to csv paper with COVID-19 and at least 1 ethic search phrase/keywords

ethic_paper = covid_paper.loc[covid_paper['no_ethics_keywords']> 1].copy()

ethic_paper.to_csv('ethic_paper.csv')
length_of_covid = len(covid_paper)

print(f'There are {length_of_covid} research papers related to COVID-19.')
length_of_ethic = sum(covid_paper['ethic_topics'] > 0)

print(f'There are {length_of_ethic} research papers related to COVID-19, and contain ethic-related keywords.')
covid_summary = covid_paper.groupby(by = ['COVID']).sum()



moral_no = covid_summary['Moral'][1]

edu_no = covid_summary['Education'][1]

measure_no = covid_summary['Measure'][1]

psychological_no = covid_summary['Psychological'][1]

misinfo_no = covid_summary['Misinformation'][1]



print(f'Within those, {moral_no} papers contain keywords in sub-topic: moral;')

print(f'              {edu_no} papers contain keywords in sub-topic: education;')

print(f'              {measure_no} papers contain keywords in sub-topic: measure;')

print(f'              {psychological_no} papers contain keywords in sub-topic: psychological health;')

print(f'              {misinfo_no} papers contain keywords in sub-topic: misinformation.')
# examine how many paper contain the search phrases/keywords

plt.rcParams['figure.figsize'] = [10, 8]

plt.figure()

plot1 = paper['no_topics'].hist(bins= 10)

plot1.set_xlabel('Number of topics')

plot1.set_ylabel('Number of research papers')
# examine how many paper has the increased topic and with COVID-19 



plt.rcParams['figure.figsize'] = [10, 8]

plt.figure()



paper_summary = paper.groupby(by = ['COVID']).sum()



plot2 = paper_summary[['Moral', 'Education', 'Measure', 

               'Misinformation','Psychological']].plot(kind = 'barh')

plot2.set_xlabel('Number of research papers')

plot2.set_ylabel('Contain COVID-19')
# examine how many paper has ethic topics within the COVID-19 papers

plt.rcParams['figure.figsize'] = [10, 8]

plt.figure()



covid_sum = covid_paper.groupby(['ethic_topics']).sum()

plot3 = covid_sum[['Moral', 'Education', 'Measure', 

           'Misinformation','Psychological']].plot(kind = 'bar')



plot3.set_xlabel('number of sub-topics')

plot3.set_ylabel('Number of research papers')
plt.rcParams['figure.figsize'] = [10, 8]

plt.figure()



plot4 = keywords_stat.plot(kind = 'barh',legend=False)

plot4.set_xlabel('Number of research papers')

plot4.set_ylabel('search phrases/keywords')
# output the results

def output_title(df, keyword):

    new_df = df.loc[df[keyword] == 1].copy()

    new_df.sort_values(by = ['no_ethics_keywords'], ascending = False, inplace = True)

    return new_df.filter(items = ['cord_uid', 'title','unique_keywords'])
pd.set_option('max_colwidth', 150)

broad_ethic_papers = covid_paper.loc[covid_paper['ethic_topics']> 1].copy()
print(f'There are {len(broad_ethic_papers)} papers contain serach phrases/keywords of more than 2 sub-topics.')

print('Within those papers are under the topics Moral, Education and Psychological health')
output_title(broad_ethic_papers, 'Moral')
output_title(broad_ethic_papers, 'Education')
output_title(broad_ethic_papers, 'Psychological')
# subset for papers with more than 1 unqiue keywords and only 1 sub-topic

ethic_subtopic = covid_paper.loc[(covid_paper['no_ethics_keywords']> 1) & (covid_paper['ethic_topics']<2)].copy()
output_title(ethic_subtopic, "Psychological")
output_title(ethic_subtopic, "Measure")
output_title(ethic_subtopic, "Misinformation")