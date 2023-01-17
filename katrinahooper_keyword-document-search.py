import numpy as np

import pandas as pd

from glob import glob

import json

# from googletrans import Translator

# kaggle does have this

import subprocess

import sys

import gensim.models as gm

from gensim.utils import simple_preprocess

import pickle

from nltk import SnowballStemmer

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

pd.options.display.max_colwidth=40
# ROOT_PATH = '/kaggle/input/CORD-19-research-challenge/'



# full_text_articles = glob(ROOT_PATH + '*/*/*.json')

# # print(full_text_articles[:10])



# articles_df = {"sha":[], "article_title":[], "article_authors": [], "location":[], "article_abstract":[], "ref_titles":[], "all_text":[]}



# dic = json.load(open(full_text_articles[0]))

# print(dic)



# count = 0



# for path in full_text_articles:

#     with open(path) as file:

#         article = json.load(file)

#         all_text = ""

#         articles_df["sha"].append(article["paper_id"])

#         try:

#             articles_df["article_title"].append(article['metadata']["title"])

#             all_text += article['metadata']["title"]

#         except:

#             articles_df["article_title"].append("")  

#         authors_dict = article["metadata"]["authors"]

#         authors = ''

#         location = ''



#         for author in authors_dict:

#             try:

#                 name = author["last"] + ", " + author["first"]

#             except:

#                 name = ""

#             authors += name + "; "



#             try:

#                 loc = author['affiliation']['location']['country'] + ':' + author['affiliation']['location']['settlement'] + ';'

#                 all_text += " " + author['affiliation']['location']['settlement'] + ' ' + author['affiliation']['location']['country']

#             except:

#                 loc=''



#             if  loc not in location and loc != '':

#                     location += loc

#         articles_df["article_authors"].append(authors)

#         articles_df['location'].append(location)



#         try:

#             articles_df["article_abstract"].append(article["abstract"]["text"])

#             all_text += ' ' + article["abstract"]["text"]

#         except:

#             articles_df["article_abstract"].append('')

            

        

#         text = ''

#         try:

#             for paragraph in article['body_text']:

#                 text += " " + paragraph['text'] 

#         finally:

#             all_text +=  ' ' + text

        

#         ref_titles = []

#         try:

#             for ref in article['bib_entries']:

#                 ref_titles.append(article['bib_entries'][ref]['title'])

#                 all_text += ' ' + article['bib_entries'][ref]['title']

#         finally:

#             articles_df['ref_titles'].append(ref_titles)

#         articles_df["all_text"].append(all_text)

#     count += 1

#     print("atricle {} has be processed".format(count))



# articles_df = pd.DataFrame(data=articles_df)



# metadata = pd.read_csv(ROOT_PATH + 'metadata.csv', usecols=['sha', 'title', 'doi', 'abstract', 'authors', 'journal'])



# new = pd.merge(metadata, articles_df, on="sha", how="outer")



# new.to_pickle("merged_data.pkl")

# def translate_text(title, abstract, all_text):

#     translator = Translator(service_urls=['translate.google.ca'])

#     lang = 'en'

#     try:

#         if title != '':

#             lang = translator.detect(title).lang

#         else:

#             raise TypeError

#     except TypeError:

#         try:

#             if abstract != '':

#                 try:

#                     lang = translator.detect(abstract[:200]).lang

#                 except:

#                     pass

#             else:

#                 raise TypeError

#         except:

#             try:

#                 lang = translator.detect(all_text[:200]).lang

#             except:

#                 pass

#     print(lang)



#     if lang != 'en':



#         try:

#             title = translator.translate(title).text

#         except:

#             pass

#         try:

#             abstract = translator.translate(abstract).text

#         except:

#             pass

#         try:

#             title = translator.translate(all_text).text

#         except:

#             pass

#     return(title, abstract, all_text)





# data = pd.read_pickle('kaggle/output/merged_data.pkl')



# temp_df = {'sha':[], 'doi':[], 'title':[], 'authors':[], 'abstract':[], 'journal':[], 'location':[], 'all_text':[]}



# count = 0

# try:

#     while count < len(data):



#         if not (pd.isnull(data['title'][count]) and 

#             pd.isnull(data['article_title'][count]) and 

#             (pd.isnull(data['abstract'][count] or 

#             data['abstract'][count] != 'Unknown'))):



#             if pd.isnull(data['sha'][count]):



#                 all_text = ''



#                 if not pd.isnull(data['title'][count]):

#                     title = data['title'][count]

#                     try:

#                         all_text += data['title'][count]

#                     except:

#                         print("could not add title to all_text")

#                         print(type(data['title'][count]))

#                         title = data['article_title'][count]

#                 else:

#                     title = data['article_title'][count]

#                     all_text = data['all_text'][count]

                

#                 if not pd.isnull(data['authors'][count]):

#                     authors = data['authors'][count]

#                 else:

#                     authors = data['article_authors'][count]

                

#                 if not pd.isnull(data['abstract'][count]):

#                     abstract = data['abstract'][count]

#                     try:

#                         all_text += ' ' + data['abstract'][count]

#                     except:

#                         abstract = data['article_abstract'][count]

#                         print("could not add abstract to all_text")

#                         print(type(data['abstract'][count]))

#                 else:

#                     abstract = data['article_abstract'][count]

                    

#             else:

#                 all_text = data['all_text'][count]

                

#                 if data['article_title'][count] == '' and not pd.isnull(data['title'][count]):

                    

#                     all_text += ' ' + data['title'][count]

                

#                 if data['article_abstract'][count] == '' and not pd.isnull(data['abstract'][count]):

                    

#                     all_text += ' ' + data['abstract'][count]

                

#                 if data['article_title'][count] != '':

#                     title = data['article_title'][count]

#                 else:

#                     title = data['title'][count]



#                 if data['article_authors'][count] != '':

#                     authors = data['article_authors'][count]

#                 else:

#                     authors = data['authors'][count]



#                 if data['article_abstract'][count] != '':

#                     abstract = data['article_abstract'][count]

#                 else:

#                     abstract = data['abstract'][count]

        

#         try:

#             title, abstract, all_text = translate_text(title, 

#             abstract, all_text)

        

#         except:

#             i = 0

#             while True and i < 5:

#                 try:

#                     process = subprocess.call(['bash', 'ip_changer.bash'])

#                     title, abstract, all_text = translate_text(title, abstract, all_text)

#                     break

#                 except:

#                     print('Error is:', sys.exc_info()[0])

#                     print('refreshing vpn')

#                     i += 1

#             pass



#         temp_df['sha'].append(data['sha'][count])

#         temp_df['doi'].append(data['doi'][count])

#         temp_df['title'].append(title)

#         temp_df['authors'].append(authors)

#         temp_df['abstract'].append(abstract)

#         temp_df['journal'].append(data['journal'][count])

#         temp_df['location'].append(data['location'][count])

#         temp_df['all_text'].append(all_text)



#         count += 1

#         print('{}/{}'.format(count, len(data)))

# except:

#     print('could not complete error was:', sys.exc_info[0])

#     pass



# for i in temp_df:

#     print(i, len(temp_df[i]))



# refined_df = pd.DataFrame(temp_df)

# refined_df.to_pickle('translated_df.pkl')

# stemmer = SnowballStemmer('english')

# problems = []



# with open('kaggle/output/translated_df.pkl', 'rb') as df:

#     translated = pd.read_pickle(df)

#     pbar = tqdm(total=len(translated))

#     count = 0

#     try:

#         for text in translated['all_text']:



#             try:

#                 text = simple_preprocess(text)

#                 text = ' '.join([stemmer.stem(word) for word in text]).strip(' ') + '\n'

                

#                 with open('preprocessed_text.cor', 'a+') as f:

#                     f.write(text)

                

#             except:

#                 problems.append(count)

#                 print(translated.loc[count])

#             count += 1

#             pbar.update(1)

#     except:

#         pass

#     pbar.close()

# translated = pd.read_pickle('kaggle/output/translated_df.pkl').copy()

# translated = translated.drop(problems)

# translated.to_pickle('refined.pkl')



# model = gm.word2vec.Word2Vec(workers=6, min_count=3)

# print('model initalized')

# model.build_vocab(corpus_file='preprocessed_text.cor')

# print('vocab built')

# model.train(corpus_file='preprocessed_text.cor', total_words=model.corpus_count, epochs=model.iter)

# print('model trained')

# model.save('wv.model')


# model = gm.Word2Vec.load('kaggle/output/wv.model')

# articles_df = pd.read_pickle('kaggle/output/refined.pkl')['sha']



# count = 0



# with open('kaggle/output/preprocessed_text.cor', 'r') as docs:

#     while count < len(articles_df):

#         doc = docs.readline().strip('\n').split(' ')

#         vec = np.zeros(100)

#         for word in doc:

#             try:

#                 vec += model.wv[word]

#             except:

#                 pass

#         vec /= len(doc)

#         with open('doc_vecs.pkl', 'ab') as f:

#             pickle.dump(vec, f)

#         count += 1

#         print('{}/44970'.format(count))
stemmer = SnowballStemmer('english')

# model = gm.Word2Vec.load('kaggle/output/wv.model')

# quick start version:

model = gm.Word2Vec.load('../input/quickstart-doc-search/wv.model')



def get_topic_vec(topic):

    topic = simple_preprocess(topic)

    topic = [stemmer.stem(word) for word in topic]

    if len(topic) != 1:

        topic_vec = np.zeros(100)

        for word in topic:

            try:

                topic_vec += model.wv[word]

            except KeyError:

                pass

        topic_vec /= len(topic)

    else:

        try:

            topic_vec = model.wv[topic[0]]

        except KeyError:

            topic = input('That is not a valid keyword choice.\nPlease enter one or more keywords: ')

            topic_vec = get_topic_vec(topic)

    return topic_vec



def get_articles(topic):



    # topic = input('Articles can be searched for based on one or more keywords.\nWords must be separated, do not click "Enter" between keyword entries.\n(All article print outs are in english and have been translated using google translate)\nInput the keyword(s) for your search: ')

    topic_vec = get_topic_vec(topic)



    read = True

    count = 0

    # with open('..'/output/doc_vecs.pkl', 'rb') as vecs:



    with open('../input/quickstart-doc-search/doc_vecs.pkl', 'rb') as vecs:

    #     pbar = tqdm(total=len(pd.read_pickle('../output/refined.pkl')))

        pbar = tqdm(total=len(pd.read_pickle('../input/quickstart-doc-search/refined.pkl')))

        cos_sim = []

        while read == True:

            try:

                vec = pickle.load(vecs)

                sim = cosine_similarity([topic_vec], [vec])[0][0]

                if len(cos_sim) <= 10:

                    cos_sim.append((sim, count))

                else:

                    if sim > min(cos_sim)[0]:

                        cos_sim.remove(min(cos_sim))

                        cos_sim.append((sim, count))



            except EOFError:

                read = False

            pbar.update(1)

            count += 1

        pbar.close()



    # article_df = pd.read_pickle('../output/refined.pkl')

    article_df = pd.read_pickle('../input/quickstart-doc-search/refined.pkl')

    aritcle_df = article_df.reset_index(drop=True, inplace=True)

    # needed to reset index here as so no memory error is caused

    articles = article_df.loc[[i[1] for i in cos_sim],['doi', 'title', 'authors', 'abstract']]

    articles.insert(loc=0, column='score', value=[i[0] for i in cos_sim])



    print(articles.sort_values(by=['score'], ascending=False).to_string(index=False))

task1 = "transmission, incubation, environmental stability"

get_articles(task1)
task2 = "COVID-19 risk factors"

get_articles(task2)
task3 = "virus genetics, origin, evolution"

get_articles(task3)
task4 = "vaccines, therapeutics"

get_articles(task4)
task5 = "medical care coronavirus"

get_articles(task5)
task7 = "diagnostics, surveillance"

get_articles(task7)
task8 = "ethical and social science considerations"

get_articles(task8)
task9 = "information sharing and inter-sectoral collaboration"

get_articles(task9)