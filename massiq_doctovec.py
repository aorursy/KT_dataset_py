import os

import pandas as pd

from nltk import tokenize

import json

import numpy as np

import json

import gensim

import warnings

warnings.filterwarnings("ignore")
combined = pd.DataFrame()

for file in os.listdir('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/'):

    if file.endswith('.csv'):

        df = pd.read_csv(f'/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/{file}')

        combined = pd.concat([combined, df], ignore_index=True)

        print(f'Total documents in {file}: ', len(df))



print('='*80)

combined['title'] = combined['title'].str.lower() 

combined['abstract'] = combined['abstract'].str.lower() 

before = len(combined)

print('Total documents in dataset: ', before)

combined.drop_duplicates(subset=['title', 'abstract'], inplace=True)

print('After removing duplicates based on title and abstract: ', len(combined))

combined.rename(columns={'title': 'combined_title'}, inplace=True)

print('Documents with same title and abstract: ', before - len(combined))
def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    

    return body
df_covid_19 = pd.read_csv(f'/kaggle/input/filtered-data/covid_19_2020.csv')

print('Total covid-19 papers: ', len(df_covid_19))

df_covid_19_sha = df_covid_19.dropna(subset=['sha'])

df_covid_19_nosha = df_covid_19[df_covid_19.sha.isnull()]



print('Papers with missing sha in metadata: ', len(df_covid_19_nosha))

df_covid_19_sha = df_covid_19_sha.merge(combined[['paper_id', 'text']], how="left", left_on = 'sha', right_on = 'paper_id')

print('Papers that have sha key in metadata but still not found in json files: ', len(df_covid_19_sha[~df_covid_19_sha['sha'].isin(combined['paper_id'])]))
print('No of Records where sha is not null and has_pdf_parse is False')

print(len(df_covid_19[(~df_covid_19.sha.isnull()) & (df_covid_19['has_pdf_parse']== False)]))

print('No of Records where sha is not null and full_text_file is null')

print(len(df_covid_19[(~df_covid_19.sha.isnull()) & (df_covid_19['full_text_file'].isnull())]))

print('No of Records where sha is not null and has_pmc_xml_parse is null')

print(len(df_covid_19[(~df_covid_19.sha.isnull()) & (df_covid_19['has_pmc_xml_parse'] == False)]))
missing_text_index = df_covid_19_sha[~df_covid_19_sha['sha'].isin(combined['paper_id'])].index.tolist()

not_found = []

import json

for index in missing_text_index:

        dir_name = df_covid_19_sha['full_text_file'].loc[index]

        filename = df_covid_19_sha['sha'].loc[index]

        filename = filename.split(';')

        

        for file in filename:

            try: 

                path = f'/kaggle/input/CORD-19-research-challenge/{dir_name}/{dir_name}/pdf_json/{filename[0]}.json'

                file = json.load(open(path, 'rb'))

                text = format_body(file['body_text'])

                df_covid_19_sha['text'].loc[index] = text

                break

            

            except:

                not_found.append(index)

                



print('No of records where sha is given but no corresponding article exists in json: ', len(set(not_found)))
print('Total records in df_covid_19_nosha: ', len(df_covid_19_nosha))

print('Records where full_text_file is also null: ', len(df_covid_19_nosha[df_covid_19_nosha['full_text_file'].isnull()]))

print('Records where full_text_file and pmcid are non null: ', len(df_covid_19_nosha[(~df_covid_19_nosha['full_text_file'].isnull()) & 

                                                                                      (~df_covid_19_nosha['pmcid'].isnull())]))
missing_text_index = df_covid_19_nosha[(~df_covid_19_nosha['full_text_file'].isnull())&(~df_covid_19_nosha['pmcid'].isnull())].index.tolist()

not_found = []

df_covid_19_nosha['text'] = np.nan

for index in missing_text_index:

        dir_name = df_covid_19_nosha['full_text_file'].loc[index]

        pmcid = df_covid_19_nosha['pmcid'].loc[index]

               

        try: 

            path = f'/kaggle/input/CORD-19-research-challenge/{dir_name}/{dir_name}/pmc_json/{pmcid}.xml.json'

            file = json.load(open(path, 'rb'))

            text = format_body(file['body_text'])

            df_covid_19_nosha['text'].loc[index] = text

            

            if pd.isnull(df_covid_19_nosha['abstract'].loc[index]):

                if 'abstract' in file.keys():

                    df_covid_19_nosha['abstract'].loc[index] = file['abstract']

                    

        except:

            print(f'{dir_name}/{dir_name}/pmc_json/{pmcid}.xml.json not found')

            not_found.append(f'{pmcid}.xml.json')



print('='*80)

print('No of records where pmcid is given but no corresponding article exists in json: ', len(set(not_found)))

print('Out of {} df_covid_19_nosha records, records where body text is missing: {}'.format(len(df_covid_19_nosha), len(df_covid_19_nosha[df_covid_19_nosha.text.isnull()])))
df_covid_full_text = pd.concat([df_covid_19_sha, df_covid_19_nosha], ignore_index=False)

print('Out of {} records of df_covid_full_text, records where body text is null: {}'.format(len(df_covid_full_text), len(df_covid_full_text[df_covid_full_text.text.isnull()])))

df_covid_full_text['text'].fillna(df_covid_full_text['abstract'], inplace=True)

print('After substituting abstract as body, records where body text is still missing: ',len(df_covid_full_text[df_covid_full_text.text.isnull()]))

df_covid_full_text.dropna(subset=['text'], inplace=True)

print('Total papers after dropping records where body text is null: ', len(df_covid_full_text))
def get_docs(df_covid, is_test=False):

    documents = []

    actual_documents = []

    if is_test:

        text = df_covid['text'].loc[0]

        documents.append((text.lower(), 'no_tag'))

    

    else:

        for row in range(0, len(df_covid)):

            text = df_covid['text'].loc[row]      

            text = text.split('\n\n')

            

            pub_time = df_covid['publish_time'].loc[row]

            authors = df_covid['authors'].loc[row]

            title = df_covid['title'].loc[row]

            

            par_no=1



            for par in text:

                par = par.lower()

                if len(par)>=300:

                    sentences = tokenize.sent_tokenize(par)

                    final_par = ''

                    for sentence in sentences:

                        if 'international license' not in sentence and 'copyright' not in sentence and 'https' not in sentence and 'doi' not in sentence:

                            final_par = ''.join([final_par, sentence, ' '])

                        else:

                            pass



                    if len(final_par)>200 and 'no reuse allowed' not in final_par:

                        tag = ''.join([str(row), '-', str(par_no)])

                        

                        documents.append([final_par, tag, pub_time, authors, title])

                        par_no+=1

                    else:

                        pass



    #                 if 'no reuse allowed' in final_par and len(final_par)<=200:

    #                     print(final_par)

    #                     print('='*70)

    #                     pass

                          

    return documents



def read_corpus(docs, tokens_only=False):

    for index, rec in enumerate(docs):

        doc = rec[0]

        tag = rec[1]

        tag = ''.join([tag, '_', str(index)])

        tokens = gensim.utils.simple_preprocess(doc)

        if tokens_only:

            yield tokens

        else:

            yield gensim.models.doc2vec.TaggedDocument(tokens, [tag])
df_covid_train = df_covid_full_text[['text', 'publish_time', 'authors', 'title']]

df_covid_train.reset_index(drop=True, inplace=True)

train_documents = get_docs(df_covid_train)

df_train = pd.DataFrame.from_records(train_documents, columns = ['document', 'tag', 'publish_time', 'authors', 'title'])

df_train.drop(columns=['document'], axis=1, inplace=True)

print('Train Documents: ', len(train_documents))

train_corpus = list(read_corpus(train_documents))
print(train_corpus[:2])
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
def get_answer(query, no_of_results):

    df_test = pd.DataFrame(data={'text':[query]})

    test_documents = get_docs(df_test, is_test=True)

    test_corpus = list(read_corpus(test_documents, tokens_only=True))



    inferred_vector = model.infer_vector(test_corpus[0])

    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))



    print('Test Document: «{}»\n'.format(' '.join(test_corpus[0])))

    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)



    results = [(f'TOP {i}', i) for i in range(0,no_of_results)]

    unique_papers = []

    answers = []

    for label, index in results:

        splits = sims[index][0].split('_')

        tag = splits[0]

        doc_index = int(splits[1])

        print(doc_index)

        unique_papers.append(int(splits[0].split('-')[0]))

        excerpt = ' '.join(map(str, train_documents[doc_index]))

        

        print(u'%s %s:\n%s\n' % (label, sims[index], excerpt))

        answers.append([tag, excerpt])

        

        print('='*80)

    

    df_excerpts = pd.DataFrame().from_records(answers, columns= ['tag', 'excerpt'])

    return df_excerpts

    #print(unique_papers)

    #print('Total unique papers: ', len(set(unique_papers)))
query = 'What is the risk of pregnancy complications in COVID-19 patients? What is the risk of COVID-19 in pregnant women? What is the risk for COVID-19 in neonates? What is the risk for secondary hospital-acquired infections among neonatal COVID-19 patients admitted to critical intensive care?'

df_output = get_answer(query, 15)

df_output = df_output.merge(df_train, how='left', on='tag')

df_output = df_output[['publish_time', 'authors', 'title', 'excerpt']]

df_output.to_csv('./pregnant_neonants.csv', index=False)
query = 'What public health measures should be taken at government level that could be effective for controling the spread of COVID-19? Also what precautionary measures should people use to avoid coming in contact with covid-19?'

df_output = get_answer(query, 15)

df_output = df_output.merge(df_train, how='left', on='tag')

df_output = df_output[['publish_time', 'authors', 'title', 'excerpt']]

df_output.to_csv('./public_health_mitigation.csv', index=False)
query = 'What are the Transmission dynamics of the covid-19? What is the basic reproductive number of covid-19? What is the incubation period of covid-19? What is the serial interval of covid-19? What are different modes of transmission of covid-19 and does environmental factors play role in the transmision?'

df_output = get_answer(query, 15)

df_output = df_output.merge(df_train, how='left', on='tag')

df_output = df_output[['publish_time', 'authors', 'title', 'excerpt']]

df_output.to_csv('./transmission_dynamics_etc.csv', index=False)
query = 'What do we know about the severity of disease among people of different age groups? Also what is the risk of fatality among symptomatic hospitalized patients and high-risk patient groups? Is covid-19 disease more severe in patients having some underlying disease? If so what are such diseases?'

df_output = get_answer(query, 15)

df_output = df_output.merge(df_train, how='left', on='tag')

df_output = df_output[['publish_time', 'authors', 'title', 'excerpt']]

df_output.to_csv('./severity_of_disease_etc.csv', index=False)
query = 'Is covid-19 more transmissible in case a person carries some co-existing respiratory and viral infections or any of the other co-morbidities?'

df_output = get_answer(query, 15)

df_output = df_output.merge(df_train, how='left', on='tag')

df_output = df_output[['publish_time', 'authors', 'title', 'excerpt']]

df_output.to_csv('./con_infections_etc.csv', index=False)


query = 'What are the economic impacts of covid-19 pandemic, what are different socio-economic and behavioral factors arised as a result of covid-19 that can affect economy? What is the difference between groups for risk for COVID-19 by education level? by income? by race and ethnicity? by contact with wildlife markets? by occupation? household size? for institutionalized vs. non-institutionalized populations (long-term hospitalizations, prisons)?'

df_output = get_answer(query, 15)

df_output = df_output.merge(df_train, how='left', on='tag')

df_output = df_output[['publish_time', 'authors', 'title', 'excerpt']]

df_output.to_csv('./economic_behavioral_factors.csv', index=False)