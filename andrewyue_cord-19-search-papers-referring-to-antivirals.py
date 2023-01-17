import numpy as np

import pandas as pd

import os

import json

import glob

import matplotlib.pyplot as plt

%matplotlib inline
potential_antivirals = ['abacavir', 'abacavir / dolutegravir / Lamivudine', 'abacavir / Lamivudine', 'abacavir / Lamivudine / Zidovudine', 'Acyclovir', 'adefovir', 'Amprenavir', 'asunaprevir', 'Atazanavir', 'Atazanavir / cobicistat', 'Baloxavir marboxil', 'bictegravir / emtricitabine / tenofovir alafenamide', 'boceprevir', 'brivudine', 'Cidofovir', 'cobicistat / darunavir', 'cobicistat / darunavir / emtricitabine / tenofovir alafenamide', 'cobicistat / elvitegravir / emtricitabine / tenofovir alafenamide', 'cobicistat / elvitegravir / emtricitabine / tenofovir disoproxil','daclatasvir', 'darunavir', 'dasabuvir', 'dasabuvir / ombitasvir / paritaprevir / Ritonavir', 'Delavirdine', 'Didanosine', 'dolutegravir', 'dolutegravir / Lamivudine', 'dolutegravir / Rilpivirine', 'DORAVIRINE', 'DORAVIRINE / Lamivudine / tenofovir disoproxil', 'efavirenz', 'efavirenz / emtricitabine / tenofovir disoproxil', 'efavirenz / Lamivudine / tenofovir disoproxil', 'elbasvir', 'elbasvir / grazoprevir', 'elvitegravir', 'emtricitabine', 'emtricitabine / Rilpivirine / tenofovir alafenamide', 'emtricitabine / Rilpivirine / tenofovir disoproxil', 'emtricitabine / tenofovir alafenamide', 'emtricitabine / tenofovir disoproxil','enfuvirtide', 'entecavir', 'etravirine', 'famciclovir', 'fosamprenavir', 'Foscarnet', 'Ganciclovir', 'glecaprevir / pibrentasvir', 'grazoprevir', 'ibalizumab', 'Idoxuridine', 'Indinavir', 'Inosine Pranobex', 'Lamivudine', 'Lamivudine / Nevirapine / Stavudine', 'Lamivudine / Nevirapine / Zidovudine', 'Lamivudine / tenofovir disoproxil', 'Lamivudine / Zidovudine', 'ledipasvir / sofosbuvir', 'letermovir', 'lopinavir / Ritonavir', 'lysozyme', 'maraviroc', 'moroxydine', 'Nelfinavir', 'Nevirapine', 'ombitasvir / paritaprevir / Ritonavir', 'Oseltamivir', 'penciclovir', 'peramivir', 'raltegravir', 'Ribavirin', 'Rilpivirine', 'Rimantadine', 'Ritonavir', 'Saquinavir', 'simeprevir', 'sofosbuvir', 'sofosbuvir / velpatasvir', 'sofosbuvir / velpatasvir / voxilaprevir', 'Stavudine', 'Tecovirimat', 'telaprevir', 'telbivudine', 'tenofovir alafenamide', 'tenofovir disoproxil', 'tipranavir', 'tromantadine', 'valacyclovir', 'valganciclovir', 'Vidarabine', 'Zalcitabine', 'Zanamivir', 'Zidovudine'] + ['Amodiaquine', 'artemether', 'artemether / lumefantrine', 'artesunate', 'Chloroquine', 'halofantrine', 'Hydroxychloroquine', 'Mefloquine', 'Primaquine', 'Proguanil', 'Pyrimethamine', 'Quinine', 'tafenoquine'] + ['remdesivir', 'galidesivir', 'favipiravir']
antiviral_search_terms = []

for drug in potential_antivirals:

    if ' / ' in drug:

        antiviral_search_terms.extend([component.lower() for component in drug.split(' / ')])

    else:

        antiviral_search_terms.append(drug.lower())

antiviral_search_terms = list(set(antiviral_search_terms))
paper_lengths = []

for paper_path in glob.glob('/kaggle/input/CORD-19-research-challenge/2020-03-13/*/*/*.json'):

    with open(paper_path, 'r') as f:

        paper = json.loads(f.read())

    paper_text = '\n'.join(item['text'] for item in paper['body_text'])

    paper_lengths.append(len(paper_text))

_ = plt.hist(paper_lengths, bins=[i*1000 for i in range(100)])

plt.show()
papers_referring_to_antivirals = []

papers_searched = 0

papers_matching = 0

for paper_path in glob.glob('/kaggle/input/CORD-19-research-challenge/2020-03-13/*/*/*.json'):

    with open(paper_path, 'r') as f:

        paper = json.loads(f.read())

    paper_text = '\n'.join(item['text'] for item in paper['body_text']).lower()

    if any([term in paper_text for term in antiviral_search_terms]):

        papers_referring_to_antivirals.append(paper_path)

        papers_matching += 1

    papers_searched += 1

    if papers_searched % 1000 == 0:

        print(papers_searched, papers_matching)
for paper_path in papers_referring_to_antivirals:

    filename = paper_path.split('/')[-1].split('.')[0] + '.txt'

    with open(filename, 'w') as f:

        with open(paper_path, 'r') as g:

            paper = json.loads(g.read())

        f.write('\n'.join(item['text'] for item in paper['body_text']).lower())
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(input='filename', lowercase=True, vocabulary=antiviral_search_terms)
output = cv.fit_transform(glob.glob('*.txt'))

presence_of_term = np.where(output.toarray() > 0, True, False)
antiviral_df = pd.DataFrame(presence_of_term, columns=antiviral_search_terms)

antiviral_df['sha'] = [fname.split('.')[0] for fname in glob.glob('*.txt')]
index_df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

output_df = index_df.merge(antiviral_df, on='sha', how='inner')

output_df.index = output_df['sha']
sha_repeats_dict = {}

times_sha_repeats = []

for sha, row in output_df.iterrows():

    if sha in sha_repeats_dict:

        sha_repeats_dict[sha] += 1

    else:

        sha_repeats_dict[sha] = 0

    times_sha_repeats.append(sha_repeats_dict[sha])
output_df['times_sha_repeated'] = times_sha_repeats

output_df = output_df[output_df['times_sha_repeated'] == 0]
output_df[antiviral_search_terms].sum().sort_values(ascending=False)[:20]
output_df['drugs_mentioned'] = output_df[antiviral_search_terms].sum(axis=1)
output_df.sort_values('drugs_mentioned', ascending=False)[:20][['title', 'publish_time', 'doi', 'drugs_mentioned']]
import spacy

nlp = spacy.load("en_core_web_sm")

nlp.max_length = 2000000 # there's at least one paper that exceeds the default max_length of 1000000
def get_article_text(sha):

    with open(sha + '.txt', 'r') as f:

        return f.read()
relevant_sentences = []

i = 0

for sha, row in output_df.iterrows():

    if i % 100 == 0:

        print(i)

    i += 1

    relevant_sentences_in_paper = []

    drugs_in_paper = [drug for drug, drug_in_paper in row[antiviral_search_terms].iteritems() if drug_in_paper]

    doc = nlp(get_article_text(sha))

    for token in doc:

        if token.text in drugs_in_paper:

            relevant_sentences_in_paper.append(token.sent.text)

    relevant_sentences.append('\n'.join(list(set(relevant_sentences_in_paper))))

output_df['relevant_sentences'] = relevant_sentences
output_df.to_csv('antiviral_paper_table.csv', index=False)
recent_favipiravir_df = output_df[(output_df['favipiravir']) & (output_df['publish_time'].notnull()) & ((output_df['publish_time'].str.contains('2019')) | (output_df['publish_time'].str.contains('2020')))]
print('\n\n'.join(recent_favipiravir_df['relevant_sentences'].tolist()))
def search_article_subset(subset_df, search_terms):

    search_results = ''

    for sha, row in subset_df.iterrows():

        sentences_matching_search = []

        # sorry for the egregious list comprehension - each article's relevant sentences are newline separated sentences. we want to check each sentence against each search term.

        matching_sentences = [sentence for sentence in row['relevant_sentences'].split('\n') if any([st in sentence for st in search_terms])]

        if matching_sentences != []:

            sentences_matching_search.extend(matching_sentences)

        if sentences_matching_search != []:

            search_results += '** article sha: {}\n'.format(sha)

            search_results += '** article title: {}\n'.format(row['title'])

            search_results += '\n\n'.join(sentences_matching_search)

            search_results += '\n\n'

    return search_results
key_terms = ['effic', 'activ', 'effect', 'model']
print(search_article_subset(recent_favipiravir_df, key_terms))