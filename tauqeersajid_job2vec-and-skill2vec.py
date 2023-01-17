import numpy as np

import pandas as pd

import json

import gzip

import seaborn as sns

import os

from gensim.models import Word2Vec

from sklearn.manifold import TSNE

%matplotlib inline



data_dir = '../input/linkedin-crawled-profiles-dataset'

profiles_path = os.path.join(data_dir, 'linkedin.json/linkedin.json')

print(os.listdir(data_dir))
import json as js

localities = []

industries = []

specialities = []

interests = []

occupations = []

companies = []

majors = []

institutionsList = []



for l in open(profiles_path):

    line = js.loads(l)

    if 'education' in line:

        institutionsList.append([exp['name'] for exp in line['education'] if 'name' in exp])

    if 'locality' in line:

        localities.append([line['locality']])

    if 'education' in line:

        majors.append([edu['major'] for edu in line['education'] if 'major' in edu])

    if 'industry' in line:

        industries.append([line['industry']])

    if 'specilities' in line:

        specialities.append([s.strip() for s in line['specilities'].split(',')])

    if 'interests' in line:

        interests.append([s.strip() for s in line['interests'].split(',')])

    if 'experience' in line:

        occupations.append([exp['title'] for exp in line['experience'] if 'title' in exp])

    if 'experience' in line:

        companies.append([exp['org'] for exp in line['experience'] if 'org' in exp])

    

print(institutionsList[:20],"\n")

print(localities[:20],"\n")

print(industries[:20],"\n")

print(specialities[:20],"\n")

print(interests[:20],"\n")

print(occupations[:20],"\n")

print(companies[:20],"\n")

print(majors[:20])
# model school2vec

school2vec = Word2Vec(institutionsList, min_count=10, iter=100, workers=32)

institutions = list(school2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab



print(institutions[:100]) ## print the first 100 vocabularies from the list
print(school2vec['Columbia University'])
# sanity check

print('Columbia University: ', school2vec.wv.most_similar(['Columbia University']))
# model localities2vec

localities2vec = Word2Vec(localities, min_count=10, iter=100, workers=32)

localities = list(localities2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab



print(localities[:100]) ## print the first 100 vocabularies from the list
print(localities2vec['United States'])
# sanity check

print('United States: ', localities2vec.wv.most_similar(['United States']))
# model industries2vec

industries2vec = Word2Vec(industries, min_count=10, iter=100, workers=32)

industries = list(industries2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab



print(industries[:100]) ## print the first 100 vocabularies from the list
print(industries2vec['Biotechnology'])
# sanity check

print('Biotechnology: ', industries2vec.wv.most_similar(['Biotechnology']))
# model specialities2vec

specialities2vec = Word2Vec(specialities, min_count=10, iter=100, workers=32)

specialities = list(specialities2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab



print(specialities[:100]) ## print the first 100 vocabularies from the list
print(specialities2vec['Internet Marketing'])
# sanity check

print('Internet Marketing: ', specialities2vec.wv.most_similar(['Internet Marketing']))
# model interests2vec

interests2vec = Word2Vec(interests, min_count=10, iter=100, workers=32)

interests = list(interests2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab



print(interests[:100]) ## print the first 100 vocabularies from the list
print(interests2vec['nanotechnology'])
# sanity check

print('nanotechnology: ', interests2vec.wv.most_similar(['nanotechnology']))
# model occupations2vec

occupations2vec = Word2Vec(occupations, min_count=10, iter=100, workers=32)

occupations = list(occupations2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab



print(occupations[:100]) ## print the first 100 vocabularies from the list
print(occupations2vec['Senior Scientist'])
# sanity check

print('Senior Scientist: ', occupations2vec.wv.most_similar(['Senior Scientist']))
# model companies2vec

companies2vec = Word2Vec(companies, min_count=10, iter=100, workers=32)

companies = list(companies2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab



print(companies[:100]) ## print the first 100 vocabularies from the list
print(companies2vec['Albert Einstein Medical Center'])
# sanity check

print('Albert Einstein Medical Center: ', companies2vec.wv.most_similar(['Albert Einstein Medical Center']))
# model majors2vec

majors2vec = Word2Vec(majors, min_count=10, iter=100, workers=32)

majors = list(majors2vec.wv.vocab) # getting the vocabulary of the model using wv.vocab



print(majors[:100]) ## print the first 100 vocabularies from the list
print(majors2vec['Computer Science'])
# sanity check

print('Computer Science: ', majors2vec.wv.most_similar(['Computer Science']))