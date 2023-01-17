import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt
#List to save contexts

contexts = []
'''

import json

import spacy



print('It\'s running!\n')



nlp = spacy.load('en')

vocab = ['transmission', 'transmitted', 'transmitting', 'incubation', 'incubate', 'incubated', 'environmental', 'stability', 'health status', 'asymptomatic', 'physical science', 'distribution', 'hydrophilic', 'phobic', 'environment', 'decontamination', 'nasal', 'fecal', 'model', 'phenotypic', 'phenotype', 'immunity', 'protective', 'protection']



paths = open('/kaggle/input/file-paths/file_paths.txt', 'r')

count = 0

for p in paths:

    

    if count >=-1:

        

        text = ""

        try:

            data = open(p[:-1]).read()

        except:

            continue

        

        data_dic = json.loads(data)

        try:

            title = data_dic['metadata']['title']

        except:

            title = ""

        try:

            abstract = data_dic['abstract'][0]['text']

        except:

            abstract = ""

        

        body_structure = data_dic['body_text']

        body = ""

        for i in range(0,len(body_structure)):

            body+= body_structure[i]['text'] + " "

        

        text += title + ". " + abstract + " " + body + "\n"



        try:

            doc = nlp(text)

        except:

            print('Article skipped due to the number of tokens, more than 1 million!')

            continue



        for token in doc:

            if str(token) in vocab:

                contexts.append(str(token.sent)) #Contexts, that is full sentences containing keywords,

                                                 #extracted here.

    count+=1

    if count == 29335:

        break

paths.close()

print('Finished!')

'''
import re



if len(contexts)==0:

    ctx = open('/kaggle/input/all-contexts/Contexts_Kaggle_Covid-19_2.txt', 'r')

else:

    ctx = contexts



contx = []

for c in ctx:

    c = re.sub('Context: ','',c)

    c = c[:-1]

    #Do not take into account contexs which contains the pronoun "we"...

    pron = c.lower().split()

    if "we" in pron:

        continue

    elif "COVID-19".lower() in c.lower() or "SARS-COV-2".lower() in c.lower(): # make sure the contexts talk about COVID-19 and not about other viruses. 

        contx.append(c.lower())



print("Number of contexts: ", len(contx))
import sklearn

from sklearn.feature_extraction.text import CountVectorizer as BoW

from sklearn.feature_extraction.text import TfidfVectorizer as TF_IDF



bow = BoW(ngram_range=(1,1)).fit(contx)

bw = bow.transform(contx)

X = bw.toarray()

print("Dimension of the matrix of Data: ", X.shape)
from sklearn.decomposition import LatentDirichletAllocation



lda = LatentDirichletAllocation(n_components=5) #Number of clusters (5)

lda.fit(bw)



#get clusters for some given samples:

clusters_ids = []

for i in range(bw.shape[0]):

    prob = lda.transform(bw[i])

    clusters_ids.append((i,np.argmax(prob)))
samplesXcluster = [0,0,0,0,0]    # 5 clusters

y = []

for c in clusters_ids:

    y.append(c[1])

    samplesXcluster[c[1]]+=1

y = np.asarray(y)
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

clt = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

samples = [samplesXcluster[0],samplesXcluster[1],samplesXcluster[2],samplesXcluster[3],samplesXcluster[4]]

ax.bar(clt,samples)

plt.show()
from sklearn.linear_model import LogisticRegression as LR

from sklearn.model_selection import KFold

from sklearn.svm import SVC



kf = KFold(n_splits=50, shuffle=True)

scores = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]



    #Logistic regression classifier

    clf = LR(C=1, max_iter=200, solver='liblinear').fit(X_train, y_train)

    

    scores.append(clf.score(X_test, y_test))

print('Score = ', np.round(np.mean(np.asarray(scores)),2))
## Test classification using some of the Task's keywords and phrases.



## Comment or uncomment according to the query to be used.



#query = ["incubation period in days"]

#query = ["how long individuals are contagious."]

#query = ["how long individuals are contagious after recovery."]

query = ["Prevalence of asymptomatic shedding"]

#query = ["Prevalence of asymptomatic transmission in children"]

#query = ["transmission in children"]

#query = ["seasonality of transmission"]

#query = ["physical science of the coronavirus"]

#query = ["Smoking, pre-existing lung disease"]

#query = ["Role of the environment in transmission"]

#quetion = ["Effectiveness of personal protective equipment"]

#query = ["control strategies"]

#query = ["how long individuals are contagious, even after recovery"]

#query = ["Physical science"]

#query = ["Persistence and stability"]

#query = ["Persistence of virus on surfaces"]

#query = ["history of the virus"]

#query = ["diagnostic"]

#query = ["phenotypic change"]





qf = bow.transform(query)

ind = clf.predict(qf)

out = []

for c in clusters_ids:

    if c[1] == ind:

        out.append(contx[c[0]][1:])
from scipy.spatial.distance import cosine as dist

import operator



answer = bow.transform(out).toarray()

qq = qf.toarray()

distances = np.zeros(answer.shape[0])

ans_dist = dict()

i = 0

for a in answer:

    d = dist(a,qq)

    distances[i] = d 

    ans_dist.update({i:d})

    i+=1

    

dist_sort = sorted(ans_dist.items(), key=operator.itemgetter(1), reverse=False)



answer_words = ""

results2show = 10

for k in range(results2show):

    try:

        ans = out[dist_sort[k][0]]

        print("- ", ans + "\n")

        answer_words += ans + ' ' 

    except:

        continue
from wordcloud import WordCloud, STOPWORDS



stopwords = set(STOPWORDS)



wordcloud = WordCloud(width = 600, height = 600,

                background_color ='black',

                stopwords = stopwords,

                min_font_size = 10).generate(answer_words)

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None)

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad = 0)

  

plt.show()