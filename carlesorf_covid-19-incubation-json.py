import os

import json

import pandas as pd

from tqdm import tqdm #llibreria per veure la barra de increment
dir1 = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv'
for dirname, _, filenames in os.walk(dir1):

    for filename in filenames:

        print(filename)

    break
for dirname, _, filenames in os.walk(dir1):

    for filename in filenames:

        if filename.endswith(".json"):

            j=json.load(open(f'{dirname}/{filename}', 'rb'))

            print(j)

        

        break
for dirname, _, filenames in os.walk(dir1):

    for filename in filenames:

        if filename.endswith(".json"):

            j=json.load(open(f'{dirname}/{filename}'))

        

            for key in j:

                print(key[0:100])

            

            print(j['metadata'])

            for k in j['metadata']:

                print(k[0:100])

        

            break
for dirname, _, filenames in os.walk(dir1):

    for filename in filenames:

        j=json.load(open(f'{dirname}/{filename}'))

        

        for text in j['body_text']:

            print (text['text'])    

        

        break
for dirname, _, filenames in os.walk(dir1):

    for filename in filenames:

        j=json.load(open(f'{dirname}/{filename}'))

        

        full_text = ''

        

        for text in j['body_text']:

            full_text += text['text']+'\n\n'   

            

        print(full_text)

        

        break
docs = []



for dirname, _, filenames in os.walk(dir1):

    print (dirname)

    for filename in tqdm(filenames):

        j=json.load(open(f'{dirname}/{filename}'))

        

        #agafem el titol de cada paper de dins del metadata

        title = j['metadata']['title']

        

        #agafem l'abstract, no se pq amb 0, per algo del format

        try:

            abstract = j['abstract'][0]

        except:

            abstract = ''

        

        #agafem tot el text del paper

        full_text = ''

        

        for text in j['body_text']:

            full_text += text['text']+'\n\n' 

            

        #si imprimim el full text veurem que son paragrafs un darrera l'altre

        #print(full_text)

        

        #ho ajuntem tot en varible docs

        docs.append([title,abstract, full_text])
df = pd.DataFrame(docs, columns = ['title', 'abstract', 'full_text'])

df.head(10)
incubation = df[df['full_text'].str.contains('incubation')]

incubation.head(10)
#pasem a llista cutre el full text

texts = incubation['full_text'].values



for t in texts:

    print(t)

    break
#pasem a llista cutre el full text i busquem sobre aixo

texts = incubation['full_text'].values



for t in texts:

    for sentence in t.split('. '):

        if ('incubation' and 'days') in sentence:

            print(sentence)

        break       

        
import re  #regular expresion module



#pasem a llista cutre el full text i busquem sobre aixo

texts = incubation['full_text'].values



for t in texts:

    for sentence in t.split('. '):

        if 'incubation' in sentence:

            single_day = re.findall(r" \d{1,2} day", sentence)

            

            if len(single_day) == 1:

                print(single_day) #single_day[0] dona text nomes

                print(sentence)
#get a list of all incubation times

incubation_times = []



for t in texts:

    for sentence in t.split('. '):

        if 'incubation' in sentence:

            single_day = re.findall(r" \d{1,2} day", sentence)

            

            if len(single_day) == 1:

                num = single_day[0].split(' ')

                incubation_times.append(float(num[1]))

                

incubation_times
import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')



plt.hist(incubation_times)

plt.ylabel('bin counts')

plt.xlabel('incubation time (days)')

plt.show()
import numpy as np



print(f'The mean projected incubation time is {np.mean(incubation_times)}')
dir = '/kaggle/input/CORD-19-research-challenge'
docs = []



for dirname, _, filenames in os.walk(dir):

    print(dirname)

    for filename in filenames:

        if filename.endswith(".json"):

            j=json.load(open(f'{dirname}/{filename}'))

            

            #agafem el titol de cada paper de dins del metadata

            title = j['metadata']['title']

        

            #agafem l'abstract, no se pq amb 0, per algo del format

            try:

                abstract = j['abstract'][0]

            except:

                abstract = ''

        

            #agafem tot el text del paper

            full_text = ''

        

            for text in j['body_text']:

                full_text += text['text']+'\n\n' 

        

            #ho ajuntem tot en varible docs

            docs.append([title,abstract, full_text])

            
df = pd.DataFrame(docs, columns = ['title', 'abstract', 'full_text'])

df.shape
incubation = df[df['full_text'].str.contains('incubation')]

incubation.head(10)
import re  #regular expresion module



#pasem a llista cutre el full text i busquem sobre aixo

texts = incubation['full_text'].values



#get a list of all incubation times

incubation_times = []



for t in texts:

    for sentence in t.split('. '):

        if 'incubation' in sentence:

            single_day = re.findall(r" \d{1,2} day", sentence)

            

            if len(single_day) == 1:

                num = single_day[0].split(' ')

                incubation_times.append(float(num[1]))

                

len(incubation_times)
#nova llista treient els valors mes grans de 20

y = [i for i in incubation_times if i<20]

len(y)
import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')



plt.hist(y, bins = 10)

plt.ylabel('bin counts')

plt.xlabel('incubation time (days)')

plt.show()
import numpy as np



print(f'The mean projected incubation time is {np.mean(y)}')