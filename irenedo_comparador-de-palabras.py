# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



import gensim, logging

from gensim.models import word2vec



stops = set(stopwords.words("english"))

simbolos = ['.','.','...','@','$','(',')','"',':',';','?']

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

pal = sns.color_palette()

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_part = df_train[0:2500]

len(df_train)
#comparador de palabras y procesado simple de texto: tokenizador, filtrado(lower y alfanumerico) 

# y limpieza(no stops) 

#probar despues con el stemer y el lemmatizador



def same_word_ratio_2(row):

    

    ratiow = []

    #question1 = row['question1'].values.tolist()

    #question2 = row['question2'].values.tolist()

    for i in range(len(row)):

        n_words = 0     

        q1 = (pd.Series(row['question1'][i]).astype(str))[0]

        q2 = (pd.Series(row['question2'][i]).astype(str))[0] 

        if q2 == []: 

            q2 =""

        if q1 == []:

            q1 ="" 

        q_tokens1 = word_tokenize(q1) #tokenizar frase por frase, las frases con sus tokens separados

        q_tokens2 = word_tokenize(q2)

        

        filtered_tokens1 = [token.lower() for token in q_tokens1 if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)

        filtered_tokens2 = [token.lower() for token in q_tokens2 if token.isalnum()]

        

        clean_tokens1 = [token for token in filtered_tokens1 if token not in stops] #quitar stopwords

        clean_tokens2 = [token for token in filtered_tokens2 if token not in stops]

        

            #meter lo de word2vec--->>> most_similar() y model.wv.simila

        

        if (len(clean_tokens1) + len(clean_tokens2)) == 0:

            r = len(set(clean_tokens1) & set(clean_tokens2))

        else:         

            r = len(set(clean_tokens1) & set(clean_tokens2))/(len(set(clean_tokens1)) + len(set(clean_tokens2)))

    

        ratiow.append(r)

        

    return ratiow
def get_clean_tokens(df, question):

     

    clean_tokens = []

    #question1 = row['question1'].values.tolist()

    #question2 = row['question2'].values.tolist()

    for i in range(len(df)):    

        q = (pd.Series(df[question][i]).astype(str))[0] 

        q_tokens = word_tokenize(q) #tokenizar frase por frase, las frases con sus tokens separados        

        filtered_tokens = [token.lower() for token in q_tokens if token not in simbolos  ]#para limpiar las frases(minusculas y alfanumeric)

        cleans = [token for token in filtered_tokens if token not in stops] #quitar stopwords

        clean_tokens.append(cleans)

        

        

    return clean_tokens

#devuelve una matriz con los clean tokens por frases.
#same_word_raio_2 reducida->>>>



def same_word_ratio(df):#df:matriz de datos (df_train)

    

    ratiow = []

    clean_tokens1 = get_clean_tokens(df, 'question1')

    clean_tokens2 = get_clean_tokens(df, 'question2')

    for i in range(len(clean_tokens1)):

        

        if (len(clean_tokens1[i]) + len(clean_tokens2[i])) == 0:

            r = len(set(clean_tokens1[i]) & set(clean_tokens2[i]))

        else:         

            r = len(set(clean_tokens1[i]) & set(clean_tokens2[i]))/(len(set(clean_tokens1[i])) + len(set(clean_tokens2[i])))

    

        ratiow.append(r)

        

    return ratiow
q1 = (pd.Series(df_part['question1'][42]).astype(str))[0]   

q2 = (pd.Series(df_part['question2'][42]).astype(str))[0]   

x = df_part['question1'].values.tolist()

y = df_part['question2'].values.tolist()

words_q1 = (" ".join(q1)).lower().split()

print(x[42])

q11 = word_tokenize(q1)

print(q11)

filt1 = [token.lower() for token in q11 if token.isalnum()]

print(filt1)
q_tokens = word_tokenize(q) #tokenizar frase por frase, las frases con sus tokens separados        

filtered_tokens = [token.lower() for token in q_tokens if token.isalnum()]#para limpiar las frases(minusculas y alfanumeric)

cleans = [token for token in filtered_tokens if token not in stops] #quitar stopwords
clean_tokens1 = get_clean_tokens(df_part, 'question1')

clean_tokens2 = get_clean_tokens(df_part, 'question2')



print(clean_tokens1[42])

print(clean_tokens2[42])

num = '50,000'

if(num.isalnum()):

    print('siiiiiiiiiiiiiiiiiiiiii')

else:

    print('nooooooooooooooooooo')

print(num.lower())



num.isalnum()
#FUNCIÓN DE WORD2VEC

def word2vec_sim(df):

    ratios = []

    clean1_redu = []

    clean2_redu = []

    clean1 = get_clean_tokens(df, 'question1')

    clean2 = get_clean_tokens(df, 'question2')

    

    

    for c in range(len(clean1)):

        count = 0

        sim = 0

        dif = 0

        for i in range(len(clean1[c])):

            for j in range(len(clean2[c])):

                

                try:

                    if(model.similarity(clean1[c][i], clean2[c][j] ) == 1):

                        clean1.remove(clean1[c][i])

                        clean2.remove(clean2[c][j])



                except KeyError:

                    pass 

                    

        minimo = min(len(clean2[c]), len(clean1[c])) 

        if (minimo == 0):

            ratio = 0

            sim = 0

            dif = 0

        else:

            ratio = count / minimo

            sim = sim /minimo

            dif = dif / minimo

        ratios.append(dif)

        #ratios.append(sim)

        

    return clean1 
cleans = word2vec_sim(df_part)

print(cleans)
#pruebas same_word_ratio reducida

ratios = same_word_ratio(df_part) #matriz de ratios para decidir 1 o 0.
print(ratios[53])
#PRUEBA PARA EL WORD2VEC

"""

def word2vec_sim(clean1, clean2):

    ratios = []

    count = 0

    for c in range(len(clean1)):

        for i in range(len(clean1[c])):

            for j in range(len(clean2[c])):

                if(model.wv.similarity(clean1[c][i], clean2[c][j] ) > 0.7):

                    count = count + 1

        minimo = min(len(clean2[c]), len(clean1[c])) 

        if (minimo == 0):

            ratio = 0

        else:

            ratio = count / minimo

            

        ratios.append(ratio)

        

    return ratios 

    

"""                
def decision(matriz_est, umbral):

    decision = {} #tabla con los datos estimados, duplicados o no.

    #matriz_est: matriz de "scores" de la que se saca una matriz decision (ratiow)

    for index in range(len(matriz_est)):

        if (matriz_est[index] > umbral):

            decision[index] = 1

        else:

            decision[index] = 0

    #print (decision)

    return decision



def porcentaje_acierto(matriz_dec, matriz_df): 

    acierto =  {} #tabla para porcentajes de aciertos en la estimación

    counter = 0

    for index in range(len(matriz_dec)):

        if (matriz_dec[index] == matriz_df['is_duplicate'][index]):

            acierto[index]=1

            counter = counter +1

        else:

            acierto[index]=0

        

    porcentaje = (counter*100)/len(matriz_df) 

    return porcentaje
ratiow = same_word_ratio_2(df_train)
decision = decision(ratiow, 0.25)

porcentaje = porcentaje_acierto(decision, df_train) 

print (porcentaje)

#0.25 es el umbral ideal para df_part
x = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]

y = [0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0,3]

for w in x:

    decision = decision(ratiow, w)

    p = porcentaje_acierto(decision,df_train)

    print(p)


def no_detections_graves(ratios, df_x ): #ratios: una matriz de decision con ratios calculados

                               #df_x: matriz original con las preguntas

    

    no_detections = {}

    no_detections_q = {}

    no_detections_q1 = {}

    no_detections_q2 = {}

    no_detections_q3 = {}

    counter = 0

    

    for index in range(len(ratios)):

        if ((ratios[index] < 0.15) and (df_x['is_duplicate'][index] == 1)):

            no_detections[index] = 1

            #no_detections_q2 = pd.DataFrame({'id': df_x['test_id'][index], 'question1': df_x['question1'][index], 'question2': df_x['question2'][index]})

            no_detections_q1[index] = df_x['question1'][index]

            no_detections_q2[index] = df_x['question2'][index]

            no_detections_q3[index] = ratios[index]



            counter = counter + 1

        else:

            no_detections[index] = 0

            

    #no_detections_q = pd.DataFrame({'question1': no_detections_q1, 'question2': no_detections_q2, 'ratio': no_detections_q3})   

    no_detections_q['question1'] = no_detections_q1

    no_detections_q['question2'] = no_detections_q2

    no_detections_q['ratio'] = no_detections_q3



    return no_detections_q
def falsas_alarmas_graves(ratios, df_x ): #ratios: una matriz de decision con ratios calculados

                               #df_x: matriz original con las preguntas

    

    falsa_alarma = {}

    falsa_alarma_q = {}

    falsa_alarma_q1 = {}

    falsa_alarma_q2 = {}

    falsa_alarma_q3 = {}

    counter = 0

    

    for index in range(len(ratios)):

        if ((ratios[index] > 0.46) and (df_x['is_duplicate'][index] == 0)):

            falsa_alarma[index] = 1

            #falsa_alarma_q2 = pd.DataFrame({'id': df_x['test_id'][index], 'question1': df_x['question1'][index], 'question2': df_x['question2'][index]})

            falsa_alarma_q1[index] = df_x['question1'][index]

            falsa_alarma_q2[index] = df_x['question2'][index]

            falsa_alarma_q3[index] = ratios[index]

            counter = counter + 1

        else:

            falsa_alarma[index]=0

            

    #falsa_alarma_q = pd.DataFrame({'question1': falsa_alarma_q1, 'question2': falsa_alarma_q2, 'ratio': falsa_alarma_q3})

    falsa_alarma_q['question1'] = falsa_alarma_q1

    falsa_alarma_q['question2'] = falsa_alarma_q2

    falsa_alarma_q['ratio'] = falsa_alarma_q3

    return falsa_alarma_q
no_detections_g = no_detections_graves(ratios , df_part)

falsas_alarmas_g = falsas_alarmas_graves(ratios, df_part )
#no_detections_g

falsas_alarmas_g
ratiow_test = same_word_ratio_2(df_test)
decision_test = decision(ratiow_test, 0.25)
decision_test = decision(ratiow_test, 0.25)
#submission = pd.DataFrame({'test_id': df_test['id'],'is_duplicate': decision_test})

submission = pd.DataFrame({'test_id': df_train['id'], 'is_duplicate': decision})



submission.to_csv("submission.csv", index=False)

#sub = pd.DataFrame()

#sub['test_id'] = df_test['id']

#sub['is_duplicate'] = decision_test

#sub.to_csv('simple_counw.csv', index=False)

submission.head()