# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

colors = sns.color_palette()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
print("Popis datoteka:")
for f in os.listdir('../input'):
    print(f.ljust(15) + str(round(os.path.getsize('../input/' + f)/1000000,2))+"MB")
train_file = pd.read_csv('../input/train.csv')
train_file.head()
print("Ucitavam podatke iz test filea")
test_file = pd.read_csv('../input/test.csv')
print("Ukupan broj parova pitanja za testiranje: {}".format(len(test_file)))
test_file.head()
from nltk.corpus import stopwords
skips = set(stopwords.words("english")) # Definiramo skup riječi koje nemaju semantičko značenje u engleskom jeziku

import gensim
#kreiramo model za word2vec na temelju riječi koje se pojavljuju u skupu i potom ćemo ga primjeniti
#na naš skup pitanja
i = 0

def dictionar(row):
    global i
    #i = i +1
    #if (i%50000 == 0):
        #print("Prošao sam {:.2f}%".format((float(i)/float(404290))*100))
    sent = []
    str1 = str(row['question1']).lower().split()
    str2 = str(row['question2']).lower().split()
    
    for word in str1:
        sent.append(word)
            
    for word in str2:
        sent.append(word)
   
    
    return sent

print("Počinjem s obradom teksta i ubacivanjem u listu:")
sentences_train = []
sentences_train.extend(train_file.apply(dictionar, axis = 1, raw = True))
print("Broj zapisa u sentences: {}".format(len(sentences_train)))
print("Odradio izbacivanje stopwordsa i prebacivanje rečenica u listu! Krećem s pripremom train_modela")
my_model_train = gensim.models.Word2Vec(sentences_train, size=300, workers=30, iter=5, negative=2)
print ("Odradio sam generiranje modela.")
#sada ćemo definirati funkciju koja omogućava uspoređivanja dva skupa pitanja po tome imaju li ili ne iste riječi

#i = 0

def new_word_match(row):
    global my_model_train
    #global i
    #if (i+1) > len(train_file):
    #   i = 0
    #if(i <= 2):
        #print (i)
    #i = i +1
    #if (i%50000 == 0):
        #print("Prošao sam {:.2f}%".format((float(i)/float(404290))*100))
    
    
    
    q1_l = str(row['question1']).lower().split()
    q2_l = str(row['question2']).lower().split()
    
    for word in q1_l:
        if word in skips:
            q1_l.remove(word)
    
    for word in q2_l:
        if word in skips:
            q2_l.remove(word)
          
    if (len(q2_l)==0) or (len(q2_l) == 0):
        return 0
    
    slicnost1 = [w for w in q1_l if w in q2_l] 
    slicnost2 = [w for w in q2_l if w in q1_l]
    
    razlika1 = [w for w in q1_l if w not in q2_l]
    razlika2 = [w for w in q2_l if w not in q1_l]
    
    koef_sl = ((len(slicnost1) + len(slicnost2))/(len(q1_l) + len(q2_l)))
    koef_rz = ((len(razlika1) + len(razlika2))/(len(q1_l) + len (q2_l)))
    
    
    RET = 0
    if (koef_rz > koef_sl):
        RET = koef_sl
    else:
        sim_coeff = 0
        k = 0
        for word1 in razlika1:
            for word2 in razlika2:
                if word1 in my_model_train.wv.vocab and word2 in my_model_train.wv.vocab:
                    sim_coeff = sim_coeff + my_model_train.similarity(word1, word2)
                    k = k + 1
        if (k == 0):
            RET = koef_sl + sim_coeff
        else:
            RET = koef_sl + (sim_coeff/k)
    return RET



print("počinjem računanje test podataka za csv")
new_word_match_train_list = test_file.apply(new_word_match, axis = 1, raw=True)

print("počinjem upisivati podatke u file")
sub = pd.DataFrame()
sub['id'] = test_file['id']
sub['new_word_match_test'] = new_word_match_train_list
sub.to_csv('test_file_new.csv', index=False)
print("Gotov sa zapisivanjem")

print("Grafiranje dobivenih podataka:")
plt.figure(figsize=(15, 5))
plt.hist(new_word_match_train_list[train_file['is_duplicate'] == 0], bins=20, normed = True, label="Nije duplikat")
plt.hist(new_word_match_train_list[train_file['is_duplicate'] == 1], bins=20, normed = True, alpha=0.5, label = "Duplikat")
plt.legend()
plt.title("prikaz odnosa duplikata i onih koji nisu duplikati", fontsize=15)
plt.xlabel("Word_match", fontsize=15)
j = 0

def word_match(row):
    #j = j +1
    global my_model_train
    """global j
    if (j <= 2):
        print(j)
    if (j%50000 == 0):
        print ("Prošao sam {:.2f}%".format((float(j)/float(404290))*100))
    j = j+1
    """
    q1_l = str(row['question1']).lower().split()
    q2_l = str(row['question2']).lower().split()
    
    for word in q1_l:
        if word in skips:
            q1_l.remove(word)
    
    for word in q2_l:
        if word in skips:
            q2_l.remove(word)
            
    if (len(q2_l)==0) or (len(q2_l) == 0):
        return 0
    """
    slicnost1 = [w for w in q1_l if w in q2_l] 
    slicnost2 = [w for w in q2_l if w in q1_l]
    
    razlika1 = [w for w in q1_l if w not in q2_l]
    razlika2 = [w for w in q2_l if w not in q1_l]
    
    koef_sl = ((len(slicnost1) + len(slicnost2))/(len(q1_l) + len(q2_l)))
    koef_rz = ((len(razlika1) + len(razlika2))/(len(q1_l) + len (q2_l)))
    """
    RET = 0
   
    
    sim_coeff = 0
    k = 0
    for word1 in q1_l:
        for word2 in q2_l:
                if word1 in my_model_train.wv.vocab and word2 in my_model_train.wv.vocab:
                    sim_coeff = sim_coeff + my_model_train.similarity(word1, word2)
                    k = k + 1
        if (k == 0):
            RET = sim_coeff
        else:
            RET = sim_coeff/k
    return RET


print("počinjem računanje train podataka")
word_match_train_list = test_file.apply(word_match, axis = 1, raw=True)

print("počinjem upisivati podatke u file")
sub = pd.DataFrame()
sub['id'] = test_file['id']
sub['word_match_test'] = word_match_train_list
sub.to_csv('test_file_old.csv', index=False)
print("Gotov sa zapisivanjem")

plt.figure(figsize=(15, 5))
plt.hist(word_match_train_list[train_file['is_duplicate'] == 0], bins=20, normed = True, label="Nije duplikat")
plt.hist(word_match_train_list[train_file['is_duplicate'] == 1], bins=20, normed = True, alpha=0.5, label = "Duplikat")
plt.legend()
plt.title("prikaz odnosa duplikata i onih koji nisu duplikati", fontsize=15)
plt.xlabel("Word_match", fontsize=15)
from sklearn.metrics import roc_auc_score
print('New Match AUC:', roc_auc_score(train_file['is_duplicate'], new_word_match_train_list))
print('Original Match AUC:', roc_auc_score(train_file['is_duplicate'], word_match_train_list))