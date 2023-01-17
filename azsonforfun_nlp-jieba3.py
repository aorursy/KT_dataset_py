# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=False

fir=False

second=False

third=False

fourth=False

!pip install opencc

!pip install opencc-python-reimplemented
from gensim.test.utils import datapath, get_tmpfile

from gensim.corpora import WikiCorpus, MmCorpus

from gensim.test.utils import common_texts

from gensim.models.word2vec import Word2Vec



import multiprocessing

import pickle, time

import gensim



def log_time():

    print("Time : {0}".format(time.time()))
def cosine(x, y):

    val = np.matmul(x.reshape((1,-1)), y.reshape((-1,1)))

    norm = np.linalg.norm(x) * np.linalg.norm(y)

    return val / norm
import jieba



wiki=None



log_time()

# wiki = WikiCorpus("/kaggle/input/wiki-corpus/zhwiki-20191120-pages-articles-multistream.xml.bz2")

with open("/kaggle/input/wiki-corpus/wiki-corpus.pkl", "rb") as f:

    wiki = pickle.load(f)

log_time()



log_time()

with open("wiki-corpus.pkl", "wb") as f:

    pickle.dump(wiki, f)

log_time()
pku_vocab = []

with open('/kaggle/input/wiki-corpus/pku_sim_test.txt', 'r', encoding='utf8') as f:

    pku_vocab = ''.join(f.readlines())

    pku_vocab = list(set(pku_vocab.split()))
if fir:

    temporary_filepath="/kaggle/input/wiki-corpus/new-wv-100-wiki-ok.model"

    batch_size=100

    step=0



    model = gensim.models.Word2Vec.load(temporary_filepath)



    more_sentences = []



    log_time()

    for idx, line in enumerate(wiki.get_texts()):



        new_sent = list(jieba.cut(''.join(line)))



        more_sentences.append(new_sent)



        if len(more_sentences) == batch_size:



            model.build_vocab(more_sentences, update=True)

            if step == 0:

                model.build_vocab(pku_vocab, update=True)

                print("do first~")



            model.train(more_sentences, total_examples=model.corpus_count, \

                        epochs=model.epochs, compute_loss=True, report_delay=60*10)

            more_sentences = []

            step += 1

            if step % 100 == 0:

                log_time()

                print(model.get_latest_training_loss())

                if step % 300 == 0:

                    print("out tmp model~")

                    model.save("new_wv-100-wiki-step-%d.model" % step)



    model.save("new2-wv-100-wiki-ok.model")

    print("finished all!")
if  fir:

    model = model_new2



    oov_num = 0

    all_cnt = 0



    with open('/kaggle/input/wiki-corpus/pku_sim_test.txt', 'r', encoding='utf8') as f:



        with open('pku_sim_ans-1.txt', 'w', encoding='utf8') as f2:



            for line in f.readlines():

                w1, w2 = line.split()

                all_cnt += 1

                f2.write(w1 + '\t' + w2 + '\t')

                if w1 in model and w2 in model:

                    score = cosine(model[w1], model[w2])

                    f2.write(str(score[0][0])+'\n')

                else:

                    f2.write('OOV\n')

                    oov_num += 1
# adust vocab

if train:

    for wd in pku_vocab:

        jieba.suggest_freq(wd, True)
if second:



    

    temporary_filepath="/kaggle/input/wiki-corpus/new2-wv-100-wiki-ok.model"

    batch_size=100

    step=0



    model = gensim.models.Word2Vec.load(temporary_filepath)

#     model = model_new2

    more_sentences = []



    log_time()

    for idx, line in enumerate(wiki.get_texts()):



        new_sent = list(jieba.cut(' '.join(line).lower(), HMM=False))



        more_sentences.append(new_sent)



        if len(more_sentences) == batch_size:



            model.build_vocab(more_sentences, update=True)

            if step == 0:

                model.build_vocab(pku_vocab, update=True)

                print("do first~")



            model.train(more_sentences, total_examples=model.corpus_count, \

                        epochs=model.epochs, compute_loss=True, report_delay=60*10)

            more_sentences = []

            step += 1

            if step % 100 == 0:

                log_time()

                print(model.get_latest_training_loss())

                if step % 300 == 0:

                    print("out tmp model~")

#                     model.save("new_wv-100-wiki-step-%d.model" % step)



    model.save("new3-wv-100-wiki-ok.model")

    print("finished all!")
from opencc import OpenCC

cc = OpenCC('t2s')  # convert from Simplified Chinese to Traditional Chinese

# can also set conversion by calling set_conversion

# cc.set_conversion('s2tw')

to_convert = '開放 中文轉換'

converted = cc.convert(to_convert)
if third:

    

    params = {'size': 100, 'window': 2, 'min_count': 5, 

              'workers': max(1, multiprocessing.cpu_count() - 1), 'sample': 1E-3,}

    model = Word2Vec(**params)

    model.save("new3-wv-100-wiki-ok.model")

    

    temporary_filepath="new3-wv-100-wiki-ok.model"

    batch_size=100

    step=0



    model = gensim.models.Word2Vec.load(temporary_filepath)

#     model = model_new2

    more_sentences = []



    log_time()

    for idx, line in enumerate(wiki.get_texts()):

        

        converted = cc.convert(' '.join(line).lower())

        new_sent = list(jieba.cut(converted, HMM=False))



        more_sentences.append(new_sent)



        if len(more_sentences) == batch_size:

            

            if step == 0:

                model.build_vocab(pku_vocab)

                print("do first~")

                

            model.build_vocab(more_sentences, update=True)



            model.train(more_sentences, total_examples=model.corpus_count, \

                        epochs=model.epochs, compute_loss=True, report_delay=60*10)

            more_sentences = []

            step += 1

            if step % 100 == 0:

                log_time()

                print(model.get_latest_training_loss())

                if step % 300 == 0:

                    print("out tmp model~")

#                     model.save("new_wv-100-wiki-step-%d.model" % step)



    model.save("new3-wv-100-wiki-ok.model")

    print("finished all!")
if third:

    oov_num = 0

    all_cnt = 0



    with open('/kaggle/input/wiki-corpus/pku_sim_test.txt', 'r', encoding='utf8') as f:



        with open('pku_sim_ans-4.txt', 'w', encoding='utf8') as f2:



            for line in f.readlines():

                w1, w2 = line.split()

                w1_t = w1.lower()

                w2_t = w2.lower()

                all_cnt += 1

                f2.write(w1 + '\t' + w2 + '\t')

                if w1_t in model and w2_t in model:

                    score = cosine(model[w1_t], model[w2_t])

                    f2.write(str(score[0][0])+'\n')

                else:

                    f2.write('OOV\n')

                    oov_num += 1
if fourth:

    

    params = {'size': 100, 'window': 2, 'min_count': 5, 

              'workers': max(1, multiprocessing.cpu_count() - 1), 'sample': 1E-3,}

    model = Word2Vec(**params)

    model.save("new4-wv-100-wiki-ok.model")

    

    temporary_filepath="new4-wv-100-wiki-ok.model"

    batch_size=100

    step=0



    model = gensim.models.Word2Vec.load(temporary_filepath)

#     model = model_new2

    more_sentences = []



    log_time()

    for epoch in range(1):

        print("epoch", epoch)

        for idx, line in enumerate(wiki.get_texts()):





            converted = cc.convert(' '.join(line).lower())

            new_sent = list(jieba.cut(converted, HMM=False))



            more_sentences.append(converted.split(' '))

            more_sentences.append(new_sent)



            if len(more_sentences) >= batch_size:



                if step == 0:

                    model.build_vocab(pku_vocab)

                    print("do first~")



                model.build_vocab(more_sentences, update=True)



                model.train(more_sentences, total_examples=model.corpus_count, \

                            epochs=model.epochs, compute_loss=True, report_delay=60*10)

                more_sentences = []

                step += 1

                if step % 100 == 0:

                    log_time()

                    print(model.get_latest_training_loss())

    #                     if step % 300 == 0:

    #                         print("out tmp model~")

    #                     model.save("new_wv-100-wiki-step-%d.model" % step)

        if len(more_sentences):

            model.build_vocab(more_sentences, update=True)

            model.train(more_sentences, total_examples=model.corpus_count, \

                        epochs=model.epochs, compute_loss=True, report_delay=60*10)

            more_sentences = []

        

    model.save("new4-wv-100-wiki-ok.model")

    print("finished all!")
if not train:

    temporary_filepath="/kaggle/input/wiki-corpus/new4-wv-100-wiki-ok.model"



    model = gensim.models.Word2Vec.load(temporary_filepath)
oov_num = 0

all_cnt = 0



with open('/kaggle/input/wiki-corpus/pku_sim_test.txt', 'r', encoding='utf8') as f:



    with open('pku_sim_ans-6.txt', 'w', encoding='utf8') as f2:



        for line in f.readlines():

            w1, w2 = line.split()

            w1_t = cc.convert(w1.lower())

            w2_t = cc.convert(w2.lower())

            all_cnt += 1

            f2.write(w1 + '\t' + w2 + '\t')

            if w1_t in model and w2_t in model:

                score = cosine(model[w1_t], model[w2_t])

                f2.write(str(score[0][0])+'\n')

            else:

                f2.write('OOV\n')

                oov_num += 1
print(oov_num) 
print(model.most_similar('故事'))
print(model)