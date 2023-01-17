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

import sentencepiece as spm





import  logging

logging.basicConfig(filename='out.log',format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)

logger=logging.getLogger()

logger.setLevel(logging.DEBUG)



# create console handler and set level to debug

ch = logging.StreamHandler()

ch.setLevel(logging.DEBUG)



# create formatter

formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')



# add formatter to ch

ch.setFormatter(formatter)



# add ch to logger

logger.addHandler(ch)





!cp /kaggle/input/fasttext-skipgram-francais-16m/sortie/textes.txt.xz /tmp/. -v

!xz -d -v /tmp/textes.txt.xz

!rm /tmp/textes.txt.xz

logger.info("text")

!ls -l

!ls -l /tmp/
input_sentence_size=900000



input_sentence_size=1800000
import sentencepiece as spm

vocab=1024

vocab_text="1024"

spm_train_options='--control_symbols=<cls>,<sep>,<pad>,<mask>,<eod> --user_defined_symbols=<eop>,.,(,),",-,–,£,€ --input_sentence_size=900000 --shuffle_input_sentence=true  '

# spm_train_options+=" --accept_language fr "



spm_train_txt=f'--input=/tmp/textes.txt --model_prefix=spm_v{vocab_text} --vocab_size={vocab} --num_threads=4 '+spm_train_options

print("SentencePiece vocab",vocab,vocab_text,spm_train_txt)

logger.debug(f"SentencePiece vocab {vocab}  {vocab_text}  {spm_train_txt}")

spm.SentencePieceTrainer.Train(spm_train_txt)

logger.info("SentencePiece 1024")

!ls -l




for vocab in [1000,8000,32000,64000,128000,256000,128000,512000,1e6,4e6,8e6,16e6]:

    vocab=int(vocab)

    vocab_text=str(vocab)



    if vocab>3000:

        if vocab>3e6:

            vocab_text=str(vocab//int(1e6))+"M"

        else:

            vocab_text=str(vocab//1000)+"k"

        if vocab>128000:

            input_sentence_size=input_sentence_size*10

            

    spm_train_txt=f'--input=/tmp/textes.txt --model_prefix=spm_v{vocab_text} --vocab_size={vocab} --num_threads=4  --input_sentence_size={input_sentence_size} --shuffle_input_sentence=true '+spm_train_options

    print("SentencePiece vocab",vocab,vocab_text,spm_train_txt)

    logger.debug(f"SentencePiece vocab {vocab}  {vocab_text}  {spm_train_txt}")

    logger.info(spm_train_txt)

    try:

        spm.SentencePieceTrainer.Train(spm_train_txt)

    except:

        logging.exception(f"SentencePiece vocab {vocab} {spm_train_txt}")

!ls -l



!ls -l