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
transcriptions = {filename: f"{dirname}{filename}" for filename in filenames for dirname, _, filenames in os.walk("/kaggle/input/explorationspeechresources/")}
tedxCorpus=pd.read_csv(transcriptions["TEDx_Spanish.transcription"],names=['sentence','path'],sep="TEDX_")
tedxCorpus.loc[:,'path']=tedxCorpus.path.apply(lambda p:f"TEDX_{p}.wav")

tedxCorpus['gender']=tedxCorpus.path.apply(lambda x: 'male' if '_M_' in x else 'female')

tedxCorpus['accent']='mexicano'

transcriptions["TEDx_Spanish.transcription"]=tedxCorpus
transcriptions["validated.tsv"]=pd.read_table(transcriptions["validated.tsv"], header = 0)
def process_crowfund(path,gender,accent):

    df=pd.read_table(path, header = None,names=['path','sentence'])

    df['gender']=gender

    df['accent']=accent

    df.loc[:,'path']=df.path.apply(lambda p:p+'.wav')

    return df
transcriptions['es_co_male.tsv']=process_crowfund(transcriptions['es_co_male.tsv'],'male','andino')

transcriptions['es_co_female.tsv']=process_crowfund(transcriptions['es_co_female.tsv'],'female','andino')

transcriptions['es_pe_male.tsv']=process_crowfund(transcriptions['es_pe_male.tsv'],'male','andino')

transcriptions['es_pe_female.tsv']=process_crowfund(transcriptions['es_pe_female.tsv'],'female','andino')

transcriptions['es_portoric_female.tsv']=process_crowfund(transcriptions['es_portoric_female.tsv'],'female','caribe')
tmp=pd.read_json(transcriptions['female_mex.json'],orient='index')

transcriptions['female_mex.json']=pd.DataFrame()

transcriptions['female_mex.json']['path']=tmp.index

transcriptions['female_mex.json']['sentence']=tmp.clean.values

transcriptions['female_mex.json']['gender']='female'

transcriptions['female_mex.json']['accent']='mexicano'

tmp=None
def mergeDataframes(**dataframes):

    return pd.concat(list(dataframes.values()),axis=0).reset_index(drop=True)
mergedTranscriptions=mergeDataframes(**transcriptions)
def getFrequencyDistribution(df,column_name):

    return df[pd.notnull(df[column_name])].groupby(df[column_name]).size()
getFrequencyDistribution(mergedTranscriptions,'accent').plot.bar()
getFrequencyDistribution(mergedTranscriptions,'age').plot.bar()
getFrequencyDistribution(mergedTranscriptions,'gender').plot.bar()
len(mergedTranscriptions.path.unique())!=len(mergedTranscriptions)
len(mergedTranscriptions[mergedTranscriptions.sentence.isnull()]) +  len(mergedTranscriptions[mergedTranscriptions.path.isnull()])
from collections import defaultdict

phonetic_groups = defaultdict(lambda:'other',{

    **dict.fromkeys(['mexicano', 'andino', 'americacentral'], 'mexican_alike'), 

    **dict.fromkeys(['canario', 'caribe','rioplatense'], 'southAmerican'),

    **dict.fromkeys(['centrosurpeninsular', 'nortepeninsular','surpeninsular'], 'spaniards'),

    'chileno':'chileno'})
mergedTranscriptions.loc[:,'accent']=mergedTranscriptions.accent.apply(lambda a:phonetic_groups[a])
getFrequencyDistribution(mergedTranscriptions,'accent').plot.bar()
import numpy as np

def apply_w2l_format(dataframe):

    dataframe=dataframe.reset_index()

    dataframe.drop('index', axis=1, inplace=True)

    dataframe['unique_id']=dataframe.index

    dataframe['duration']=np.zeros(len(dataframe))

    dataframe=dataframe[['unique_id','path','duration','sentence']]

    return dataframe
mergedTranscriptions=apply_w2l_format(mergedTranscriptions)
mergedTranscriptions.groupby(mergedTranscriptions.sentence.apply(lambda s:len(s.split()))).size().plot.bar()
mergedTranscriptions.to_csv("raw_dataset.lst", sep='\t', index=False, header=None)
import re

import string

import ftfy

co_SentenceLevel = {

    #Separate simbols from words

    '?':' ? ',

    '¿':' ¿ ',

    ',':' , ',

    '\'':' \' ',

    '\.{2,}':' ',

    '.':' . ',

    ':':' : ',

    ftfy.fix_encoding('á'):ftfy.fix_encoding('A'),

    ftfy.fix_encoding('é'):ftfy.fix_encoding('E'),

    ftfy.fix_encoding('í'):ftfy.fix_encoding('I'),

    ftfy.fix_encoding('ó'):ftfy.fix_encoding('O'),

    ftfy.fix_encoding('ú'):ftfy.fix_encoding('U'),

    #delete some useless simbols

    '-':' ',

    '(':' ',

    ')':' ',

    #delete double space, and sequences of "-,*,^,."

    '\?{2,}|\!{2,}':' '

}





def escapePattern(pattern):

    """Helper function to build our regex"""

    if len(pattern)==1:

        pattern=re.escape(pattern)

    return pattern



def compileCleanerRegex(cleaningOptions):

    """Given a dictionary of rules this contruct the regular expresion to detect the patterns """

    return re.compile("(%s)" % "|".join(map(escapePattern,cleaningOptions.keys())))
delete=ftfy.fix_encoding('\"!¡#$%&()*+-/:<=>@[\\]^_`{|}\'~')

replaceVocal=ftfy.fix_encoding('äëïöü')



clean_regex=compileCleanerRegex(co_SentenceLevel)

rmPunc =str.maketrans('','',delete)

rPVocal =str.maketrans(replaceVocal,'aeiou')

norm_spaces=re.compile("\s{1,}")
def clean_text(text,cleaningOptions,cleaningRegex,removePunct,replaceVocab,norm_spaces):

    """Cleaning function for text

       Given a text this function applies the cleaning rules defined

       in a dictionary using a regex to detect the patterns.

   Args:

       text (str): The text we want to clean.

       cleaningRegex(regex): Regular expression to detect

                                    the patterns defined in the cleaning options

                                    compiled using the compileCleanerRegex(cleaningOptions) function.



    Returns:

        The cleaned text applying the cleaning options.

    """

    text=ftfy.fix_encoding(text).lower()

    text=cleaningRegex.sub(lambda mo:cleaningOptions.get(mo.group(1),), text)

    text=text.translate(removePunct)

    text=text.translate(replaceVocab)

    return ' '.join(norm_spaces.split(text.strip()))
from functools import partial

clean=partial(clean_text,cleaningOptions=co_SentenceLevel,cleaningRegex=clean_regex,removePunct=rmPunc,replaceVocab=rPVocal,norm_spaces=norm_spaces)
ph="""\"Tal programa, ""Rog-O-Matic"",el pingüino fue desarrollado para jugar.... y ganar el juego.\"  ángel ,  diego gómez ,  carlos o'connor reina ,  ma . """

clean(ph)
from multiprocessing import Pool

from tqdm.notebook import tqdm

with Pool(8) as p:

    mergedTranscriptions.loc[:,'sentence']=tqdm(p.imap(clean,mergedTranscriptions.sentence.values),

                                                total=len(mergedTranscriptions))
mergedTranscriptions['sentence'].sample(10).values
mergedTranscriptions.to_csv("punc_dataset.lst", sep='\t', index=False, header=None)
punclst=string.punctuation+'¿'

rmPunc =str.maketrans('','',punclst)

def remPunct(text,rmPunc=rmPunc,norm_spaces=norm_spaces):

    text=text.translate(rmPunc)

    return ' '.join(norm_spaces.split(text.strip())) 
with Pool(8) as p:

    mergedTranscriptions.loc[:,'sentence']=tqdm(p.imap(remPunct,mergedTranscriptions.sentence.values),

                                                total=len(mergedTranscriptions))
mergedTranscriptions['sentence'].sample(10).values
mergedTranscriptions.to_csv("np_accents_dataset.lst", sep='\t', index=False, header=None)
mergedTranscriptions.loc[:,'sentence']=mergedTranscriptions.sentence.apply(lambda s:s.lower())

mergedTranscriptions.to_csv("np_dataset.lst", sep='\t', index=False, header=None)

mergedTranscriptions['sentence'].sample(10).values