import pandas as pd

import numpy as np

from keras.utils import to_categorical 

from Bio.SeqIO.FastaIO import SimpleFastaParser
# Sequenze



def get_df_sequence(pathFile):

    with open(pathFile) as file:

        df = pd.DataFrame(SimpleFastaParser(file),columns=['id','seq'])

        df = df.assign(seq = df.seq.apply(lambda x : x.upper()))

        return df

   
# esempio

path = '../input/GM12878.fa'

df = get_df_sequence(path)

df.head()
def get_df_class(path):

    df = pd.read_csv(path,header=None,names=["class"])

    return df
# esempio

path = '../input/GM12878_class.csv'

df = get_df_class(path)

df.head()
def get_df_epigenomic(path):

    df = pd.read_csv(path)

    df = df.drop(["Unnamed: 0"],axis=1)

    return df
path = '../input/GM12878.csv'

df_s = get_df_epigenomic(path)

df_s.head()
df_s.info()
path_c = '../input/GM12878_class.csv'

path_e = '../input/GM12878.csv'

path_s = '../input/GM12878.fa'



df_c = get_df_class(path_c)

df_e = get_df_epigenomic(path_e)

df_s = get_df_sequence(path_s)
def convertToNum(iterable_singleton_elem):

    '''

    Crea una funzione di mapping da elementi a Naturali

    '''

    assert len(set(iterable_singleton_elem)) == len(iterable_singleton_elem)

    diz = {}

    for counter, value in enumerate(iterable_singleton_elem):

        diz[value] = counter 

    return lambda x : diz[x]











def get_type_of_nucl_from_list_of_sequence(seq):

    type_of_nucl = set()

    seq.map(lambda x : set(x)).apply(lambda x : type_of_nucl.update(x))

    type_of_nucl = list(type_of_nucl)

    type_of_nucl = sorted(type_of_nucl)

    len_of_type_of_nucl = len(type_of_nucl)

    return type_of_nucl



t = get_type_of_nucl_from_list_of_sequence(df_s['seq'])





nucl2int = convertToNum(t)





print(f'Tipi letti: {t}')



df_s["seq"][0]