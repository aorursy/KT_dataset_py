import os

import math

import numpy as np

import pandas as pd

import seaborn as sns



%matplotlib inline

import matplotlib.pyplot as plt

train = pd.read_csv("../input/training_variants")

test = pd.read_csv("../input/test_variants")

#train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

#test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])



print("Train shape".ljust(15), train.shape)

print("Test shape".ljust(15), test.shape)
df_joint = pd.concat([train,test])

print("train+test rows:",df_joint.shape[0])



genes = set(df_joint["Gene"])

print("%i unique Genes" %(len(genes)))



variations = set(df_joint["Variation"])

print("%i unique Variations" %(len(variations)))
"Could use bioPython's IUPAC alphabet. This is simpler for now, although realistically we might want to handle non standard AA (e.g. selenocysteine, B,U,Z..)"

AA_VALID = 'ACDEFGHIKLMNPQRSTVWY'
df_joint["simple_variation_pattern"] = df_joint.Variation.str.contains(r'^[A-Z]\d{1,7}[A-Z]',case=False)
print("We capture most variants with this (and possibly even more in train)")

df_joint["simple_variation_pattern"].describe()

df_joint[df_joint["simple_variation_pattern"]==False]["Variation"].head(15)
# Get location in gene / first number , from first word (otherwise numbers appear later)

df_joint['location_number'] = df_joint.Variation.str.extract('(\d+)')
df_joint['variant_letter_first'] = df_joint.apply(lambda row: row.Variation[0] if row.Variation[0] in (AA_VALID) else np.NaN,axis=1)

df_joint['variant_letter_last'] = df_joint.apply(lambda row: row.Variation.split()[0][-1] if (row.Variation.split()[0][-1] in (AA_VALID)) else np.NaN ,axis=1)
df_joint['variant_letter_last'].describe()
df_joint[['variant_letter_first',"Variation",'variant_letter_last',"simple_variation_pattern"]].head(4)
" Replace letters with NaNs for cases that don't match our pattern. (Need to check if this actually improves results!)"

df_joint.loc[df_joint.simple_variation_pattern==False,['variant_letter_last',"variant_letter_first"]] = np.NaN
"""

## Bioinformatics Code + alphabet feature engineering from: https://github.com/ddofer/ProFET/blob/master/ProFET/feat_extract/AAlphabets.py



ProFET: Feature engineering captures high-level protein functions.

Ofer D, Linial M.

Bioinformatics. 2015 Nov 1;31(21):3429-36. doi: 10.1093/bioinformatics/btv345.

PMID: 26130574

"""



def TransDict_from_list(groups):

    '''

    Given a list of letter groups, returns a dict mapping each group to a

    single letter from the group - for use in translation.

    >>> alex6=["C", "G", "P", "FYW", "AVILM", "STNQRHKDE"]

    >>> trans_a6 = TransDict_from_list(alex6)

    >>> print(trans_a6)

    {'V': 'A', 'W': 'F', 'T': 'D', 'R': 'D', 'S': 'D', 'P': 'P',

     'Q': 'D', 'Y': 'F', 'F': 'F',

     'G': 'G', 'D': 'D', 'E': 'D', 'C': 'C', 'A': 'A',

      'N': 'D', 'L': 'A', 'M': 'A', 'K': 'D', 'H': 'D', 'I': 'A'}

    '''

    transDict = dict()



    result = {}

    for group in groups:

        g_members = sorted(group) #Alphabetically sorted list

        for c in g_members:

            result[c] = str(g_members[0]) #K:V map, use group's first letter as represent.

    return result



ofer8=TransDict_from_list(["C", "G", "P", "FYW", "AVILM", "RKH", "DE", "STNQ"])



sdm12 =TransDict_from_list(

    ["A", "D", "KER", "N",  "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"] )



pc5 = {"I": "A", # Aliphatic

         "V": "A",         "L": "A",

         "F": "R", # Aromatic

         "Y": "R",         "W": "R",         "H": "R",

         "K": "C", # Charged

         "R": "C",         "D": "C",         "E": "C",

         "G": "T", # Tiny

         "A": "T",         "C": "T",         "S": "T",

         "T": "D", # Diverse

         "M": "D",         "Q": "D",         "N": "D",

         "P": "D"}
"You can encode the reduced alphabet as OHE features; in peptidomics this gives highly generizable features."

df_joint['AAGroup_ofer8_letter_first'] = df_joint["variant_letter_first"].map(ofer8)

df_joint['AAGroup_ofer8_letter_last'] = df_joint["variant_letter_last"].map(ofer8)

df_joint['AAGroup_ofer8_equiv'] = df_joint['AAGroup_ofer8_letter_first'] == df_joint['AAGroup_ofer8_letter_last']



df_joint['AAGroup_m12_equiv'] = df_joint['variant_letter_last'].map(sdm12) == df_joint['variant_letter_first'].map(sdm12)

df_joint['AAGroup_p5_equiv'] = df_joint['variant_letter_last'].map(pc5) == df_joint['variant_letter_first'].map(pc5)
df_joint['AAGroup_ofer8_equiv'].describe()
print(df_joint.shape)

df_joint.head()
train = df_joint.loc[~df_joint.Class.isnull()]

test = df_joint.loc[df_joint.Class.isnull()]
train.to_csv('train_variants_featurized_raw.csv', index=False)

test.to_csv('test_variants_featurized_raw.csv', index=False)