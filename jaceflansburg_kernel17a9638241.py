# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
strDNA = "3’-TGACCATCAATCAGCTCCACCACAGCGCTTCTCCTAGAGTTCATCAGACTAGCCTTGGAACAAACTGAGCACAAGAGCCTAGCTAGTTTAAGTTCCGCAA-5’"

result=""

for x in strDNA:

    if x=="T":

        x="A"

    elif x=="A":

        x="U"

    elif x=="C":

        x="G"

    elif x=="G":

        x="C"

    elif x=="3":

        x="5"

    elif x=="5":

        x="3"

    result=result+x

print(result)
dictionary={'UUU':'PHE','UUC':'PHE','UUA':'PHE','UUG':'PHE','UCU':'SER','UCC':

'SER','UCA':'SER','UCG':'SER','UAU':'TYR','UAC':'TYR','UAA':'STOP','UAG':'STOP',

'UGU':'CYS','UGC':'CYS','UGA':'STOP','UGG':'TRP','CUU':'LEU','CUC':'LEU','CUA':

'LEU','CUG':'LEU','CCU':'PRO','CCC':'PRO','CCA':'PRO','CCG':'PRO','CAU':'HIS',

'CAC':'HIS','CAA':'GLN','CAG':'GLN','CGU':'ARG','CGC':'ARG','CGA':'ARG','CGG':

'ARG','AUU':'ILE','AUC':'ILE','AUA':'ILE','AUG':'MET','ACU':'THR','ACC':'THR'

,'ACA':'THR','ACG':'THR','AAU':'ASN','AAC':'ASN','AAA':'LYS','AAG':'LYS','AGU'

:'SER','AGC':'SER','AGA':'ARG','AGG':'ARG','GUU':'VAL','GUC':'VAL','GUA':'VAL',

'GUG':'VAL','GCU':'ALA','GCC':'ALA','GCA':'ALA','GCG':'ALA','GAU':'ASP','GAC':

'ASP','GAA':'GLU','GAG':'GLU','GGU':'GLY','GGC':'GLY','GGA':'GLY','GGG':'GLY'}

strRNA="ACUGGUAGUUAGUCGAGGUGGUGUCGCGAAGAGGAUCUCAAGUAGUCUGAUCGGAACCUUGUUUGACUCGUGUUCUCGGAUCGAUCAAAUUCAAGGCGUU"

result=""

i = 0

while i < 97:

    s=strRNA[i: i+3]

    for key,value in dictionary.items():

        if s == key:

            s=value

    result=result+s

    i=i+3

print(result)