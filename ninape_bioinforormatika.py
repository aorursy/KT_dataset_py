# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import Bio

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from Bio import Entrez

Entrez.email = "nina.petreska@strudents.finki.ukim.com"

handle = Entrez.einfo()

result = handle.read()

#print(result)

handle = Entrez.einfo()

record = Entrez.read(handle)

record.keys()

record["DbList"]

handle = Entrez.einfo(db="pubmed")

record = Entrez.read(handle)

record["DbInfo"]["Description"]

record['DbInfo'].keys()



for field in record["DbInfo"]["FieldList"]:

    print("%(Name)s, %(FullName)s, %(Description)s" % field)
from urllib.request import urlopen # Python 3 only

from io import TextIOWrapper

handle = TextIOWrapper(urlopen("https://raw.githubusercontent.com/biopython/biopython/master/Tests/SwissProt/F2CXE6.txt"))

from Bio import SwissProt

record = SwissProt.read(handle)

print(record.description)
from Bio import SCOP

handle = SCOP.search(pdb=None, key=None, sid=2000014, disp=None, dir=None, loc=None, cgi='http://scop.mrc-lmb.cam.ac.uk/scop/search.cgi')

print(handle.info())
from Bio.Seq import Seq

from Bio.Alphabet import IUPAC

my_seq = Seq("ACCGACTTGCGA", IUPAC.unambiguous_dna)

print(my_seq)



#Seq("AGTACACTGGT", IUPAC.protein)

#Seq("ACCGACTTGCGA", IUPAC.unambiguous_dna)



print(my_seq[2]) #third letter

print('count',Seq("AAGAHDBUAAHAAA").count("AA"))

print('subseq',my_seq[4:8])

print('complement: ',my_seq.complement())

messenger_rna = my_seq.transcribe()

print('rna',messenger_rna)
coding_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG", IUPAC.unambiguous_dna)

coding_dna.translate()
from Bio import SeqIO

nucl_id = 'NC_005816'



handle = Entrez.efetch(db="nucleotide",

                       id=nucl_id,

                       rettype="fasta",

                       retmode="text")

yersinia = SeqIO.read(handle, "fasta")

print(yersinia.description)

print(yersinia.seq)

coding_dna_yersinia = yersinia.seq

template_dna_yersinia = coding_dna_yersinia.reverse_complement()

print(coding_dna_yersinia[0:10])

print(template_dna_yersinia[::-1][0:10])

t1 = coding_dna_yersinia.complement()

c1 = template_dna_yersinia.reverse_complement()

print(c1 == coding_dna_yersinia)

mRna = coding_dna_yersinia.translate(table="Bacterial")

cds = mRna.split('*')

print(len(cds))

c = cds[8]

c
from Bio.Data import CodonTable

b_table = CodonTable.unambiguous_dna_by_id[11]

stop_codons = b_table.stop_codons

start_codons = b_table.start_codons

print('stop',stop_codons)

print('start',start_codons)

stops = []

starts = []

for i in range(0,len(c)-3):

    codon = c[i:i+3]

    if codon in stop_codons:

        stops.append(codon)

    elif codon in start_codons:

        starts.append(codon)



print('start:',starts)

print('stop',stops)
coding_dna_yersinia.translate()