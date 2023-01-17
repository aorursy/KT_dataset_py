import Bio

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import scipy.stats as S
from Bio.SeqRecord import SeqRecord

from Bio import SeqIO

DNAseq = SeqIO.read('/kaggle/input/coronavirus-genome-sequence/MN908947.fna', "fasta")
from Bio.Data import CodonTable

print(CodonTable.unambiguous_rna_by_name['Standard'])
# Credit to Abish Pius

DNA = DNAseq.seq

mRNA = DNA.transcribe()

Amino_Acid = mRNA.translate()
Proteins = Amino_Acid.split('*')

#Remove chains smaller than 20 amino acids long

for i in Proteins[:]:

    if len(i) < 20:

        Proteins.remove(i)

Proteins
genome_dict = []

for i in range(80):

    genome_dict.append(i)
genomes = pd.DataFrame({'id': genome_dict, 'protein_id': Proteins})
genomes.protein_id = genomes.protein_id.astype('str')

genomes['unique_amino'] = genomes['protein_id'].apply(lambda x: len(set(str(x))))

genomes['num_amino'] = genomes['protein_id'].apply(lambda x: len(x))

genomes['unique_per_amino'] = genomes['unique_amino'] / genomes['num_amino']

genomes['unique_per_amino'].value_counts()
genomes
L = ['A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for x in L:

    genomes[f'{x}_count'] = genomes['protein_id'].str.count(x)

    genomes[f'{x}_per_amino'] = genomes[f'{x}_count'] / genomes['num_amino']