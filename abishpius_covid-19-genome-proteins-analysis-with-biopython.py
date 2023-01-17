# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# This code is from Paul Mooney, link here: https://www.kaggle.com/paultimothymooney/explore-coronavirus-sars-cov-2-genome

from Bio import SeqIO

for sequence in SeqIO.parse('/kaggle/input/coronavirus-genome-sequence/MN908947.fna', "fasta"):

    print('Id: ' + sequence.id + '\nSize: ' + str(len(sequence))+' nucleotides')
# Loading Complementary DNA Sequence into an alignable file

from Bio.SeqRecord import SeqRecord

from Bio import SeqIO

DNAseq = SeqIO.read('/kaggle/input/coronavirus-genome-sequence/MN908947.fna', "fasta")
DNA = DNAseq.seq

print(DNA)
#Obtain mRNA Sequence

mRNA = DNA.transcribe()

print(mRNA)
# Obtain Amino Acid Sequence from mRNA

Amino_Acid = mRNA.translate()

print(Amino_Acid)

print("Length of Protein:",len(Amino_Acid))

print("Length of Original mRNA:",len(mRNA))
from Bio.Data import CodonTable

print(CodonTable.unambiguous_rna_by_name['Standard'])
#Identify all the Proteins (chains of amino acids)

Proteins = Amino_Acid.split('*')

Proteins
#Remove chains smaller than 20 amino acids long

for i in Proteins[:]:

    if len(i) < 20:

        Proteins.remove(i)
Proteins
# Code should match proteins with online database, however kaggle is not able to connect to the URL

from Bio.Blast import NCBIWWW

result_handle = NCBIWWW.qblast("blastp", "nt", Proteins)
from Bio.SeqUtils.ProtParam import ProteinAnalysis

MW = []

aromaticity =[]

AA_Freq = []

IsoElectric = []

for j in Proteins[:]:

    a = ProteinAnalysis(str(j))

    MW.append(a.molecular_weight())

    aromaticity.append(a.aromaticity())

    AA_Freq.append(a.count_amino_acids())

    IsoElectric.append(a.isoelectric_point())



MW = pd.DataFrame(data = MW,columns = ["Molecular Weights"] )

MW.head()
# Plot Molecular Weights Distribution

sns.set_style('whitegrid');

plt.figure(figsize=(10,6));

sns.distplot(MW,kde=False);

plt.title("SARS-CoV-2 Protein Molecular Weights Distribution");
MW.idxmax()

print(Proteins[48])

len(Proteins[48])
# Protein of Interest

POI = AA_Freq[48]
plt.figure(figsize=(10,6));

plt.bar(POI.keys(), list(POI.values()), align='center')
print('The aromaticity % is ',aromaticity[48])

print('The Isoelectric point is', IsoElectric[48])