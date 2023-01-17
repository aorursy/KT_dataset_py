"""import Image
myImage = Image.open("/Users/fta/Desktop/GADataScience/Projects/SARS-CoV-2_Virus_Genetics/ProteinAnalysis/images");
myImage.show();
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
color = sns.color_palette()
from tqdm import tqdm # progress bar

# Increase default figure and font sizes for easier viewing.
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

plt.style.use('fivethirtyeight')
%matplotlib inline 
# Loading Complementary DNA Sequence into an alignable file
# pip3 install biopython

"""import sys
sys.path.append("/usr/local/lib/python3.7/site-packages")
"""

import Bio
from Bio import SeqIO
DNAseq = SeqIO.read('../input/coronavirus-genome-sequence/MN908947.fna', "fasta") # This is DNA Sequence

DNAseq
DNA = DNAseq.seq
print('DNA', DNA[:10], '...')

mRNA = DNA.transcribe()
print('RNA', mRNA[:10], '...')

print('Total Nucleotides: ', len(mRNA))

# Get amino acid sequence from mRNA (translation)
amino_acids = mRNA.translate(table = 1, cds = False)
print('Amino Acid', amino_acids[:30])
print('Total Amino Acids: ', len(amino_acids))
amino_acids
proteins = amino_acids.split('*') # * is translated stop codon
dataframe = pd.DataFrame(proteins)
dataframe.describe()

print('Total Proteins: ', len(dataframe))
dataframe.describe()
def conv(item):
    return len(item)


def to_str(item):
    return str(item)

dataframe['sequence_str'] = dataframe[0].apply(to_str)
dataframe['length'] = dataframe[0].apply(conv)

dataframe.rename(columns={0 : "sequence"}, inplace = True)
dataframe.head()
# Take only longer than 20
functional_proteins = dataframe.loc[dataframe['length'] >= 20]

print('Total functional proteins:', len(functional_proteins))
functional_proteins.describe()
# Plot lengths:

plt.figure(figsize=(20,5))

plt.subplot(111)
plt.hist(functional_proteins['length'])
plt.title('Length of Proteins (Histogram)')


# Remove extremes:

plt.figure(figsize=(20,5))
wo_extreme = functional_proteins.loc[functional_proteins['length'] < 60]
plt.subplot(121)
plt.hist(wo_extreme['length'])
plt.title('Length of Proteins (< 60)')

wo_extreme = functional_proteins.loc[functional_proteins['length'] < 1000]
plt.subplot(122)
plt.hist(wo_extreme['length'])
plt.title('Length of Proteins (< 1000)')
# Investigate large protein (<1000)

large_protein = functional_proteins.loc[functional_proteins['length'] > 2700]

seq_sample = large_protein['sequence'].tolist()[0]
print('Sequence Sample: ', '...', seq_sample[1000:1150],'...')
from Bio import pairwise2
SARS = SeqIO.read('../input/coronavirus-accession-sars-mers-cov2/sars.fasta','fasta')

MERS = SeqIO.read('../input/coronavirus-accession-sars-mers-cov2/mers.fasta','fasta')

COV2 = SeqIO.read('../input/coronavirus-accession-sars-mers-cov2/cov2.fasta','fasta')


print('Sequence Lengths:')
print('SARS:', len(SARS.seq))
print('COV2:', len(COV2.seq))
print('MERS:', len(MERS.seq))
# http://biopython.org/DIST/docs/api/Bio.pairwise2-module.html

SARS_COV = pairwise2.align.globalxx(SARS.seq, COV2.seq, one_alignment_only = True, score_only = True)
print('SARS/COV Similarity (%): ', SARS_COV / len(SARS.seq) * 100)

MERS_COV = pairwise2.align.globalxx(MERS.seq, COV2.seq, one_alignment_only = True, score_only = True)
print('MERS/COV Similarity (%): ', MERS_COV / len(MERS.seq) * 100)

MERS_SARS = pairwise2.align.globalxx(MERS.seq, SARS.seq, one_alignment_only = True, score_only = True)
print('MERS/COV Similarity (%): ', MERS_SARS / len(SARS.seq) * 100)


X = ['SARS/COV2', 'MERS/COV2', 'MERS/SARS']
Y = [SARS_COV/ len(SARS.seq) * 100, MERS_COV/ len(MERS.seq)*100, MERS_SARS/len(SARS.seq)*100]
plt.title('Sequence Identity (%)')
plt.bar(X,Y)