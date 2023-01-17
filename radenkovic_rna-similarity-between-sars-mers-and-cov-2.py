from Bio import pairwise2

from Bio.pairwise2 import format_alignment

import pandas as pd

from Bio import SeqIO

from matplotlib import pyplot as plt
# Define sequences to be aligned

SARS = SeqIO.read("/kaggle/input/coronavirus-accession-sars-mers-cov2/sars.fasta", "fasta")

MERS = SeqIO.read("/kaggle/input/coronavirus-accession-sars-mers-cov2/mers.fasta", "fasta")

COV2 = SeqIO.read("/kaggle/input/coronavirus-accession-sars-mers-cov2/cov2.fasta", "fasta")



print('Sequence Lengths:')

print('SARS:', len(SARS.seq))

print('COV2:', len(COV2.seq))

print('MERS:', len(MERS.seq))
SARS_COV = pairwise2.align.globalxx(SARS.seq, COV2.seq, one_alignment_only=True, score_only=True)

print('SARS/COV Similarity (%):', SARS_COV / len(SARS.seq) * 100)
MERS_COV = pairwise2.align.globalxx(MERS.seq, COV2.seq, one_alignment_only=True, score_only=True)

print('MERS/COV Similarity (%):', MERS_COV / len(MERS.seq) * 100)
MERS_SARS = pairwise2.align.globalxx(MERS.seq, SARS.seq, one_alignment_only=True, score_only=True)

print('MERS/SARS Similarity (%):', MERS_SARS / len(SARS.seq) * 100)
X = ['SARS/COV2', 'MERS/COV2', 'MERS/SARS']

Y = [SARS_COV/ len(SARS.seq) * 100, MERS_COV/ len(MERS.seq)*100, MERS_SARS/len(SARS.seq)*100]

plt.title('Sequence identity (%)')

plt.bar(X,Y)