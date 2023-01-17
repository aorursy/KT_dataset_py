import os

import gc

import numpy as np



# Import pairwise2, SeqIO modules

from Bio import pairwise2, SeqIO



# Import format_alignment method

from Bio.pairwise2 import format_alignment



# Import new class-based Align method

from Bio import Align

from Bio.Seq import Seq



# Toy example and visualization

X = "cat"

Y = "tatctggcgtgtgtgttcccacataat"



alignments = pairwise2.align.localxx(Y, X)



# Use format_alignment method to format the alignments in the list

for a in alignments:

    print(format_alignment(*a))
print(os.listdir("../input"))



# The genome data are from NCBI:

# https://www.ncbi.nlm.nih.gov/nuccore/MN908947?report=fasta

# https://www.ncbi.nlm.nih.gov/nuccore/NC_004718.3?report=fasta



for seq_record in SeqIO.parse("../input/ncov2019/nCoV-2019.fasta", "fasta"):

    print(seq_record.description)

    print(repr(seq_record.seq))

    nCoV2019 = seq_record.seq

    print(len(seq_record))

    print("\n")

    

for seq_record in SeqIO.parse("../input/ncov2019/SARS.fasta", "fasta"):

    print(seq_record.description)

    print(repr(seq_record.seq))

    SARS = seq_record.seq

    print(len(seq_record))
print(f"nCoV-2019 has {len(nCoV2019)-len(SARS)} nucleotide bases more than SARS")
aligner = Align.PairwiseAligner()



def get_chunks(lst, n):

    """Yield successive n-sized chunks from lst."""

    for item in np.array_split(lst, n):

        yield Seq("".join(list(item)))

        

scores = list()



# Separate each sequence into 3 segments -> Total of 9 comparisons

for x in get_chunks(nCoV2019,3):

    for y in get_chunks(SARS,3):

        score = aligner.score(x, y)

        scores.append(score)
print("The mean of similar location/segment score %{:.2f}".format((scores[0]+scores[4]+scores[8])/3*100/len(SARS)*3))

print("The mean of all/mixed locations/segments score %{:.2f}".format(np.mean(scores)*100/len(SARS)*3))
print("Score matrix:")

print(np.array(scores).reshape(3,3))
# Using half segments of sequences



scores = list()

gc.collect()



for x in get_chunks(nCoV2019,2):

    for y in get_chunks(SARS,2):

        score = aligner.score(x, y)

        scores.append(score)



print("The mean of similar location/segment score %{:.2f}".format((scores[0]+scores[3])/2*100/len(SARS)*2))

print("The mean of all/mixed locations/segments score %{:.2f}".format(np.mean(scores)*100/len(SARS)*2))
print("Score matrix:")

print(np.array(scores).reshape(2,2))
# Calculate score of whole genomes

SCORE = aligner.score(nCoV2019, SARS)
print("Actual Pairwise Score %{:.2f}".format(SCORE/len(SARS)*100))