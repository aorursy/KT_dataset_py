import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import Bio

from Bio import pairwise2
# Find the best global alignment between the two sequences.

# Identical characters are given 1 point.  No points are deducted

# for mismatches or gaps.



alignments = pairwise2.align.globalxx("ACCGT", "ACG")

for a in pairwise2.align.globalxx("ACCGT", "ACG"):

    print(pairwise2.format_alignment(*a))
# Same thing as before, but with a local alignment.



for a in pairwise2.align.localxx("ACCGT", "ACG"):

     print(pairwise2.format_alignment(*a))
# Do a global alignment.  Identical characters are given 2 points,

# 1 point is deducted for each non-identical character.

for a in pairwise2.align.globalmx("ACCGT", "ACG", 2, -1):

    print(pairwise2.format_alignment(*a))
# Same as above, except now 0.5 points are deducted when opening a

# gap, and 0.1 points are deducted when extending it.

for a in pairwise2.align.globalms("ACCGT", "ACG", 2, -1, -.5, -.1):

     print(pairwise2.format_alignment(*a))
#The alignment function can also use known matrices already included in Biopython ( Bio.SubsMat -> MatrixInfo ):



from Bio.SubsMat import MatrixInfo as matlist

matrix = matlist.blosum62

for a in pairwise2.align.globaldx("KEVLA", "EVL", matrix):

     print(pairwise2.format_alignment(*a))
## Next lets try using the actual data from our dataset: ##
# First lets import our chromosomes from genome.fa



from Bio import SeqIO

count = 0

sequences = []



for seq_record in SeqIO.parse("../input/genome.fa", "fasta"):

    if (count < 6):

        sequences.append(seq_record)

        print("Id: " + seq_record.id + " \t " + "Length: " + str("{:,d}".format(len(seq_record))) )

        print(repr(seq_record.seq) + "\n")

        count = count + 1

        

chr2L = sequences[0].seq

chr2R = sequences[1].seq

chr3L = sequences[2].seq

chr3R = sequences[3].seq

chr4 = sequences[4].seq

chrM = sequences[5].seq
# Next lets grab some mRNA sequences to test from mrna-genbank.fa



count = 0

mrna_sequences = []



for seq_record in SeqIO.parse("../input/mrna-genbank.fa", "fasta"):

    if (count < 6):

        mrna_sequences.append(seq_record)

        print("Id: " + seq_record.id + " \t " + "Length: " + str("{:,d}".format(len(seq_record))) )

        print(repr(seq_record.seq) + "\n")

        count = count + 1

        

mRNA1 = mrna_sequences[0].seq

mRNA2 = mrna_sequences[1].seq

mRNA3 = mrna_sequences[2].seq

mRNA4 = mrna_sequences[3].seq

mRNA5 = mrna_sequences[4].seq

mRNA1 = sequences[5].seq