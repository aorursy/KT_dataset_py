import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import Bio

print("Biopython v" + Bio.__version__)
from Bio.Seq import Seq

my_seq = Seq("AGTACACTGGT")

print(my_seq)

my_seq.alphabet
print(my_seq + " - Sequence")

print(my_seq.complement() + " - Complement")

print(my_seq.reverse_complement() + " - Reverse Complement")
from Bio import SeqIO

count = 0

sequences = [] # Here we are setting up an array to save our sequences for the next step



for seq_record in SeqIO.parse("../input/genome.fa", "fasta"):

    if (count < 6):

        sequences.append(seq_record)

        print("Id: " + seq_record.id + " \t " + "Length: " + str("{:,d}".format(len(seq_record))) )

        print(repr(seq_record.seq) + "\n")

        count = count + 1
# Lets set these sequences up for easy access later



chr2L = sequences[0].seq

chr2R = sequences[1].seq

chr3L = sequences[2].seq

chr3R = sequences[3].seq

chr4 = sequences[4].seq

chrM = sequences[5].seq
print(len(chr2L))
print("First Letter: " + chr2L[0])

print("Third Letter: " + chr2L[2])

print("Last Letter: " + chr2L[-1])
print("AAAA".count("AA"))

print(Seq("AAAA").count("AA"))
print("Length:\t" + str(len(chr2L)))

print("G Count:\t" + str(chr2L.count("G")))
print("GC%:\t\t" + str(100 * float((chr2L.count("G") + chr2L.count("C")) / len(chr2L) ) ))
from Bio.SeqUtils import GC

print("GC% Package:\t" + str(GC(chr2L)))
print("GgCcSs%:\t" + str(100 * float((chr2L.count("G") + chr2L.count("g") + chr2L.count("C") + chr2L.count("c") + chr2L.count("S") + chr2L.count("s") ) / len(chr2L) ) ))

print("GC% Package:\t" + str(GC(chr2L)))
print(chr2L[4:12])
chr2LSHORT = chr2L[0:20]

print("Short chr2L: " + chr2LSHORT)



print("Codon Pos 1: " + chr2LSHORT[0::3])

print("Codon Pos 2: " + chr2LSHORT[1::3])

print("Codon Pos 3: " + chr2LSHORT[2::3])
print("Reversed: " + chr2LSHORT[::-1])
chr2LSHORT = chr2L[0:20]

print("Short chr2L: " + chr2LSHORT)



chr2RSHORT = chr2R[0:20]

print("Short chr2R: " + chr2RSHORT)



concat = chr2LSHORT + chr2RSHORT

print("Concat: " + concat)
from Bio.Alphabet import IUPAC

protein_seq = Seq("EVRNAK", IUPAC.protein)

dna_seq = Seq("ACGT", IUPAC.unambiguous_dna)



# This will fail since they have different alphabets

# print(protein_seq + dna_seq)



# Error: Incompatible alphabets IUPACProtein() and IUPACUnambiguousDNA()



# But if we give them the same generic alphabet it works



from Bio.Alphabet import generic_alphabet

protein_seq.alphabet = generic_alphabet

dna_seq.alphabet = generic_alphabet

print(protein_seq + dna_seq)
from Bio.Alphabet import generic_dna



list_of_seqs = [Seq("ACGT", generic_dna), Seq("AACC", generic_dna), Seq("GGTT", generic_dna)]

concatenated = Seq("", generic_dna)

for s in list_of_seqs:

    concatenated += s

print(concatenated)
list_of_seqs = [Seq("ACGT", generic_dna), Seq("AACC", generic_dna), Seq("GGTT", generic_dna)]

print(sum(list_of_seqs, Seq("", generic_dna)))
dna_seq = Seq("acgtACGT", generic_dna)

print("Original: " + dna_seq)

print("Upper: " + dna_seq.upper())

print("Lower: " + dna_seq.lower())
print("GTAC" in dna_seq)

print("GTAC" in dna_seq.upper())
print("Original: " + chr2LSHORT)

print("Complement: " + chr2LSHORT.complement())

print("Reverse Complement: " + chr2LSHORT.reverse_complement())
print("Coding DNA: " + chr2LSHORT)

template_dna = chr2LSHORT.reverse_complement()

print("Template DNA: " + template_dna)
messenger_rna = chr2LSHORT.transcribe()

print("Messenger RNA: " + messenger_rna)
from Bio.Alphabet import IUPAC

messenger_rna = Seq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG", IUPAC.unambiguous_rna)

print("Messenger RNA: " + messenger_rna)

print("Protein Sequence: " + messenger_rna.translate())
coding_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG", IUPAC.unambiguous_dna)

print("Coding DNA: " + coding_dna)

print("Protein Sequence: " + coding_dna.translate())
print("Vertebrate Mitochondrial Table Result: " + coding_dna.translate(table="Vertebrate Mitochondrial"))
print ("Table 2 Result: " + coding_dna.translate(table=2))
print("Standard Translation: " + coding_dna.translate())

print("Stop as in Biology: " + coding_dna.translate(to_stop=True))

print("Table 2 Translation: " + coding_dna.translate(table=2))

print("Table 2 Translation with Stop: " + coding_dna.translate(table=2, to_stop=True))
from Bio.Seq import Seq

from Bio.Alphabet import generic_dna

gene = Seq("GTGAAAAAGATGCAATCTATCGTACTCGCACTTTCCCTGGTTCTGGTCGCTCCCATGGCAGCACAGGCTGCGGAAATTACGTTAGTCCCGTCAGTAAAATTACAGATAGGCGATCGTGATAATCGTGGCTATTACTGGGATGGAGGTCACTGGCGCGACCACGGCTGGTGGAAACAACATTATGAATGGCGAGGCAATCGCTGGCACCTACACGGACCGCCGCCACCGCCGCGCCACCATAAGAAAGCTCCTCATGATCATCACGGCGGTCATGGTCCAGGCAAACATCACCGCTAA", generic_dna)

print(gene)

print("Bacterial Translation With Stop: " + gene.translate(table="Bacterial", to_stop=True))
print ("Bacterial Translation of CDS: " + gene.translate(table="Bacterial", cds=True))
from Bio.Seq import Seq

from Bio.Alphabet import IUPAC

seq1 = Seq("ACGT", IUPAC.unambiguous_dna)

seq2 = Seq("ACGT", IUPAC.ambiguous_dna)

print(str(seq1) == str(seq2))

print(str(seq1) == str(seq1))
print(seq1 == seq2)

print(seq1 == "ACGT")