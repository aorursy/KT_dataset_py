import Bio.SeqIO

for sequence in Bio.SeqIO.parse('/kaggle/input/coronavirus-genome-sequence/MN908947.fna', "fasta"):

    print('Id: ' + sequence.id + '\nSize: ' + str(len(sequence))+' nucleotides')
sequence = '../input/coronavirus-genome-sequence/MN908947.txt'

with open(sequence) as text: 

    print (text.read(100000))