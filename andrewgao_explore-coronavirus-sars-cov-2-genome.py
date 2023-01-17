import Bio.SeqIO

for sequence in Bio.SeqIO.parse('/kaggle/input/coronavirus-genome-sequence/MN908947.fna', "fasta"):

    print('Id: ' + sequence.id + '\nSize: ' + str(len(sequence))+' nucleotides')
sequence = '../input/coronavirus-genome-sequence/MN908947.txt'

with open(sequence) as text: 

    sequencestring = text.read(500)

    sequencestring = sequencestring[96:]

sequencestring = sequencestring.replace('\n','')



print(sequencestring)

cDNA = Seq(sequencestring)

RNA = cDNA.transcribe()

protein = RNA.translate()

print(protein)

print(len(protein))