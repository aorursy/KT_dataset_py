from Bio import SeqIO
for record in SeqIO.parse("/kaggle/input/sars-cov2.fasta", "fasta"):
    print(record.seq)