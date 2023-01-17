! pip install wget;
! pip install dna_features_viewer;
import wget
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.SubsMat import MatrixInfo as matlist
from Bio.SeqUtils import GC
from dna_features_viewer import GraphicFeature, GraphicRecord

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

accession_numbers = [
    "MT019530",
    "AY278488",
]

genbank_base_url = 'https://www.ncbi.nlm.nih.gov/search/api/download-sequence/?db=nuccore&id='
sequences = {}
for an in accession_numbers:
    url = genbank_base_url + an
    if not os.path.isfile(an + '.fna'):
        filename = wget.download(url)
        print("Succesfully downloaded: " + filename)
    else:
        filename = an + '.fna'
    filename = filename  # "/kaggle/working/"+filename
    fasta_sequences = SeqIO.parse(open(filename), 'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        sequences.update({name: Seq(sequence)})
        print(name, sequence[:10] + "..." + sequence[-10:])
        print()

cov2 = "MT019530.1"
cov1 = "AY278488.2"
v1seq = sequences[cov2]
v2seq = sequences[cov1]
def plot_gc_content(seqs, names):
    windows = [500]#np.linspace(100, 1000, 10)
    #windows = np.round(windows)
    
    plt.figure(figsize=(16, 6))
    for seq in seqs:
        for window in windows:
            x = np.linspace(0, len(seq), int(len(seq)/window))
            x_center = []
            y = []
            for i, j in enumerate(x):
                #print(x[i-1], j)
                y.append(GC(seq[int(x[i-1]): int(j)]))
                x_center.append(int(j-(window/2)))
            y.pop(0)
            x_center.pop(0)
            sns.lineplot(x_center, y)
    #plt.legend(list(map(str, windows)))
    plt.legend(names)
    plt.title("GC Content Distrubution [Window Size = 1000]");
    
plot_gc_content([v2seq, v1seq], ["SARS-CoV-2", "SARS-CoV"])
# Index all instances of a sub sequence
# Returns a list indexes
def find_all(seq, sub_seq):
    sub_seq_len = len(sub_seq)
    result = seq.find(sub_seq)

    indexes = []
    while result != -1:
        if result != -1:
            indexes.append(result)
        result = seq.find(sub_seq, start=result + sub_seq_len)
        # print(result)
    return (indexes)


# Index all instances of a list of sub sequences
# Returns dictionary where the key is the sub sequence
# and the value is a list of indexes
def find_all_multiples(seq, sub_seqs):
    results = {}
    for sub_seq in sub_seqs:
        indexes = find_all(seq, sub_seq)
        results.update({sub_seq: indexes})
    return (results)


# Find open reading frames in a sequence given
# the stop and start codons passed to the function
# Returns a list of tuples with start and stop locations
def find_orf(seq, threshold, start_codon, stop_codons):
    orf = []
    start = seq.find(start_codon)
    start_codon_len = len(start_codon)
    while start != -1:
        stop_index = []
        for stop_codon in stop_codons:
            stop_index.append(seq.find(stop_codon, start=start + start_codon_len))
        stop = min(stop_index)
        if start != -1 and (stop - start) > threshold:
            orf.append((start+3, stop))
        last_start = start
        start = seq.find(start_codon, start=stop + 3)
        if (start < last_start):
            break
    return (orf)

# Find how many ORFs are present in a sequence (forwards and backwards) at a given minimum and maximum frame length 
def orf_threshold_counts(seq, start_codon, stop_codon, threshold_max):
    thresholds = np.linspace(10, threshold_max, 25)
    y = []
    x = []
    for threshold in thresholds:
        norm_orf = find_orf(seq, threshold, start_codon, stop_codon)
        revc_orf = find_orf(seq.reverse_complement(), threshold, start_codon, stop_codon)
        y.append(len(norm_orf) + len(revc_orf))
        x.append(threshold)
    return (x, y)
# %%
start_codon = 'ATG'
stop_codons = ['TGA', 'TAA', 'TAG']
orf_max_length = 125
v1x, v1y = orf_threshold_counts(v1seq, start_codon, stop_codons, orf_max_length)
v2x, v2y = orf_threshold_counts(v2seq, start_codon, stop_codons, orf_max_length)

plt.figure(figsize=(16, 6))
sns.scatterplot(v1x, v1y)
plot = sns.scatterplot(v2x, v2y)
#plot.set(yscale="log")
plt.grid(True,which="both",ls="--",c='gray')
plt.title('ORF Lenths')
plt.xlabel('Length (in Nucleotides)')
plt.ylabel('Total ORFs')
plt.legend(['SARS-CoV-2', 'SARS-CoV'])
plt.show()
start_codon = 'ATG'
stop_codons = ['TGA', 'TAA', 'TAG']
orf_min_length = 20
v2forf_indices = find_orf(v2seq, orf_min_length, start_codon, stop_codons)
v2rorf_indices = find_orf(v2seq.reverse_complement(), orf_min_length, start_codon, stop_codons)
v1forf_indices = find_orf(v1seq, orf_min_length, start_codon, stop_codons)
v1rorf_indices = find_orf(v1seq.reverse_complement(), orf_min_length, start_codon, stop_codons)

class OpenReadingFrame:
    def __init__(self, start, stop, seq, forward):
        self.start_index = start
        self.stop_index = stop
        self.seq = seq
        self.forward = forward
        self.length = len(seq)

v2orfs = []

for index in v2forf_indices:
    new_seq = v2seq[index[0]:index[1]]
    if len(new_seq) % 3 == 0:
        new_orf = OpenReadingFrame(index[0],index[1], new_seq, True)
        v2orfs.append(new_orf)
        

for index in v2rorf_indices:
    new_seq = v2seq.reverse_complement()[index[0]:index[1]]
    if len(new_seq) % 3 == 0:
        new_orf = OpenReadingFrame(index[0],index[1], new_seq, False)
        v2orfs.append(new_orf)

v1orfs = []

for index in v1forf_indices:
    new_seq = v1seq[index[0]:index[1]]
    if len(new_seq) % 3 == 0:
        new_orf = OpenReadingFrame(index[0],index[1], new_seq, True)
        v1orfs.append(new_orf)

for index in v1rorf_indices:
    new_seq = v1seq.reverse_complement()[index[0]:index[1]]
    if len(new_seq) % 3 == 0:
        new_orf = OpenReadingFrame(index[0],index[1], new_seq, False)
        v1orfs.append(new_orf)

v2aa = []
for orf in v2orfs:
    v2aa.append(orf.seq.translate())

v1aa = []
for orf in v1orfs:
    v1aa.append(orf.seq.translate())

v2orfs.sort(key=lambda x: x.length)
v2aa = sorted(v2aa, key=len)
with open ('proteins.txt', 'w') as f:
    for item in v2aa:
        f.write("%s\n" % item)

orf = v2orfs[-2]
#full_seq = v2seq.reverse_complement()
seq = str(orf.seq)#str(full_seq[orf.start_index:orf.stop_index])
seq = str(v2seq[orf.start_index-9:orf.stop_index+9])
record = GraphicRecord(sequence=seq, features=[
    GraphicFeature(start=len(seq)-5, end=len(seq)-6, strand=+1, color='#ffcccc', label="Stop Codon"),
    GraphicFeature(start=5, end=6, strand=+1, color='#ccccff', label="Start Codon")
])

ax, _ = record.plot(figure_width=20)
record.plot_sequence(ax)
record.plot_translation(ax, location=(9,len(seq)-9), long_form_translation=True)
plt.title("Example Detected Protein");
matrix = matlist.blosum62
max_score = 0.6
for protein1 in v1aa:
    for protein2 in v2aa:
        for alignment in pairwise2.align.globalxx(protein2, protein1):
            score = alignment.score/alignment.end
            if score >= max_score:
                max_score = score
                print("Normalized Score %.3f"%score)
                print(format_alignment(*alignment))