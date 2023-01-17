! pip install wget
# Lets find a way to automatically download data.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import wget

import os

from Bio import SeqIO

from Bio.Seq import Seq

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



accession_numbers = [

    "NC_001416",

    "NC_012920",

    "NC_001643",

    "NC_001665",

    "NC_001807",

    "NC_001643",

    "NC_005089",

    "NC_000907",

]



genbank_base_url = 'https://www.ncbi.nlm.nih.gov/search/api/download-sequence/?db=nuccore&id='

sequences = {}

for an in accession_numbers:

    url = genbank_base_url + an

    if not os.path.isfile(an+'.fna'): 

        filename = wget.download(url)

        print("Succesfully downloaded: "+filename)

    else:

        filename = an+'.fna'

    filename = "/kaggle/working/"+filename

    fasta_sequences = SeqIO.parse(open(filename),'fasta')

    for fasta in fasta_sequences:

        name, sequence = fasta.id, str(fasta.seq)

        sequences.update({name: Seq(sequence)})

        print(name, sequence[:10]+"..."+sequence[-10:])
def plot_gc_content(seq, name):

    windows = np.linspace(100, 1000, 10)

    windows = np.round(windows)

    plt.figure(figsize=(16, 6))

    for window in windows:

        x = np.linspace(0, len(seq), int(len(seq)/window))

        x_center = []

        y = []

        for i, j in enumerate(x):

            #print(x[i-1], j)

            y.append(seq[int(x[i-1]): int(j)].count('GC'))

            x_center.append(int(j-(window/2)))

        y.pop(0)

        x_center.pop(0)

        sns.lineplot(x_center, y)

    plt.legend(list(map(str, windows)))

    plt.title(name+" GC Content vs Window Size");



plot_gc_content(sequences['NC_001416.1'], "Lambda")

mtHuman = 'NC_012920.1'

mtChimp = 'NC_001643.1'

plot_gc_content(sequences[mtHuman], "mtHuman")

plot_gc_content(sequences[mtChimp], "mtChimp")
# Index all instances of a sub sequence

# Returns a list indexes

def find_all(seq, sub_seq):

    sub_seq_len = len(sub_seq)

    result = seq.find(sub_seq)

    

    indexes = []

    while result != -1:

        if result != -1:

            indexes.append(result)

        result = seq.find(sub_seq, start=result+sub_seq_len)

        #print(result)

    return(indexes)



# Index all instances of a list of sub sequences

# Returns dictionary where the key is the sub sequence

# and the value is a list of indexes

def find_all_multiples(seq, sub_seqs):

    results = {}

    for sub_seq in sub_seqs:

        indexes = find_all(seq, sub_seq)

        results.update({sub_seq: indexes})

    return(results)

        



# Find open reading frames in a sequence given

# the stop and start codons hardcoded in the function

# Returns a list of tuples with start and stop locations 

def find_orf(seq, threshold, start_codon, stop_codons):



    orf = []

    start = seq.find(start_codon)

    start_codon_len = len(start_codon)

    while start != -1:

        stop_index = []

        for stop_codon in stop_codons:

            stop_index.append(seq.find(stop_codon, start=start+start_codon_len))

        stop = min(stop_index)

        if start != -1 and (stop-start) > threshold:

            orf.append((start, stop))

        last_start = start

        start = seq.find(start_codon, start=stop+3)

        if(start < last_start):

            break

    return(orf)



def orf_threshold_counts(seq, start_codon, stop_codon, threshold_max):

    thresholds = np.linspace(10, threshold_max, 25)

    y = []

    x = []

    for threshold in thresholds:

        norm_orf = find_orf(seq, threshold, start_codon, stop_codon)

        revc_orf = find_orf(seq.reverse_complement(), threshold, start_codon, stop_codon)

        y.append(len(norm_orf)+len(revc_orf))

        x.append(threshold)

    return(x,y)

    



mtHuman = "NC_001807.4"

mtChimp = "NC_001643.1"

mtMouse = "NC_005089.1"



mt_start_codon = 'ATG'

mt_stop_codons = ['TAG', 'TAA', 'AGA', 'AGG']



hx,hy = orf_threshold_counts(sequences[mtHuman], mt_start_codon, mt_stop_codons, 80)

cx,cy = orf_threshold_counts(sequences[mtChimp], mt_start_codon, mt_stop_codons, 80)

mx,my = orf_threshold_counts(sequences[mtMouse], mt_start_codon, mt_stop_codons, 80)



plt.figure(figsize=(16, 6))

sns.scatterplot(hx,hy)

sns.scatterplot(cx,cy)

plot = sns.scatterplot(mx,my)

#plot.set(yscale="log")

plt.title('Mitochondrial ORF Lenths')

plt.xlabel('Length (in Nucleotides)')

plt.ylabel('Total ORFs')

plt.legend(['Human', 'Chimp', 'Mouse'])
# Randomly generate sequence with same nucleotide distribution

seq = sequences[mtMouse]

pA = seq.count('A')/len(seq)

pT = seq.count('T')/len(seq)

pC = seq.count('C')/len(seq)

pG = seq.count('G')/len(seq)



probs = [pA, pT, pC, pG]

choices = ['A', 'T', 'C', 'G']



random_seq = Seq(''.join(np.random.choice(choices, len(seq), probs)))



#rnorm_orf = find_orf(random_seq, threshold, mt_start_codon, mt_stop_codons)

#rrevc_orf = find_orf(random_seq.reverse_complement(), threshold,  mt_start_codon, mt_stop_codons)



# Plot comparison

x,y = orf_threshold_counts(seq, mt_start_codon, mt_stop_codons, 80)

x_rand,y_rand = orf_threshold_counts(random_seq, mt_start_codon, mt_stop_codons, 80)



plt.figure(figsize=(16, 6))

sns.scatterplot(x,y)

sns.scatterplot(x_rand,y_rand)

plt.xlabel('ORF Length (in Nucleotides)')

plt.legend(['Actual Seq', 'Randomly Generated Seq'])

plt.title('ORF Length Distribution')

plt.show()
infl = "NC_000907.1"



start_codon = 'ATG'

stop_codons = ['TAG', 'TGA', 'TAA']



x,y = orf_threshold_counts(sequences[infl], start_codon, stop_codons, 80)



plt.figure(figsize=(16, 6))

sns.scatterplot(x,y)

plt.title('Influenza ORF Lenths')

plt.xlabel('Length (in Nucleotides)')

plt.ylabel('Total ORFs')
print(len(sequences[infl]))

print(len(sequences[mtHuman]))