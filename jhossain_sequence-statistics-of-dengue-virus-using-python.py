# for sequence analysis 
from collections import Counter
import Bio 
from skbio import Sequence
from skbio.sequence.distance import hamming
from Bio import SeqIO
import pandas as pd
import numpy as np

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# set figuresize and fontsize
plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['font.size'] = 14

# data files 
import os 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# examine first few lines of den-1 
!head /kaggle/input/dengue-virus-4-complete-genomes/den1.fasta
# examine last few lines of den-1 
!tail /kaggle/input/dengue-virus-4-complete-genomes/den1.fasta
# examine first few lines of den-2 
!head /kaggle/input/dengue-virus-4-complete-genomes/den2.fasta
# examine last few lines of den-2 
!tail /kaggle/input/dengue-virus-4-complete-genomes/den2.fasta
# examine first few lines of den-3 
!head /kaggle/input/dengue-virus-4-complete-genomes/den3.fasta
# examine last few lines of den-1 
!tail /kaggle/input/dengue-virus-4-complete-genomes/den3.fasta
# examine first few lines of den-4
!head /kaggle/input/dengue-virus-4-complete-genomes/den4.fasta
# examine last few lines of den-1 
!tail /kaggle/input/dengue-virus-4-complete-genomes/den4.fasta
def readFASTA(inputfile): 
    """Reads a sequence file and returns as string"""
    with open(inputfile, "r") as seqfile:
        # skip the name line 
        seq = seqfile.readline()
        seq = seqfile.read()
        seq = seq.replace("\n", "")
        seq = seq.replace("\t", "") 
    return seq 
# load sequence 
den1 = readFASTA('/kaggle/input/dengue-virus-4-complete-genomes/den1.fasta')
den2 = readFASTA('/kaggle/input/dengue-virus-4-complete-genomes/den2.fasta')
den3 = readFASTA('/kaggle/input/dengue-virus-4-complete-genomes/den3.fasta')
den4 = readFASTA('/kaggle/input/dengue-virus-4-complete-genomes/den4.fasta')
print("Length of DEN1: ", len(den1))
print("Length of DEN2: ", len(den2))
print("Length of DEN3: ", len(den3))
print("Length of DEN4: ", len(den4))
from collections import Counter
def basecount_fast(seq): 
    """"Count the frequencies of each bases in sequence including every letter""" 
    freqs = Counter(seq)
    return freqs
print("Frequency of DEN1: ", basecount_fast(den1))
print("Frequency of DEN2: ", basecount_fast(den2))
print("Frequency of DEN3: ", basecount_fast(den3))
print("Frequency of DEN4: ", basecount_fast(den4))
def ntFrequency(seq, useall=False, calpc=False):
    """Count the frequencies of each bases in sequence including every letter"""
    length = len(seq)
    if calpc:
        # Make a dictionary "freqs" to contain the frequency(in % ) of each base.
        freqs = {}
    else:
        # Make a dictionary "base_counts" to contain the frequency(whole number) of each base.
        base_counts = {}
    if useall:
        # If we want to look at every letter that appears in the sequence.
        seqset = set(seq)
    else:
        # If we just want to look at the four bases A, T, C, G
        seqset = ("A", "T", "G", "C")

    for letter in seqset:
        num = seq.count(letter)
        if calpc:
            # The frequency is calculated out of the total sequence length, even though some bases are not A, T, C, G
            freq = round(num/length, 2)
            freqs[letter] = freq
        else:
            # Contain the actual number of bases.
            base_counts[letter] = num
    if calpc:
        return freqs
    else:
        return base_counts
print("Frequency of DEN1: ", ntFrequency(den1, useall=True))
print("Frequency of DEN2: ", ntFrequency(den2,  useall=True))
print("Frequency of DEN3: ", ntFrequency(den3,  useall=True))
print("Frequency of DEN4: ", ntFrequency(den4,  useall=True))
print("Percentage of DEN1: ", ntFrequency(den1, calpc=True))
print("Percentage of DEN2: ", ntFrequency(den2, calpc=True))
print("Percentage of DEN3: ", ntFrequency(den3, calpc=True))
print("Percentage of DEN4: ", ntFrequency(den4, calpc=True))
def plotNucleotideFrequency(seq, title= False, xlab="Bases",ylab="Frequency", kind=None):
    """Plots the Nucleotide Frequency"""
    if kind == 'bar':
        freq = ntFrequency(seq)
        df = pd.DataFrame(freq.items(), columns = ['letters', 'frequency'])
        sns.barplot(x='letters', y='frequency', data=df)
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.tight_layout()
        plt.show()
    elif kind == 'pie':
        freq = ntFrequency(seq)
        plt.pie(freq.values(), labels=freq.keys(), autopct='%1.1f%%', shadow=True)
        plt.tight_layout()
        plt.show()
    else:
        print("Please select your visualization type either bar or pie chart!")
# frequency distribution of den1 
plotNucleotideFrequency(den1, "Nucleotide Frequency Distribution of DEN1", kind='bar')
# frequency distribution of den2
plotNucleotideFrequency(den2, "Nucleotide Frequency Distribution of DEN2", kind='bar')
# frequency distribution of den3
plotNucleotideFrequency(den3, "Nucleotide Frequency Distribution of DEN3", kind='bar')
# frequency distribution of den4
plotNucleotideFrequency(den4, "Nucleotide Frequency Distribution of DEN4", kind='bar')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
# ax1 
freq1 = ntFrequency(den1)
ax1.bar(freq1.keys(), freq1.values())
ax1.set_title("DEN1")
ax1.set_ylabel("Frequency")

# ax2 
freq2 = ntFrequency(den2)
ax2.bar(freq2.keys(), freq2.values())
ax2.set_title("DEN2")
ax2.set_ylabel("Frequency")

# ax3
freq3 = ntFrequency(den3)
ax3.bar(freq3.keys(), freq3.values())
ax3.set_title("DEN3")
ax3.set_xlabel("Bases")
ax3.set_ylabel("Frequency")

# ax4 
freq4 = ntFrequency(den4)
ax4.bar(freq4.keys(), freq4.values())
ax4.set_title("DEN4")
ax4.set_xlabel("Bases")
ax4.set_ylabel("Frequency")

# layout
plt.tight_layout()
# plt.savefig('../output_figs/den_plot.png')
plt.show() 
# pie chart of den1
plotNucleotideFrequency(den1, "Nucleotide Frequency Distribution of DEN1", kind='pie')
# pie chart of den2
plotNucleotideFrequency(den2, "Nucleotide Frequency Distribution of DEN2", kind='pie')
# pie chart of den3
plotNucleotideFrequency(den3, "Nucleotide Frequency Distribution of DEN3", kind='pie')
# pie chart of den4
plotNucleotideFrequency(den4, "Nucleotide Frequency Distribution of DEN4", kind='pie')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
# ax1 
freq1 = ntFrequency(den1)
ax1.pie(freq1.values(), labels=freq1.keys(), autopct='%1.1f%%', shadow=True)
ax1.set_title("DEN1")
# ax2 
freq2 = ntFrequency(den2)
ax2.pie(freq2.values(), labels=freq2.keys(), autopct='%1.1f%%', shadow=True)
ax2.set_title("DEN2")

# ax3
freq3 = ntFrequency(den3)
ax3.pie(freq3.values(), labels=freq3.keys(), autopct='%1.1f%%', shadow=True)
ax3.set_title("DEN3")

# ax4 
freq4 = ntFrequency(den4)
ax4.pie(freq4.values(), labels=freq4.keys(), autopct='%1.1f%%', shadow=True)
ax4.set_title("DEN4")

# layout
plt.tight_layout()
# plt.savefig('../output_figs/den_plot.png')
plt.show() 
def calculateGC(seq):
    """
    Take DNA sequence as input and calculate the GC content.
    """
    no_of_g = seq.count("G")
    no_of_c = seq.count("C")
    total = no_of_g + no_of_c
    gc = round(total/len(seq) * 100, 2)
    return gc
print("GC Content of DEN1:", calculateGC(den1))
print("GC Content of DEN2:", calculateGC(den2))
print("GC Content of DEN3:", calculateGC(den3))
print("GC Content of DEN4:", calculateGC(den4))
def calculateAT(seq):
    """Take DNA sequence as input and calculate the AT content."""
    no_of_a = seq.count("A")
    no_of_t = seq.count("T")
    total = no_of_a + no_of_t
    at = round(total/len(seq) * 100, 2)
    return at

print("AT Content of DEN1:", calculateAT(den1))
print("AT Content of DEN2:", calculateAT(den2))
print("AT Content of DEN3:", calculateAT(den3))
print("AT Content of DEN4:", calculateAT(den4))
def subSeqGC(seq, window=300):
    """Returns sub-sequence GC distribution"""
    res = [] 
    for i in range(0, len(seq)-window+1, window):
        subseq = seq[i:i+window]
        gc = calculateGC(subseq)
        res.append(gc)
    return res
gc1 = subSeqGC(den1, window=300)
gc2 = subSeqGC(den2, window=300)
gc3 = subSeqGC(den3, window=300)
gc4 = subSeqGC(den4, window=300)
def plotGCDistribution(seq, title="GC Distribution of Sub-sequence", xlab="Ranges", ylab="% GC"):
    """Plots the GC content along a sequence using a sliding window"""
    gc = subSeqGC(seq, 10000)
    sns.lineplot(range(len(gc)), sorted(gc))
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
# ax1 
gc1 = subSeqGC(den1, 300)
ax1.plot(range(len(gc1)), gc1)
ax1.set_title("DEN1")
ax1.set_ylabel("% GC")

# ax2 
gc2 = subSeqGC(den2, 300)
ax2.plot(range(len(gc2)), gc2)
ax2.set_title("DEN2")
ax2.set_ylabel("% GC")

# ax3
gc3 = subSeqGC(den3, 300)
ax3.plot(range(len(gc3)), gc3)
ax3.set_title("DEN3")
ax3.set_xlabel("Base-pair Position")
ax3.set_ylabel("% GC")

# ax4 
gc4 = subSeqGC(den4, 300)
ax4.plot(range(len(gc4)), gc4)
ax4.set_title("DEN4")
ax4.set_xlabel("Base-pair Position")
ax4.set_ylabel("% GC")

# layout
plt.tight_layout()
# plt.savefig('../output_figs/den_plot.png')
plt.show() 
def subSeqAT(seq, window=1000):
    """Returns sub-sequence GC distribution"""
    res = []
    for i in range(0, len(seq)-window+1, window):
        subseq = seq[i:i+window]
        gc = calculateAT(subseq)
        res.append(gc)
    return res
at1 = subSeqAT(den1, window=300)
at2 = subSeqAT(den2, window=300)
at3 = subSeqAT(den3, window=300)
at = subSeqAT(den4, window=300)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
# ax1 
at1 = subSeqAT(den1, 300)
ax1.plot(range(len(at1)), at1)
ax1.set_title("DEN1")
ax1.set_ylabel("% AT")

# ax2 
at2 = subSeqAT(den2, 300)
ax2.plot(range(len(at2)), at2)
ax2.set_title("DEN2")
ax2.set_ylabel("% AT")

# ax3
at3 = subSeqAT(den3, 300)
ax3.plot(range(len(at3)), at3)
ax3.set_title("DEN3")
ax3.set_xlabel("Base-pair Position")
ax3.set_ylabel("% AT")

# ax4 
at4 = subSeqAT(den4, 300)
ax4.plot(range(len(at4)), at4)
ax4.set_title("DEN4")
ax4.set_xlabel("Base-pair Position")
ax4.set_ylabel("% AT")

# layout
plt.tight_layout()
# plt.savefig('../output_figs/den_plot.png')
plt.show() 
def buildKmers(sequence, ksize):
    """Returns k-mers on the basis of ksize."""
    kmers = []
    n_kmers = len(sequence) - ksize + 1
    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)
    return kmers
# build dimers 
km1 = buildKmers(den1, 2)
km2 = buildKmers(den2, 2)
km3 = buildKmers(den3, 2)
km4 = buildKmers(den4, 2)
# dimer frequency
def kmerFrequency(seq):
    """Returns frequencies of kmers"""
    counts = Counter(seq)
    return counts
print("Dimer Frequency of DEN1:\n", kmerFrequency(km1))
print("Dimer Frequency of DEN2:\n", kmerFrequency(km2))
print("Dimer Frequency of DEN3:\n", kmerFrequency(km3))
print("Dimer Frequency of DEN4:\n", kmerFrequency(km4))
def plotKmerFrequency(seq, title=False, xlab='Dimer', ylab='Frequency', kind=None):
    """Plots the kmers frequency"""
    freq = kmerFrequency(seq)
    df = pd.DataFrame(freq.items(), columns = ['letters', 'frequency'])
    if kind == 'bar':
        sns.barplot(x='letters', y='frequency', data=df)
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.tight_layout()
        plt.show()
    elif kind == 'pie':
        plt.pie(freq.values(), labels=freq.keys(), autopct='%1.1f%%', shadow=True)
        plt.tight_layout()
        plt.show()
    else:
        print("Please select your visualization type either bar or pie chart!")

# dimer frequency of den1
plotKmerFrequency(km1, "Dimer Frequency of DEN1", kind='bar')
# dimer frequency of den2
plotKmerFrequency(km2, "Dimer Frequency of DEN2", kind='bar')
# dimer frequency of den3
plotKmerFrequency(km3, "Dimer Frequency of DEN3", kind='bar')
# dimer frequency of den4
plotKmerFrequency(km4, "Dimer Frequency of DEN4", kind='bar')
# k-mer frequency of den1 using pie chart
plotKmerFrequency(km1, kind='pie')
# k-mer frequency of den2 using pie chart
plotKmerFrequency(km2,kind='pie')
# k-mer frequency of den3 using pie chart
plotKmerFrequency(km3,kind='pie')
# k-mer frequency of den4 using pie chart
plotKmerFrequency(km4,kind='pie')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)

# ax1 
freq1 = kmerFrequency(km1)
ax1.bar(freq1.keys(), freq1.values())
ax1.set_title("DEN1")
ax1.set_ylabel("Frequency")
# ax2 
freq2 = kmerFrequency(km2)
ax2.bar(freq2.keys(), freq2.values())
ax2.set_title("DEN2")
ax2.set_ylabel("Frequency")

# ax3
freq3 = kmerFrequency(km3)
ax3.bar(freq3.keys(), freq3.values())
ax3.set_title("DEN3")
ax3.set_xlabel("Bases")
ax3.set_ylabel("Frequency")

# ax4 
freq4 = kmerFrequency(km4)
ax4.bar(freq4.keys(), freq4.values())
ax4.set_title("DEN4")
ax4.set_xlabel("Bases")
ax4.set_ylabel("Frequency")

# layout
plt.tight_layout()
# plt.savefig('../output_figs/den_plot.png')
plt.show() 
# using python 
def hamming_distance(seq1, seq2): 
    """Returns hamming distance between 2 sequences"""
    return len([(x,y) for x, y in zip(seq1, seq2) if x != y])
print("Hamming Distance of DEN1 and DEN2:", hamming_distance(den1[1:200], den2[1:200]))
print("Hamming Distance of DEN1 and DEN3:", hamming_distance(den1[1:200], den3[1:200]))
print("Hamming Distance of DEN1 and DEN4:", hamming_distance(den1[1:200], den4[1:200]))
print("Hamming Distance of DEN2 and DEN3:", hamming_distance(den2[1:200], den3[1:200]))
print("Hamming Distance of DEN2 and DEN4:", hamming_distance(den2[1:200], den4[1:200]))
print("Hamming Distance of DEN3 and DEN4:", hamming_distance(den3[1:200], den4[1:200]))
# using scikit-bio library
def calculateHammingDistance(seq1, seq2):
    """Returns hamming distance between two equal length sequences"""
    seq1 = Sequence(seq1)
    seq2 = Sequence(seq2)
    result = hamming(seq1, seq2)
    return result
print("Hamming Distance of DEN1 and DEN2:", round(calculateHammingDistance(den1[1:200], den2[1:200]), 2))
print("Hamming Distance of DEN1 and DEN3:", round(calculateHammingDistance(den1[1:200], den3[1:200]), 2))
print("Hamming Distance of DEN1 and DEN4:", round(calculateHammingDistance(den1[1:200], den4[1:200]), 2))
print("Hamming Distance of DEN2 and DEN3:", round(calculateHammingDistance(den2[1:200], den3[1:200]), 2))
print("Hamming Distance of DEN2 and DEN4:", round(calculateHammingDistance(den2[1:200], den4[1:200]), 2))
print("Hamming Distance of DEN3 and DEN4:", round(calculateHammingDistance(den3[1:200], den4[1:200]), 2))

def __delta(x,y):
    return 0 if x == y else 1

def __M(seq1,seq2,i,j,k):
    return sum(__delta(x,y) for x,y in zip(seq1[i:i+k],seq2[j:j+k]))

def __makeMatrix(seq1,seq2,k):
    n = len(seq1)
    m = len(seq2)
    return [[__M(seq1,seq2,i,j,k) for j in range(m-k+1)] for i in range(n-k+1)]


def __plotMatrix(__M,t, seq1, seq2, nonblank = chr(0x25A0), blank = ' '):
    print(' |' + seq2)
    print('-'*(2 + len(seq2)))
    for label,row in zip(seq1,M):
        line = ''.join(nonblank if s < t else blank for s in row)
        print(label + '|' + line)

def dotPlot(seq1,seq2):
    """Create a dotplot for checking sequence similarity"""
    plt.imshow(np.array(__makeMatrix(seq1,seq2,1)), cmap='viridis', interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.clim(-1, 1)
    # on x-axis list all sequences of seq 2
    xt=plt.xticks(np.arange(len(list(seq2))),list(seq2))
    # on y-axis list all sequences of seq 1
    yt=plt.yticks(np.arange(len(list(seq1))),list(seq1))
    plt.show()

# 20x20 matrix of den1 and den2 
dotPlot(den1[1:21], den2[1:21])
# 20x20 matrix of den1 and den3 
dotPlot(den1[20:41], den3[20:41])
# 20x20 matrix of den1 and den4
dotPlot(den1[40:61], den4[40:61])
def __ntDensityPlot(xvar,ydict,xlab,ylab):
    """Makes a scatterplot of y-variable(s) against an x-variable"""
    yvarnames = []
    for yvar in ydict:
        yvarnames.append(yvar)
        sns.lineplot(xvar,ydict[yvar])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(yvarnames, loc="upper right")
    plt.show()
def ntDensityOne(seq,windowsize,verbose=False,jumpsize=1000,makePlot=True):
    """Plots the base frequency along a sequence using a sliding window"""
    length = len(seq)
    # make a dictionary to contain four empty lists
    freqs = { "A": [], "T": [], "G": [], "C": [] }
    myset = ("A", "C", "T", "G")
    midpoints = []
    # Move the window by jumpsize bp at a time.
    for i in range(0,length-windowsize+1,jumpsize):
        subseq = seq[i:i+windowsize]
        if verbose:
            start = i
            end = i+windowsize
            print("start %d end %d subseq is %s length %d windowsize %d" % (start,end,subseq,length,windowsize))
        assert len(subseq)==windowsize, "ERROR: ntdensity2: length of subseq is not windowsize"
        for letter in myset:
            num = subseq.count(letter)
            pc = 100 * num/windowsize
            freqs[letter].append(pc)
        # Find the mid-point of the window:
        # For example, if the window is from i=1000 to i=11000,
        # midpoint = 12000/2 = 6000
        midpoint = (i + i + windowsize)/2
        midpoints.append(midpoint)
    if makePlot:
        # Call the plotting function
        midpoints2 = [x/1000 for x in midpoints] # list comprehension
        __ntDensityPlot(midpoints2,freqs,'Base-Pair Position (kb)','% of Nucleotide') # Convert to kb for plotting

# nucleotide density of den1 
ntDensityOne(den1, windowsize=2000)
# nucleotide density of den2 
ntDensityOne(den2, windowsize=2000)
# nucleotide density of den3 
ntDensityOne(den3, windowsize=2000)
# nucleotide density of den4 
ntDensityOne(den4, windowsize=2000)
def ntDensityTwo(seq,windowsize,verbose=False,jumpsize=1000,makePlot=True):
    """Plots the G+C content along a sequence using a sliding window"""
    length = len(seq)
    # Make a dictionary to contain two empty lists
    freqs = { "G+C": [], "A+T": [] }
    myset = ("A+T", "G+C")
    # Note: instead of storing G+C, A+T in a hash, could just have coded G+C=0, A+T=1, and stored these values in arrays.
    midpoints = []
    # Move the window by jumpsize bp at a time.
    # The first window we look at is from i=0 to i=windowsize.
    # For example, if the sequence is 30000 bases long, windowsize=10000.
    # In the first loop, we look at i=0 to i=10000.
    # In the second loop, we look at i=1000 to i=11000. ...
    # In the last loop, we look at i=29000 to i=30000.
    # Note: for i = range(0,10) goes from i=0...9.
    for i in range(0,length-windowsize+1,jumpsize):
        subseq = seq[i:i+windowsize]
        if verbose:
            start = i
            end = i+windowsize
            print("start %d end %d subseq is %s length %d windowsize %d" % (start,end,subseq,length,windowsize))
        assert len(subseq)==windowsize, "ERROR: ntdensity1: length of subseq is not windowsize"
        for dimer in myset:
            letter1 = dimer[0:1]
            letter2 = dimer[2:3]
            num1 = subseq.count(letter1)
            num2 = subseq.count(letter2)
            num = num1 + num2
            pc = (100 * num)/windowsize
            freqs[dimer].append(pc)
        # Find the mid-point of the window:
        # For example, if the window is from i=1000 to i=11000,
        # midpoint = 12000/2 = 6000
        midpoint = (i + i + windowsize)/2
        midpoints.append(midpoint)
    if makePlot:
        # Call the plotting function
        midpoints2 = [x/1000 for x in midpoints] # list comprehension
        __ntDensityPlot(midpoints2,freqs,'Base-Pair Position (kb)','%  of Nucleotide') # Convert to kb for plotting

# dimer density of den1 
ntDensityTwo(den1, windowsize=2000)
# dimer density of den2 
ntDensityTwo(den2, windowsize=2000)
# dimer density of den3 
ntDensityTwo(den3, windowsize=2000)
# dimer density of den4 
ntDensityTwo(den4, windowsize=2000)
# load sequence as biopython
s1 = SeqIO.read('/kaggle/input/dengue-virus-4-complete-genomes/den1.fasta', "fasta")
s2 = SeqIO.read('/kaggle/input/dengue-virus-4-complete-genomes/den2.fasta', "fasta")
s3 = SeqIO.read('/kaggle/input/dengue-virus-4-complete-genomes/den3.fasta', "fasta")
s4 = SeqIO.read('/kaggle/input/dengue-virus-4-complete-genomes/den4.fasta', "fasta")
# get sequence object 
seq1 = s1.seq 
seq2 = s2.seq 
seq3 = s3.seq 
seq4 = s4.seq 
# translation
prt1 = seq1.translate()
prt2 = seq2.translate()
prt3 = seq3.translate()
prt4 = seq4.translate()
# examine few lines 
prt1[1:500]
# examine few lines 
prt2[1:500]
# examine few lines 
prt3[1:500]
# examine few lines 
prt4[1:500]
print("Protein Length of DEN1: ", len(prt1))
print("Protein Length of DEN2: ", len(prt2))
print("Protein Length of DEN3: ", len(prt3))
print("Protein Length of DEN4: ", len(prt4))
def proteinFrequency(seq):
    """Count the frequencies of each protein in sequence including every letter"""
    prt_freq = Counter(seq)
    return prt_freq
print("Protein Frequency of DEN1:\n ", proteinFrequency(prt1))
print("Protein Frequency of DEN2:\n ", proteinFrequency(prt2))
print("Protein Frequency of DEN3:\n ", proteinFrequency(prt3))
print("Protein Frequency of DEN4:\n ", proteinFrequency(prt4))
print("Most Common Amino Acids in DEN1:\n ", Counter(prt1).most_common(10))
print("Most Common Amino Acids in DEN2:\n ", Counter(prt2).most_common(10))
print("Most Common Amino Acids in DEN3:\n ", Counter(prt3).most_common(10))
print("Most Common Amino Acids in DEN4:\n ", Counter(prt4).most_common(10))
def plotProteinFrequency(seq, title=False, xlab="Proteins",ylab="Frequency", kind=None):
    """Makes a scatterplot of y-variable(s) against an x-variable"""
    if kind == 'bar':
        freq = Counter(seq)
        df = pd.DataFrame(freq.items(), columns = ['letters', 'frequency'])
        sns.barplot(x='letters', y='frequency', data=df)
        plt.title(title,  fontsize=14)
        plt.xlabel(xlab,  fontsize=14)
        plt.ylabel(ylab,  fontsize=14)
        plt.tight_layout()
        plt.show()
    elif kind == 'pie':
        freq = Counter(seq)
        plt.pie(freq.values(), labels=freq.keys(), autopct='%1.1f%%', shadow=True)
        plt.tight_layout()
        plt.show()
    else:
        print("Please select your visualization type either bar or pie chart!")

# protein frequency distribution of den1 
plotProteinFrequency(prt1, "DEN1 Protein Frequency Distribution", kind='bar')
# protein frequency distribution of den2
plotProteinFrequency(prt2, "DEN2 Protein Frequency Distribution", kind='bar')
# protein frequency distribution of den3
plotProteinFrequency(prt3, "DEN3 Protein Frequency Distribution", kind='bar')
# protein frequency distribution of den4
plotProteinFrequency(prt4, "DEN4 Protein Frequency Distribution", kind='bar')
# dotplot of den1 protein and den2 
dotPlot(prt1[1:21], prt1[1:21])
# dotplot of den1 protein and den3
dotPlot(prt1[1:21], prt3[1:21])
# dotplot of den1 protein and den3
dotPlot(prt1[1:21], prt4[1:21])
# import sequence alignment library
from Bio import pairwise2 
from Bio.pairwise2 import format_alignment
# alignments 
alignments = pairwise2.align.globalxx(seq1, seq2)
# print alignments 
print(alignments)
# to see it weel 
print(format_alignment(*alignments[0]))
# to see 2nd part 
print(format_alignment(*alignments[1]))
# to see all alignments 
for a in alignments: 
    print(format_alignment(*a))
# get only the score for alignments 
alignment_scores = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True, score_only=True)
alignment_scores
# check for the similarity 
alignment_scores/len(seq1) * 100
# sequence similarity dengue virus 
alignment_scores1 = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True, score_only=True)
alignment_scores2 = pairwise2.align.globalxx(seq1, seq3, one_alignment_only=True, score_only=True)
alignment_scores3 = pairwise2.align.globalxx(seq1, seq4, one_alignment_only=True, score_only=True)
print("Similarity between DEN1 and DEN2:", alignment_scores1/len(seq1)* 100)
print("Similarity between DEN1 and DEN3:", alignment_scores1/len(seq1)* 100)
print("Similarity between DEN1 and DEN4:", alignment_scores1/len(seq1)* 100)
# sequence similarity dengue virus 
alignment_scores1 = pairwise2.align.globalxx(seq2, seq3, one_alignment_only=True, score_only=True)
alignment_scores2 = pairwise2.align.globalxx(seq2, seq4, one_alignment_only=True, score_only=True)
print("Similarity between DEN2 and DEN3:", alignment_scores1/len(seq2)* 100)
print("Similarity between DEN2 and DEN4:", alignment_scores1/len(seq2)* 100)
