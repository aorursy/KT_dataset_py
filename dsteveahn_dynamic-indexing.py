import random
from collections import defaultdict
import time

k = 100
len_t = 10**5
AGCT = "AGCT"
genome = "".join(random.choice(AGCT) for __ in range(len_t))
mygenome = list(genome)

board = ["_"] * len(genome)

for i in random.sample(range(len(genome)), len(genome)//1000):
    gen = t = mygenome[i]
    while gen == t:
        gen = random.choice("AGCT")
        mygenome[i] = gen
mygenome = "".join(mygenome)

numOfReads = 3000
read_length_range = (3*k,4*k)
reads = list()

for __ in range(numOfReads):
    r_l = random.randrange(k,2*k)
    r_i = random.randrange(len(mygenome)-r_l)
    reads.append(mygenome[r_i:r_i+r_l])

start_time = time.time()
def write_on_buffer(p, i, buffer = board):
    for j in range(len(p)):
        if board[i+j] == '_':
            board[i+j] = p[j]
def bestApproximateMatchEditDistance(p, t):
    """Returns the edit distance between two strings, p and t"""
    # Create distance matrix
    D = []
    for i in range(len(p)+1):
        D.append([0]*(len(t)+1))
    
    # Initialize first row and column of matrix
    for i in range(len(p)+1):
        D[i][0] = i
    # See slide 4 on  0440_approx__editdist3.pdf
    # First row is already initialised to zero so we simply just comment the following two lines.
    #for i in range(len(p)+1):
    #    D[0][i] = i
    
    # Fill in the rest of the matrix
    for i in range(1, len(p)+1):
        for j in range(1, len(t)+1):
            distHor = D[i][j-1] + 1
            distVer = D[i-1][j] + 1
            if p[i-1] == t[j-1]:
                distDiag = D[i-1][j-1]
            else:
                distDiag = D[i-1][j-1] + 1
            D[i][j] = min(distHor, distVer, distDiag)

    # Best Approximate Match Distance is the smallest value of the last row
    return min(D[-1]), D[-1].index(min(D[-1]))


class Index(object):

    def __init__(self, t, k):
        self.k = k  # k-mer length (k)
        self.index = defaultdict(list)
        for i in range(len(t) - k + 1):  # for each k-mer
            self.index[t[i:i+k]].append(i)  # add (k-mer, offset) pair

    def query(self, p_kmer):
        return self.index[p_kmer]
genome_k_mer_index = Index(genome,k)
def editDistance(x, y):
    D=[]
    for i in range(len(x)+1):
        D.append([0]* (len(y)+1))
        
    for i in range(len(x)+1):
        D[i][0] = i
    for i in range(len(y)+1):
        D[0][i] = i
        
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            distHor = D[i][j-1] + 1
            distVer = D[i-1][j] + 1
            if x[i-1] == y[j-1]:
                distDiag = D[i-1][j-1]
            else :
                distDiag = D[i-1][j-1] + 1
                
            D[i][j] = min(distHor, distVer, distDiag)
            
    return D[-1][-1]
def get_kmers(s,k):
    return [(j,s[j:j+k]) for j in range(len(s) - k + 1)]
#implementing global alignment
alphabet = ['A', 'C', 'G', 'T']
score = [[0, 4, 2, 4, 8],
         [4, 0, 4, 2, 8],
         [2, 4, 0, 4, 8],
         [4, 2, 4, 0, 8],
         [8, 8, 8, 8, 16]
        ]

def globalAlignment(x, y):
    D=[]
    for i in range(len(x)+1):
        D.append([0]* (len(y)+1))
    for i in range(len(x)+1):
        D[i][0] = D[i-1][0] + score[alphabet.index(x[i-1])][-1]
    for i in range(len(y)+1):
        D[0][i] = D[0][-1] + score[-1][alphabet.index(y[i-1])]
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            distHor = D[i][j-1] + score[-1][alphabet.index(y[j-1])]
            distVer = D[i-1][j] + score[alphabet.index(x[i-1])][-1]
            if x[i-1] == y[j-1]:
                distDiag = D[i-1][j-1]
            else :
                distDiag = D[i-1][j-1] + score[alphabet.index(x[i-1])][alphabet.index(y[j-1])]
            D[i][j] = min(distHor, distVer, distDiag)
    return D[-1][-1]

for read in reads:
    hit_candidate = []
    for j, kmer in get_kmers(read,k):
        for i in genome_k_mer_index.query(kmer):
            write_on_buffer(read, i - j)

#write_on_buffer(read, i - j)
reconstructed = "".join(board)
print(reconstructed)


def hamming(x,y):
    s = 0
    for i,x in enumerate(zip(x,y)):
        a,b = x
        if a != b:
            s += 1
            #print(i, a, b)
    return s
hamming(reconstructed,mygenome)

print("running_time :", time.time() - start_time)
max(len(x) for _, x in genome_k_mer_index.index.items())
