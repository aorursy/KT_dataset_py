import numpy as np
import random
import math
C = "abcdegsacbdcefabafgaaa"
feature_size = 10
Z = [random.randint(0,feature_size-1) for z in range(len(C))]
n = len(C)
Z
char_to_ix = {}
for char in C:
    if char not in char_to_ix:
        char_to_ix[char] = len(char_to_ix)
len(char_to_ix)
print(char_to_ix)
#P(C) and P(C.i|C.i-1)
PC = np.zeros( len(char_to_ix))
cond_PC = np.zeros(( len(char_to_ix), len(char_to_ix)) )

for e,c in enumerate(C):
    ix = char_to_ix[c]
    PC[ix] += 1 
    if( e > 0 ):
        cond_PC[char_to_ix[ C[e-1] ], char_to_ix[ C[e] ] ] += 1
        
for row in range( len(char_to_ix) ):
    cond_PC[row] = cond_PC[row]/np.sum(cond_PC[row])

print(cond_PC)
#p(z|c)
pz = np.zeros( ( feature_size, len(char_to_ix) ) )
for e,c in enumerate(C):
    pz[ Z[e], char_to_ix[c] ] += 1

for row in range( feature_size ):
    pz[row] = pz[row]/( 0.0001 + np.sum(pz[row]) )

print(pz)
n = 10
Z_new = [ random.randint(0, feature_size-1) for z in range(n)]
print(Z_new)
#Initialize structures
length = 0
words = [ [] for i in range( len(char_to_ix) ) ]
len_path = np.zeros( len(char_to_ix) )
#First layer
for ix,c in enumerate(char_to_ix):
    len_path[ix] = -math.log( PC[ix] + 0.001 ) - math.log( pz[ Z_new[0]][ix] + 0.001 )
    words[ ix ].append(c)
    print( words[ix], len_path[ ix] )
print(words)
for i in range(1, n):
    new_len_path = np.zeros( len(char_to_ix) )
    for ch in char_to_ix:
        min_len_path = 100000000
        best = 0
        for c in char_to_ix:
            cand =  len_path[ char_to_ix[c] ] - math.log( cond_PC[char_to_ix[c]][char_to_ix[ch]] + 0.01) -math.log( pz[i][ char_to_ix[c] ] + 0.01 )
            if cand < min_len_path:
                min_len_path = cand
                best = c
        words[char_to_ix[ch]].append(best)
        new_len_path[ char_to_ix[ch] ] = min_len_path
    len_path = new_len_path.copy()
print( len_path)
print(words)
