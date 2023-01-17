from Bio.SubsMat import MatrixInfo
import numpy as np
import matplotlib.pyplot as plt
# Pretty print matrices
def print_matrix(mat):

    for i in range(0, len(mat)):
        print("[", end = "")
        
        for j in range(0, len(mat[i])):

            print(mat[i][j], end = "")

            if j != len(mat[i]) - 1:
                print("\t", end = "")
                
        print("]\n")
# Make an empty matrix of 0s
def zeros(rows, cols):
    retval = []

    for x in range(rows):
        retval.append([])

        for y in range(cols):
            retval[-1].append(0)
            
    return retval
# Blosum 50
def blosum(alpha, beta):
    
    pair = (alpha, beta)
    b50 = MatrixInfo.blosum50
    
    if pair not in b50:
        return b50[tuple(reversed(pair))]
    else:
        return b50[pair]
# Determinines the score between any two bases in alignment
def match_score(alpha, beta):
    
    if alpha == beta:
        return blosum(alpha, beta)
    elif alpha == '-' or beta == '-':
        return gap_penalty
    else:
        return blosum(alpha, beta)
gap_penalty = -8
def needleman_wunsch(seq1, seq2):
    
    n = len(seq1)  
    m = len(seq2)
    
    # Initiate empty scoring matrix
    score = zeros(m+1, n+1)
    
    # Fill out first column
    for i in range(0, m + 1):
        score[i][0] = gap_penalty * i
    
    # Fill out first row
    for j in range(0, n + 1):
        score[0][j] = gap_penalty * j
    
    # Fill out all other values in the score matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            
            # Calculate the score by checking the top, left, and diagonal cells
            match = score[i - 1][j - 1] + match_score(seq1[j-1], seq2[i-1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            
            # Record the maximum score from the three possible scores calculated above
            score[i][j] = max(match, delete, insert)
    
    #print_matrix(score)
    
    align1 = ""
    align2 = ""
    
    # Start from the bottom right cell in matrix
    i = m
    j = n
    
    # We finish this loop when we reach the top or the left edge
    while i > 0 and j > 0:
        score_current = score[i][j]
        score_diagonal = score[i-1][j-1]
        score_up = score[i][j-1]
        score_left = score[i-1][j]
        
        # Figure out which cell the score came from
        if score_current == score_diagonal + match_score(seq1[j-1], seq2[i-1]):
            align1 += seq1[j-1]
            align2 += seq2[i-1]
            i -= 1
            j -= 1
        elif score_current == score_up + gap_penalty:
            align1 += seq1[j-1]
            align2 += '-'
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1 += '-'
            align2 += seq2[i-1]
            i -= 1

    # Finish tracing up to the top left corner
    while j > 0:
        align1 += seq1[j-1]
        align2 += '-'
        j -= 1
    while i > 0:
        align1 += '-'
        align2 += seq2[i-1]
        i -= 1
    
    align1 = align1[::-1]
    align2 = align2[::-1]
    
    # Alignment score
    align_score = 0
    for i in range(len(align1)):
        if align1[i] == '-' or align2[i] == '-':
            align_score += gap_penalty
        else:
            align_score += match_score(align1[i], align2[i])
    
    return(align1, align2, align_score)
# The two sequences on the lecture slides
seq1 = "HEAGAWGHEE"
seq2 = "PAWHEAE"

output1, output2, score1 = needleman_wunsch(seq1, seq2)

print(output1 + "\n" + output2 + "\n")
print("The alignment score is " + str(score1) + ".\n")
print("That is a very low alignment score, which means the match is very poor.")
seq3 = "SALPQPTTPVSSFTSGSMLGRTDTALTNTYSAL"
seq4 = "PSPTMEAVTSVEASTASHPHSTSSYFATTYYHLY"

output3, output4, score2 = needleman_wunsch(seq3, seq4)

print(output3 + "\n" + output4 + "\n")
print("The alignment score is " + str(score2) + ".\n")
print("That is still a low alignment score, which means the match is poor.")
# Determine cell with highest score
def max_element(A):
    r, (c, l) = max(map(lambda t: (t[0], max(enumerate(t[1]), key=lambda v: v[1])), enumerate(A)), key=lambda v: v[1][1])
    return (r, c)
def smith_waterman(seq1, seq2):
    
    n = len(seq1)  
    m = len(seq2)
    
    # Initiate empty scoring matrix
    score = zeros(m+1, n+1)
    
    # Fill out first column
    for i in range(0, m + 1):
        score[i][0] = 0
    
    # Fill out first row
    for j in range(0, n + 1):
        score[0][j] = 0
    
    # Fill out all other values in the score matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            
            # Calculate the score by checking the top, left, and diagonal cells
            match = score[i - 1][j - 1] + match_score(seq1[j-1], seq2[i-1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            
            # Record the maximum score from the three possible scores calculated above
            score[i][j] = max(match, delete, insert, 0)
    
    #print_matrix(score)

    align1 = ""
    align2 = ""
    
    # Start from the max cell in matrix
    i, j = max_element(score)
    
    while score[i][j] > 0:
        score_current = score[i][j]
        score_diagonal = score[i-1][j-1]
        score_up = score[i][j-1]
        score_left = score[i-1][j]
        
        # Figure out which cell the score came from
        if score_current == score_diagonal + match_score(seq1[j-1], seq2[i-1]):
            align1 += seq1[j-1]
            align2 += seq2[i-1]
            i -= 1
            j -= 1
        elif score_current == score_up + gap_penalty:
            align1 += seq1[j-1]
            align2 += '-'
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1 += '-'
            align2 += seq2[i-1]
            i -= 1
    
    align1 = align1[::-1]
    align2 = align2[::-1]
    
    # Alignment score
    align_score = 0
    for i in range(len(align1)):
        if align1[i] == '-' or align2[i] == '-':
            align_score += gap_penalty
        else:
            align_score += match_score(align1[i], align2[i])
    
    return(align1, align2, align_score)
output5, output6, score3 = smith_waterman(seq1, seq2)

print(output5 + "\n" + output6 + "\n")
print("The alignment score is " + str(score3) + ".\n")
print("That is a fairly high alignment score, which means the match is good.")
seq5 = "MQNSHSGVNQLGGVFVNGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRY"
seq6 = "TDDECHSGVNQLGGVFVGGRPLPDSTRQKIVELAHSGARPCDISRI"

output7, output8, score4 = smith_waterman(seq5, seq6)

print(output7 + "\n" + output8 + "\n")
print("The alignment score is " + str(score4) + ".\n")
print("That is a very high alignment score, which means the match is strong.")
# The hidden states
states = ["AT rich", "CG rich"]

tr_matrix = {
    # The probabilities of switching states from AT rich
    'AT rich': {'AT rich': 0.9998, 'CG rich': 0.0002},
    # The probabilities of switching states from CG rich
    'CG rich': {'AT rich': 0.0003, 'CG rich': 0.9997}
}

# Pretty print
print('        ', end=' ')
for state in states:
    print(state, end='  ')
    
print("\nAT rich", end='  ')
for state in tr_matrix['AT rich']:
    print(tr_matrix['AT rich'][state], end='   ')
    
print("\nCG rich", end='  ')
for state in tr_matrix['CG rich']:
    print(tr_matrix['CG rich'][state], end='   ')
observations = ["A", "T", "C", "G"]

em_matrix = {
    'AT rich': {'A': 0.2698, 'T': 0.3237, 'C': 0.2080, 'G': 0.1985},
    'CG rich': {'A': 0.2459, 'T': 0.2079, 'C': 0.2478, 'G': 0.2984}
}

# Pretty print
print('        ', end=' ')
for obs in observations:
    print(obs, end='        ')

print("\nAT rich", end='  ')
for obs in em_matrix['AT rich']:
    print(em_matrix['AT rich'][obs], end='   ')

print("\nCG rich", end='  ')
for prob in em_matrix['CG rich']:
    print(em_matrix['CG rich'][obs], end='   ')
# Assume the initial state is CG rich 
initial_state = "CG rich"
probs = em_matrix['CG rich']

# Determine the first observation in the sequence
initial_obs = np.random.choice(observations, size=None, replace=True, p=list(probs.values()))
print("Position 1: state = " + initial_state + "; observation = " + initial_obs)

sequence = list()
seen = list()

# Update the sequence and seen states
sequence.append(initial_obs)
seen.append(initial_state)

# Determine the rest of the sequence
for i in range(9):
    i = i + 2
    
    previous_state = seen[i-2]
    
    if previous_state == "AT rich":
        current_state_p = tr_matrix['AT rich']
        current_obs_p = em_matrix['AT rich']
    else:
        current_state_p = tr_matrix['CG rich']
        current_obs_p = em_matrix['CG rich']
        
    current_state = np.random.choice(states, size=None, replace=True, p=list(current_state_p.values()))
    current_obs = np.random.choice(observations, size=None, replace=True, p=list(current_obs_p.values()))
    sequence.append(current_obs)
    seen.append(current_state)
    
    print("Position " + str(i) + ": state = " + current_state + 
          "; observation = " + current_obs)
    
print("\nThe generated sequence is:")
print(sequence)
# Creates a HMM configuration to run Viterbi on
class ViterbiHMM:
    
    def __init__(self, states, tr_matrix, em_matrix, st_prob, observations):

        # Initialise HMM config
        self.states = states
        self.tr_matrix = tr_matrix
        self.em_matrix = em_matrix
        self.st_prob = st_prob
        self.observations = observations
        
        # Forward pass dict
        self.v = dict([[k,{}] for k in self.states])
        self.sub_path = dict([[k, []] for k in self.states])
        self.path = []
    
    def _v_i(self, t, csi=None, i=None):
        
        if t == 0:
            if i == states[0]:
                return 0
            else:
                # log(0) is undefined
                return -np.inf
            
        e_i = self.em_matrix[i][csi]
        
        # We want to use log probabilities
        v_js = [v[t-1] + np.log(self.tr_matrix[i][state]) for state, v in self.v.items()]
        v_j = np.max(v_js)
        v_ptr = np.argmax(v_js)
        v_i = np.log(e_i) + v_j
        
        self.sub_path[i].append(v_ptr)
        
        return v_i
    
    # The forward pass algorithm
    def forward(self):
        
        for state, v in self.v.items():
            v[0] = self._v_i(0, csi='N/A', i=state)
            
        for t, k in enumerate(self.observations, 1):
            for state, v in self.v.items():
                v[t] = self._v_i(t, csi=k, i=state)
    
    # Backtracking
    def backward(self):
        
        q_T = self.states[np.argmax([list(v.values())[-1] for v in self.v.values()])]
        self.path.append(q_T)
        
        for i in range(len(self.sub_path[q_T])-1,0,-1):
            q_T_prev = self.sub_path[q_T][i]
            q_T_prev = self.states[q_T_prev]
            self.path.append(q_T_prev)
            q_T = q_T_prev
            
    # Run forward and backward
    def runViterbi(self):
        self.forward()
        self.backward()
# Dishonest Casino config
states = ['Honest', 'Dishonest']
tr_casino = {
    'Honest': {'Honest': 0.9, 'Dishonest': 0.1},
    'Dishonest': {'Honest': 0.1, 'Dishonest': 0.9}
}
em_casino = {
    'Honest': {'1': 1/6, '2': 1/6, '3': 1/6, '4': 1/6, '5': 1/6, '6': 1/6},
    'Dishonest': {'1': 1/10, '2': 1/10, '3': 1/10, '4': 1/10, '5': 1/10, '6': 1/2}
}
observations = [obs for obs in 
                '54535254566666643656666356614166263656666211662113' + 
                '11155566351166565663466653642535666662541345464155']

# Run Viterbi on Dishonest Casino
v_casino = ViterbiHMM(states, tr_casino, em_casino, None, observations)
v_casino.runViterbi()
# Print the full path
casino_path = ['H' if thing == 'Honest' else 'D' for thing in v_casino.path]
# print(casino_path)
predicts = np.reshape(np.array([1 if thing == 'Honest' else 0 for thing in v_casino.path]), (-1, 10))
fig, ax = plt.subplots(1,2,figsize=(20,10))

# Path of predicted states
for i in range(predicts.shape[0]):
    for j in range(predicts.shape[1]):
        ax[0].text(i, j, casino_path[i+j*10], color='w')
        
ax[0].imshow(predicts, cmap='jet')

# Path of predicted observations
for i in range(predicts.shape[0]):
    for j in range(predicts.shape[1]):
        ax[1].text(i, j, v_casino.observations[i+j*10], color='w')
        
ax[1].imshow(predicts, cmap='jet')
# The config matches the initial one set up for the given HMM
states = ["AT rich", "CG rich"]
observations = []

# Observations from the phase lambda genome file
lines = open("../input/lambda/genome.txt").read().splitlines()
for line in lines:
   for char in line:
      observations.append(char)

# Run Viterbi on the genome
v_genome = ViterbiHMM(states, tr_matrix, em_matrix, None, observations)
v_genome.runViterbi()
# Print the full path
# print(['AT' if thing == 'AT rich' else 'CG' for thing in v_genome.path])
# Generating colour map for the genome's region predictions
predicts = [1 if thing == 'AT rich' else 0 for thing in v_genome.path]
predicts = np.array(predicts)
predicts = np.concatenate((predicts, np.zeros(2)))
predicts = np.reshape(predicts, (-1, 188))

fig, ax = plt.subplots(figsize=(10,20))
ax.imshow(predicts, cmap='jet')