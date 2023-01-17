blosum_letters = 'ARNDCQEGHILKMFPSTWYV'
dimension = len(blosum_letters) + 1
blosum_matrix = [['' for x in range(dimension)] for y in range(dimension)]

def fill_blosum(blosum_matrix):
	# Fill the matrix top
	start_x = 1
	y = 0
	for letter in blosum_letters:
		blosum_matrix[y][start_x] = letter
		start_x += 1
	
	# Fill the matrix leftmost side
	x = 0
	start_y = 1
	for letter in blosum_letters:
		blosum_matrix[start_y][x] = letter
		start_y += 1
	
	# Fill matrix with scores
	rows = []
	with open("../input/blosum/blosum50.txt") as f:
		lines = f.readlines()
		for line in lines:
			row_of_nums = line.split()
			rows.append(row_of_nums)

	start_row = 0
	start_number = 0
	begin_x = 1
	begin_y = 1
	i = 1
	cells_to_fill = (dimension - 1)* (dimension - 1)
	while i <= cells_to_fill:
		blosum_matrix[begin_y][begin_x] = int(rows[start_row][start_number])

		if begin_x == dimension - 1:
			begin_y += 1
			begin_x = 1
			start_row += 1
			start_number = 0
		else:
			begin_x += 1
			start_number += 1

		i += 1

	# Pretty print the matrix
	#for x in range(dimension):
	#	print(blosum_matrix[x])

	#print("")

def run_algorithm(seq1, seq2, method):
	results = forward_track(seq1, seq2, method)
	filled_matrix = results[0]
	position = results[1]
	result = backwards_track(seq1, seq2, filled_matrix, method, position)

	print('Matching the protein sequence ' + seq1 + ' with ' + seq2 + ' using the ' + method + ' algorithm:')
	print(result[0])
	print(result[1])
	print('The score of this alignment is: ' + str(result[2]))
	print('')

def forward_track(seq1, seq2, method):
	# Define the matrix dimensions
	extra_cells = 2
	width = len(seq1) + extra_cells
	height = len(seq2) + extra_cells

	# Fill the matrix with x's 
	matrix = [['' for x in range(width)] for y in range(height)]

	# Fill the matrix top with the first sequence
	start_x = 2 
	y = 0
	for letter in seq1:
		matrix[y][start_x] = letter
		start_x += 1

	# Fill the matrix leftmost side with the second sequence
	x = 0
	start_y = 2
	for letter in seq2:
		matrix[start_y][x] = letter
		start_y += 1
	
	# Initialise the matrix
	matrix[1][1] = 0

	# Fill matrix with scores
	begin_x = 2
	begin_y = 1
	i = 1
	cells_to_fill = ((len(seq1) + 1) * (len(seq2) + 1)) - 1
	# Task 2 - Storing the location of the highest value
	highest_val = 0
	position = [0, 0]
	while i <= cells_to_fill:

		cell_value = max_value(begin_x, begin_y, matrix, method)
		if (cell_value > highest_val):
			highest_val = cell_value
			position[0] = begin_x
			position[1] = begin_y

		matrix[begin_y][begin_x] = cell_value

		if begin_x == width - 1:
			begin_x = 1
			begin_y += 1
		else:
			begin_x += 1

		i += 1

	# Pretty print the matrix
	for x in range(height):
		print(matrix[x])
	
	return (matrix, position)

def backwards_track(seq1, seq2, matrix, method, position):
	str_a = ''
	str_b = ''
	extra_cells = 2
	d = -8
	x = 0
	y = 0

	if method == 'Needleman':
		# Index the bottom right cell, the starting position
		x = len(seq1) + extra_cells - 1 
		y = len(seq2) + extra_cells - 1 
	else:
		# Index the cell with the highest value, the starting position
		x = position[0]
		y = position[1]

	while x > 1 or y > 1:
		if method != 'Needleman' and matrix[y][x] == 0:
			break
		if x > 1 and y > 1 and matrix[y][x] == matrix[y - 1][x - 1] + blosum_score(matrix[0][x], matrix[y][0]):
			str_a = matrix[0][x] + str_a
			str_b = matrix[y][0] + str_b
			x -= 1
			y -= 1
		elif x > 1 and matrix[y][x] == matrix[y][x - 1] + d:
			str_a = matrix[0][x] + str_a
			str_b = "-" + str_b
			x -= 1
		else:
			str_a = '-' + str_a
			str_b = matrix[y][0] + str_b
			y -= 1
		
	score = 0 # Alignment score
	for i in range(len(str_a)):
		if str_a[i] == '-' or str_b[i] == '-':
			score += d
		else:
			score += blosum_score(str_a[i], str_b[i])
		
	return (str_a, str_b, score)

def max_value(x, y, matrix, method):
	d = -8

	if y == 1:
		if method == 'Needleman':
			return matrix[y][x - 1] + d
		else:
			return 0
	elif x == 1:
		if method == 'Needleman':
			return matrix[y - 1][x] + d
		else:
			return 0
	elif method == 'Needleman':
		val1 = matrix[y - 1][x - 1] + blosum_score(matrix[0][x], matrix[y][0])
		val2 = matrix[y][x - 1] + d
		val3 = matrix[y - 1][x] + d
		return max(val1, val2, val3)
	else:
		val1 = matrix[y - 1][x - 1] + blosum_score(matrix[0][x], matrix[y][0])
		val2 = matrix[y][x - 1] + d
		val3 = matrix[y - 1][x] + d
		val4 = 0
		return max(val1, val2, val3, val4)

def blosum_score(a, b):
	score_x = ''
	score_y = ''

	for i in range(1, dimension):
		if blosum_matrix[0][i] == a:
			score_x = i
		if blosum_matrix[i][0] == b:
			score_y = i

	return blosum_matrix[score_y][score_x]

fill_blosum(blosum_matrix)
run_algorithm('HEAGAWGHEE', 'PAWHEAE', 'Needleman')
run_algorithm('SALPQPTTPVSSFTSGSMLGRTDTALTNTYSAL', 'PSPTMEAVTSVEASTASHPHSTSSYFATTYYHLY', 'Needleman')
run_algorithm('HEAGAWGHEE', 'PAWHEAE', 'Smith')
run_algorithm('MQNSHSGVNQLGGVFVNGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRY', 'TDDECHSGVNQLGGVFVGGRPLPDSTRQKIVELAHSGARPCDISRI', 'Smith')
obs = ('A', 'T', 'C', 'G')
states = ('AT rich', 'CG rich')
start_p = {}
trans_p = {
   'AT rich' : {'AT rich': 0.9998, 'CG rich': 0.0002},
   'CG rich' : {'AT rich': 0.0003, 'CG rich': 0.9997}
   }
emit_p = {
   'AT rich' : {'A': 0.2698, 'T': 0.3237, 'C': 0.2080, 'G': 0.1985},
   'CG rich' : {'A': 0.2459, 'T': 0.2079, 'C': 0.2478, 'G': 0.2984}
   }
import math
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Gives the most likely valid sequence of states that generated the sequence obs
def viterbi(obs, states, start_p, trans_p, emit_p):
    
    V = [{}]
    randomIndex = random.randint(0, len(states) - 1) # Assume we can start from any state
    for i in range(len(states)):
        if i == randomIndex:
            V[0][states[i]] = {"prob": 1, "prev": None} # The probability of being in the initial state at t = 0 is 1
        else:
            V[0][states[i]] = {"prob": 0, "prev": None} # The probability of being in any other state at t = 0 is 0

    # Run Viterbi when t > 0 
    for t in range(1, len(obs)):
        V.append({})
        # Iteratively compute, for each state, the probability of being in that state at time t
        for st in states:
            max_tr_prob = V[t-1][states[0]]["prob"]+math.log(trans_p[states[0]][st]) 
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t-1][prev_st]["prob"]+math.log(trans_p[prev_st][st]) 
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                    
            max_prob = max_tr_prob + math.log(emit_p[st][obs[t]]) 
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

    # Stores the Viterbi predictions
    opt = []
    max_prob = -10000000000000
    previous = None
    best_st = ''
    # Get most probable state and its backtrack. In other words, find the state with the highest probability at the last time step.
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"] 
            best_st = st
    opt.append(best_st)
    previous = best_st
    
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    # Prints the Viterbi predictions using run-length encoding 
    print("The Viterbi predictions are: " + encode(opt).replace(" ", ""))

    
# Function to do run-length encoding 
def encode(string):
    
    if string == "":
        return ""
    
    string += "#" 
    encoded_string = ""
    count = 0
    current_char = string[0] 
    counts_per_region = []
    regions = []

    for i in string:
        if i != current_char or i == "#":
            if count != 1:
                encoded_string += str(count) + current_char
            else:
                encoded_string += current_char
            regions.append(current_char)
            counts_per_region.append(count) 
            current_char = i
            count = 1
        else:
            count += 1
    
    #draw_regions(counts_per_region, regions)

    return encoded_string

# Function to draw the regions
def draw_regions(counts_per_region, regions):

    total_regions = sum(counts_per_region)
    rect_lower_x = [0]

    state_colour = {}
    state_colour['AT rich'] = 'orange'
    state_colour['CG rich'] = 'pink'

    for i in range(len(counts_per_region)):
        rect_lower_x.append(rect_lower_x[i] + counts_per_region[i])

    fig1 = plt.figure()
    fig1.suptitle('AT rich and CG rich regions for the genome Escherichia phage Lambda', fontsize=18)
    axes = plt.gca()
    axes.set_xlim([0, total_regions])
    axes.set_ylim([0, 200])
    
    for i in range(len(counts_per_region)):
        axes.add_patch(Rectangle((rect_lower_x[i], 0), counts_per_region[i], 200, color = state_colour[regions[i]]))
    
    orange_patch = mpatches.Patch(color='orange', label='AT rich')
    pink_patch = mpatches.Patch(color='pink', label='CG rich')
    plt.legend(handles=[orange_patch, pink_patch], loc = 'upper right')
    rect_lower_x.remove(0)
    axes.set_xticks(rect_lower_x)
    plt.gca().axes.get_yaxis().set_visible(False)
    
    #plt.show()
obs = ('1', '2', '3', '4', '5', '6')
states = ('H', 'D')
start_p = {'H' : 0.5, 'D' : 0.5}
trans_p = {
   'H' : {'H': 0.9, 'D': 0.1},
   'D' : {'H': 0.1, 'D': 0.9}
   }
frac = 1/6
emit_p = {
   'H' : {'1': frac, '2': frac, '3': frac, '4': frac, '5': frac, '6': frac},
   'D' : {'1': 0.1, '2': 0.1, '3': 0.1, '4': 0.1, '5': 0.1, '6': 0.5},
   }

dice_rolls = "5453525456666664365666635661416626365666621166211311155566351166565663466653642535666662541345464155"
observations = []
for num in dice_rolls:
    observations.append(num)
    
viterbi(observations, states, start_p, trans_p, emit_p)
states = ('AT rich', 'CG rich')
start_p = {}
trans_p = {
   'AT rich' : {'AT rich': 0.9998, 'CG rich': 0.0002},
   'CG rich' : {'AT rich': 0.0003, 'CG rich': 0.9997}
   }
emit_p = {
   'AT rich' : {'A': 0.2698, 'T': 0.3237, 'C': 0.2080, 'G': 0.1985},
   'CG rich' : {'A': 0.2459, 'T': 0.2079, 'C': 0.2478, 'G': 0.2984}
   }

# Stores the sequence of observations 
observations = []
lines = open("../input/phaselambda/phaseLambda.txt").read().splitlines()
for line in lines:
   for char in line:
      observations.append(char)

viterbi(observations, states, start_p, trans_p, emit_p)