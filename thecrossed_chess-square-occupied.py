# https://www.kaggle.com/datasnaek/chess
import numpy as np
import pandas as pd # pandas
import csv
import re
import matplotlib
import matplotlib.pyplot as plt
# Steps
# Step 1 extract moves from each game
# Step 2 turn move string into a new column
# Step 3 extract occupied squares from moves in each game
# Step 4 build a dictionary and store frequency of square occupied as values (normalized)
# Step 5 draw a board and visualize data
chess = pd.read_csv('../input/games.csv')
chess.head()
# extract moves from each game
chess_moves = chess[['moves']].copy()
chess_moves.head()
# Step 2 turn move string into a new column each_move
def each_move(chess_moves):
    each = (chess_moves['moves']).split(' ')

    return each
chess_moves['each_move'] = chess_moves.apply(each_move, axis=1)
chess_moves.head()
# Step 3 extract occupied squares from moves in each game
def square(chess_moves):
    squares = []
    for m in chess_moves['each_move']:
        if m == 'O-O' or m == 'O-O+':
            if chess_moves['each_move'].index(m)%2 == 0:
                squares.append('g1')
                squares.append('f1')
            elif chess_moves['each_move'].index(m)%2 == 1:
                squares.append('g8')
                squares.append('f8')            
        elif m == 'O-O-O' or m =='O-O-O+':
            if chess_moves['each_move'].index(m)%2 == 0:
                squares.append('c1')
                squares.append('d1')
            elif chess_moves['each_move'].index(m)%2 == 1:
                squares.append('c8')
                squares.append('d8')             
        else:
            if '=' in m:
                squares.append(m.split('=')[0][-2:])
            elif m[-1] == '+' or m[-1] =='#':
                squares.append(m[-3:-1])           
            else:
                squares.append(m[-2:])

    return squares
chess_moves['square'] = chess_moves.apply(square, axis=1)
chess_moves.head()
# Step 4 build a dictionary and store frequency of square occupied as values (normalized)
sq_count = {}
for s in chess_moves['square']:
    for i in s:
        sq_count[i] = sq_count.get(i, 0) + 1
# normalize value
def normalize(d, target=1.0):
   raw = sum(d.values())
   factor = target/raw
   return {key:round(value*factor*100,2) for key,value in d.items()}
sq_norm = normalize(sq_count)
sq_norm
# turn dictionary values into lists, for data visualization in next steps
letter = ['a','b','c','d','e','f','g','h']
eight = []
for k in letter: 
    eight.append(sq_norm[k+'8'])
    
sev = []
for k in letter: 
    sev.append(sq_norm[k+'7'])
    
six = []
for k in letter: 
    six.append(sq_norm[k+'6'])
    
fiv = []
for k in letter: 
    fiv.append(sq_norm[k+'5'])
    
four = []
for k in letter: 
    four.append(sq_norm[k+'4'])
    
thr = []
for k in letter: 
    thr.append(sq_norm[k+'3'])
    
two = []
for k in letter: 
    two.append(sq_norm[k+'2'])
    
one = []
for k in letter: 
    one.append(sq_norm[k+'1'])
# Step 5 draw a board and visualize data
number = ["8","7", "6", "5", "4",
              "3", "2", "1"]
alphabet = ["a", "b", "c",
           "d", "e", "f", "g","h"]
board = np.array([eight,sev,six,fiv,four,thr,two,one])


fig, ax = plt.subplots()
im = ax.imshow(board)

# We want to show all ticks...
ax.set_xticks(np.arange(len(alphabet)))
ax.set_yticks(np.arange(len(number)))
# ... and label them with the respective list entries
ax.set_xticklabels(alphabet)
ax.set_yticklabels(number)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(number)):
    for j in range(len(alphabet)):
        text = ax.text(j, i, board[i, j],
                       ha="center", va="center", color="w")

ax.set_title("square occupied frequency(normalized)")
fig.tight_layout()
plt.show()
# tada!