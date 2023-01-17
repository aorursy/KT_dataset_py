from os import listdir
from os.path import isfile, join
import json
import pprint

pp = pprint.PrettyPrinter(indent=4)

my_path = "../input/shobu-randomly-played-games-104k/Shobu_Games_104k_Dataset/"

game_files = ["black/" + f for f in listdir(my_path + "black/") if isfile(join(my_path + "black/", f))]
game_files += ["white/" + f for f in listdir(my_path + "white/") if isfile(join(my_path + "white/", f))]
game_dicts = [] #  type: List[Dict]

opening_move_dict = {}
wins_by_color = {'WHITE': 0, 'BLACK': 0}
for i, game_file in enumerate(game_files):
    #if i >= 10000:
    #    break # TODO remove me later, only loads first 100 games for PROTOTYPING SPEED
    with open(my_path + game_file) as json_file:
        data = json.load(json_file)
        data['filename'] = game_file
        
        #if len(data['turns']) > 25: # TODO remove me, only looking at SHORT games
        #    continue
        
        game_dicts.append(data)
        

def get_board_cell(board, x, y) -> str:
    return board[(y*8)+x]

def set_board_cell(board, x, y, replacement) -> str:
    board[(y*8)+x] = replacement
    return board

def str_to_list(string):
    new_list = [i for i in string]
    return new_list

def flip_game_across_vertical_axis(game_dict):
    game_dict = game_dict.copy()
    for i, turn in enumerate(game_dict['turns']):
        game_dict['turns'][i]['passive']['origin']['x'] = (turn['passive']['origin']['x'] * -1) + 7
        game_dict['turns'][i]['passive']['heading']['x'] = (turn['passive']['heading']['x'] * -1) 
        
        game_dict['turns'][i]['aggressive']['origin']['x'] = (turn['aggressive']['origin']['x'] * -1) + 7
        game_dict['turns'][i]['aggressive']['heading']['x'] = (turn['aggressive']['heading']['x'] * -1) 
        
    for i, game_state in enumerate(game_dict['game_states']):
        new_board = str_to_list(game_state['board'])
        for x in range(0,8):
            for y in range(0,8):
                new_board = set_board_cell(new_board, x, y, get_board_cell(game_state['board'], (x * -1) + 7, y))
        new_board = ''.join(new_board)
        game_dict['game_states'][i]['board'] = new_board
    
    return game_dict

#print("Old game")
#print_shobu_game_dict(game_dicts[0])
#print("")
#print("New game")
#print_shobu_game_dict(flip_game_across_vertical_axis(game_dicts[0]))

# Pretty print a game

def print_shobu_board(board):
    if len(board) != 64:
        return
    
    for y in range(0,8):
        print(board[y*8:y*8 + 4] + "|" + board[y*8 + 4:y*8 + 8])
        if y == 3:
            print("----|----")
            
def print_shobu_game_dict(data):
    for i, board in enumerate(data['game_states']):
        print("TURN: " + board['turn'])
        print_shobu_board(board['board'])
        print("p(" + str(data['turns'][i]['passive']['origin']['x']) + "," + str(data['turns'][i]['passive']['origin']['y']) + ") h(" + str(data['turns'][i]['passive']['heading']['x']) + "," + str(data['turns'][i]['passive']['heading']['y']) + ")")
        print("a(" + str(data['turns'][i]['aggressive']['origin']['x']) + "," + str(data['turns'][i]['aggressive']['origin']['y']) + ") h(" + str(data['turns'][i]['aggressive']['heading']['x']) + "," + str(data['turns'][i]['aggressive']['heading']['y']) + ")")
        print("")
            
def print_shobu_game_file(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        print_shobu_game_dict(data)
        
#print_shobu_board("...o...o..ox.oox......o.xxoox.xx.ooo...oo.xxx....x...xoo...x...x")
#print_shobu_game_file("/home/jaywalker/gitsources/Shobu/games/f3687e95d2c04de28e98158faf743ee62994e8d9b4516ae024b290d1be2e2ba7.json")

print("Number of game files " + str(len(game_files)))

# Flip games across vertical axis if blacks first move is in lower right quadrant
# We do this so that we can count unique opening moves without double counting symmetric moves
for i, data in enumerate(game_dicts):
    if data['turns'][0]['passive']['origin']['x'] > 3:
        #print(game_dicts[i]['filename'])
        #print(data['filename'])
        game_dicts[i] = flip_game_across_vertical_axis(data)

for data in game_dicts:
    if 'winner' in data:
        #print(data['winner'])
        if data['winner'] not in wins_by_color:
            wins_by_color[data['winner']] = 0
        if data['winner'] == '':
            print('No winner in file: ' + data['filename'])
        wins_by_color[data['winner']] = wins_by_color[data['winner']] + 1
    else:
        print('File ' + game_file + ' has no winner.')

    if data['winner'] == 'WHITE':
        continue # skip white wins

    opening_move = json.dumps(data['turns'][0])
    if opening_move not in opening_move_dict:
        opening_move_dict[opening_move] = 1
    else:
        opening_move_dict[opening_move] += 1
            
pp.pprint(wins_by_color)
            
#pp.pprint(opening_move_dict)
total_openers = 0
for k, v in opening_move_dict.items():
    total_openers += 1
    
print("Opening moves observed: " + str(total_openers))
    
listofTuples = sorted(opening_move_dict.items() ,  key=lambda x: x[1])
 
# Iterate over the sorted sequence
for elem in listofTuples :
    print(elem[0],  "::" , elem[1] )
import matplotlib.pyplot as plt
import numpy as np

print("Number of game files found: " + str(len(game_dicts)))
print("Opening moves observed: " + str(total_openers))

heights = [e[1] for e in listofTuples]
plt.bar(range(0, len(listofTuples)), heights)
plt.show()
# How many games won by white?
print('Wins by WHITE: ' + str(wins_by_color['WHITE']))
# How many games won by black? Show percentage
print('Wins by BLACK: ' + str(wins_by_color['BLACK']))

print('BLACK win percentage: ' + str((wins_by_color['BLACK'] / (wins_by_color['BLACK'] + wins_by_color['WHITE']))))
# Mean, median, mode number of turns?
game_turn_counts = {}
for i, game in enumerate(game_dicts):
    num_turns = len(game['turns'])
    if num_turns not in game_turn_counts:
        game_turn_counts[num_turns] = 1
    else:
        game_turn_counts[num_turns] += 1
    
# What does the distribution of turns to win look like?
for i in range(0, 52):
    if i in game_turn_counts:
        print(str(i) + ": " + str(game_turn_counts[i]))
from typing import List, Dict

def get_quadrant(x,y):
    if x < 4:
        if y < 4:
            return 0;
        return 2
    if y < 4:
        return 1
    return 3

print("0,0 is upper-left corner. Positive X is right, positive Y is down.")
print("0,0 is in quadrant: " + str(get_quadrant(0,0)))
print("4,0 is in quadrant: " + str(get_quadrant(4,0)))
print("0,4 is in quadrant: " + str(get_quadrant(0,4)))
print("4,4 is in quadrant: " + str(get_quadrant(4,4)))

def get_quadrant_win_counts(games: List[Dict[int, int]]):
    q_wins = {0:0, 1:0, 2:0, 3:0}
    for g in games:
        last_aggressive_move = g['turns'][len(g['turns']) - 1]['aggressive']
        quadrant = get_quadrant(last_aggressive_move['origin']['x'], last_aggressive_move['origin']['y'])
        q_wins[quadrant] += 1
    return q_wins

    
black_wins = []
white_wins = []
for g in game_dicts:
    if g['winner'] == 'BLACK':
        black_wins.append(g)
    if g['winner'] == 'WHITE':
        white_wins.append(g)
        
black_win_quadrants = get_quadrant_win_counts(black_wins)
white_win_quadrants = get_quadrant_win_counts(white_wins)

black_win_percentages = {
    0: black_win_quadrants[0] / len(black_wins),
    1: black_win_quadrants[1] / len(black_wins),
    2: black_win_quadrants[2] / len(black_wins),
    3: black_win_quadrants[3] / len(black_wins)
}

white_win_percentages = {
    0: white_win_quadrants[0] / len(white_wins),
    1: white_win_quadrants[1] / len(white_wins),
    2: white_win_quadrants[2] / len(white_wins),
    3: white_win_quadrants[3] / len(white_wins)
}

print("Black quadrant win percentages: ")
pp.pprint(black_win_percentages)
print("White quadrant win percentages: ")
pp.pprint(white_win_percentages)
    
