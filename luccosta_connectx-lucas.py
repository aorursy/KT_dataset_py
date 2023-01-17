!pip install git+https://github.com/matheusgmaia/kaggle-environments
from kaggle_environments import evaluate, make, utils
import numpy as np
import random
from prettytable import PrettyTable

# Imports do algorítmo genético
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import lines

from ipywidgets import interact
import ipywidgets as widgets

from collections import deque
import math
import heapq

import time
import bisect
class Log:
    moves = []
    
    def add_move(pieces, fit_list):
        move = [pieces, fit_list]
        self.moves.append(move)

    def show_log():
        t = PrettyTable(['Column', 1, 2, 3, 4, 5, 6, 7])
        for move in moves:
            t.add_row(['Pieces'].append(move[0]))
            t.add_row(['Fit list'].append(move[1]))
        print(t)

def show_board(board):
    for row in reversed(board):
        print (row)
def is_adversary_in_mid_spaces(mid_spaces, board, adversary):
    print("-------------")
    print(list(mid_spaces))
    for mid_space in mid_spaces:
        print (mid_space)
        if board[int(mid_space[0])][int(mid_space[1])] == adversary:
            return True

def evaluation(state_space, board, fit=0):
    ones, twos = state_space

    for i in range(len(ones)):
        for j in range(i+1, len(ones)):
            distance = np.array(ones[i]) - np.array(ones[j])
            if abs(distance[0]) == abs(distance[1]) and abs(distance[0]) < 4: # Diagonal
                step_x = 1 if distance[0] > 0 else -1
                step_y = 1 if distance[1] > 0 else -1
                mid_spaces = map(lambda x: (ones[i][0]+x*step_x, ones[i][1]+x*step_y), range(1, abs(distance[0])))

                adversary_in_mid_spaces = is_adversary_in_mid_spaces(mid_spaces, board, 2)

                if not adversary_in_mid_spaces:
                    fit += 2
            if (distance[0] == 0 or distance[1] == 0) and (abs(distance[0] + distance[1]) < 4): # Horizontal e Vertical
                if distance[0] == 0:
                    x_array = np.zeros(abs(distance[1])-1)
                    y_array = range(1, abs(distance[1]))
                else:
                    x_array = range(1, abs(distance[0]))
                    y_array = np.zeros(abs(distance[0])-1)
                mid_spaces = map(lambda x, y: (x+ones[i][0], y+ones[i][1]), x_array, y_array)

                adversary_in_mid_spaces = is_adversary_in_mid_spaces(mid_spaces, board, 2)

                fit += 2

    return fit
def disturb_oponent_evaluation(state_space, board, target, fit=0):
    r, c = target
    
    # -- Horizontal --
    if c < 6:
        if board[r][c+1] == 2:
            if c > 0:
                if board[r][c-1] == 2:
                    fit += 2
                    if c < 5:
                        if board[r][c+2] == 2:
                            fit += 40
                    if c > 1:
                        if board[r][c-2] == 2:
                            fit += 40
                elif c > 4:
                    pass
                elif board[r][c+2] == 2:
                    fit += 2
                    if c < 4:
                        if board[r][c+3] == 2:
                            fit += 20
            else: 
                if board[r][c+2] == 2:
                    fit += 2
                    if board[r][c+3] == 2:
                        fit += 40

    if c > 1:
        if board[r][c-1] == 2 and board[r][c-2] == 2:
            fit += 2
            if c > 2:
                if board[r][c-3] == 2:
                    fit += 40
    
    # -- Vertical --
    if r > 1:
        if board[r-1][c] == 2 and board[r-2][c] == 2:
            fit += 2
            if r > 2:
                if board[r-3][c] == 2:
                    fit += 40
    
    # -- Diagonal --

    # Diagonal secundária
    if not (c > 5 or r > 4):
        if board[r+1][c+1] == 2:
            if not (c < 1 or r < 1):
                if board[r-1][c-1] == 2:
                    fit +=2
                    if not (c > 4 or r > 3):
                        if board[r+2][c+2] == 2:
                            fit += 20
                    if not (c < 2 or r < 2):
                        if board[r-2][c-2] == 2:
                            fit += 20
                elif not (c < 5 and r < 4):
                    pass
                elif board[r+2][c+2] == 2:
                    fit += 2
                    if not (c > 3 or r > 2):
                        if board[r+3][c+3] == 2:
                            fit += 40
            elif not (c > 3 or r > 2):
                if board[r+2][c+2] == 2:
                    fit += 2
                    if board[r+3][c+3] == 2:
                        fit += 40
                 
    if not (c < 2 or r < 2):
        if board[r-1][c-1] == 2 and board[r-2][c-2] == 2:
            fit += 2
            if not (c < 3 or r < 3):
                if board[r-3][c-3] == 2:
                    fit += 40

    # Diagonal primária
    if not (c > 5 or r < 1):
        if board[r-1][c+1] == 2:
            if not (c < 1 or r > 4):
                if board[r+1][c-1] == 2:
                    fit +=2
                    if not (c < 2 or r > 3):
                        if board[r+2][c-2] == 2:
                            fit += 40
                    if not (c > 4 or r < 2):
                        if board[r-2][c+2] == 2:
                            fit += 40
                elif not (c < 5 and r > 1):
                    pass
                elif board[r-2][c+2] == 2:
                    fit += 2
                    if not (c > 3 or r < 3):
                        if board[r-3][c+3] == 2:
                            fit += 40
            elif not (c > 3 or r < 3):
                if board[r-2][c+2] == 2:
                    fit += 2
                    if board[r-3][c+3] == 2:
                        fit += 40

    if not (c > 4 or r < 2):
        if board[r-1][c+1] == 2 and board[r-2][c+2] == 2:
            fit += 2
            if not (c > 3 or r < 3):
                if board[r-3][c+3] == 2:
                    fit += 40

    return fit
weights = [2, 6, 100, 100]

def find_seven_row(step, board, target):
    seven_row = [0, 0, 0, 0, 0, 0, 0]

    begin = (target[0] - step[0]*3, target[1] - step[1]*3)

    row_locations = map(lambda x: (begin[0]+step[0]*x, begin[1]+step[1]*x), range(7))

    for row_location, i in zip(row_locations, range(7)):
        if row_location[0] < 0 or row_location[1] < 0 or row_location[0] > 5 or row_location[1] > 6:
            seven_row[i] = -1
        else:
            seven_row[i] = board[row_location[0]][row_location[1]]

    return seven_row

def evaluate_seven_row(seven_row, player, non_valid=-1, fit=0): # Avaliação do seven row por meio de filtros de 4 espaços
    oponent = 2 if player == 1 else 1
    w = weights[0:4]
    
    for i in range(4):
        four_filter = seven_row[i:i+4]
        #print(four_filter, four_filter.count(1))

        if oponent in four_filter or non_valid in four_filter:
            pass
        elif four_filter.count(player) == 2:
            fit += w[0]
        elif four_filter.count(player) == 3:
            fit += w[1]
        elif four_filter.count(player) == 4:
            fit += w[2]
            if player == 1:
                fit += w[3]

    return fit

def alternate_evaluation(state_space, board, target, player=1, fit=0):
    possibilities = [(1, 0), (0, 1), (1, 1), (-1, 1)]

    for possibilitie in possibilities:
        # Linhas de 7 com o target no centro
        seven_row = find_seven_row(possibilitie, board, target)

        fit += evaluate_seven_row(seven_row, player)
    
    return fit
# Functions to perception of the ambient

def fill_piece_lists(piece_list, piece, row, r):
    temp = [i for i,x in enumerate(row) if x==piece]

    for c in temp:
        piece_list += [(r, c)]

def perception(board):
    ones, twos = [], []

    r = -1

    for row in (board):
        r += 1
        if (1 in row) or (2 in row):
            # Tem peças nessa linha e vamos registrar quais são elas

            fill_piece_lists(ones, 1, row, r)
            fill_piece_lists(twos, 2, row, r)

    return (ones, twos)

# Agent

def my_agent(observation, configuration):
    columns, rows, inarow = configuration.columns, configuration.rows, configuration.inarow

    first_play = 1 not in observation.board

    board = [observation.board[:7]] + [observation.board[7:14]] + [observation.board[14:21]] + [observation.board[21:28]] + [observation.board[28:35]] + [observation.board[35:42]]
    board = list(reversed(board))

    state_space = perception(board)
    fit = []

    for c in range(columns):
        # Encontra a linha livre para a coluna selecionada
        r = 0
        while board[r][c] != 0:
            r += 1
            if r == rows:
                break

        if (r < rows):
            temp_fit = 0
            # Colocando uma peça no tabuleiro na coluna c
            state_space[0].append((r, c))
            board[r][c] = 1

            # Avaliar o quanto a jogada leva a vitória
            #temp_fit += evaluation(state_space, board)

            temp_fit += alternate_evaluation(state_space, board, (r, c))
            
            # Evitar que a jogada beneficie o oponente na próxima rodada checando se ele ganha por causa dela
            if r + 1 < 6: 
                board[r+1][c] = 2 
                temp_fit -= alternate_evaluation(state_space, board, (r+1, c), 2)
                board[r+1][c] = 0

            # Tirando peça
            state_space[0].pop()
            board[r][c] = 0

            # Avaliar como a jogada afeta o oponente
            temp_fit += disturb_oponent_evaluation(state_space, board, (r, c))

            # Colocando peça oponente
            board[r][c] = 2
            temp_fit += alternate_evaluation(state_space, board, (r, c), 2)

            # Tirando peça
            board[r][c] = 0

            fit.append(temp_fit)
        else:
            fit.append(-100)

    decision = fit.index(max(fit))

    if first_play:
        #decision = random.choice([2, 3, 4])
        decision = 2
        first_play = False

    #print ("---- Fit list ----")
    #print (observation.board)
    #print (fit)
    #print (decision)

    return decision
class ObservationTest:
    def __init__(self):
        self.board = [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 2, 0, 0, 0, 1,
                      0, 0, 1, 0, 0, 0, 2,
                      0, 1, 1, 1, 2, 2, 2,
                      0, 1, 1, 2, 1, 2, 2]

class ConfigurationTest:
    def __init__(self):
        self.columns = 7
        self.rows = 6
        self.inarow = 4

obs, config = ObservationTest(), ConfigurationTest()
result = my_agent(obs, config)

print ("\nResult: " + str(result))
env.reset()
# Play as the first agent against default "random" agent.
env.run([my_agent, "negamax"])
env.render(mode="ipython", width=500, height=450)
def mean_win_draw(rewards):
    return sum( 1 for r in rewards if (r[0] == 1 or r[0] == 0.)) / len(rewards)

# Run multiple episodes to estimate its performance.
vs_random = mean_win_draw(evaluate("connectx", [my_agent, "random"], num_episodes=10))
print("My Agent vs Random Agent:", vs_random)

vs_negamax = mean_win_draw(evaluate("connectx", [my_agent, "negamax"], num_episodes=10))
print("My Agent vs Negamax Agent:", vs_negamax)

vs_rules = mean_win_draw(evaluate("connectx", [my_agent, "rules"], num_episodes=10))
print("My Agent vs Rule Agent:", vs_rules)

vs_greedy = mean_win_draw(evaluate("connectx", [my_agent, "greedy"], num_episodes=10))
print("My Agent vs Greedy Agent:", vs_greedy)
import csv

seu_nome = "LUCAS_FERNANDO"

rows = [['Id', 'Predicted'],['random',vs_random],[ 'negamax', vs_negamax],[ 'rules', vs_rules],[ 'greedy', vs_greedy]]
f = open(seu_nome+'-ConnectX.csv', 'w')
with f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)