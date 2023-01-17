import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
game_choices = ['red', 'black']

def game(n, ai=lambda x, y, z, a: ('red', 1)):
    player_gain = 0
    player_gain_over_game = []
    player_play_over_game = []
    
    past_moves = []
    player_won_last_move = True
    for i in range(n):
        player_move, player_bet = ai(i, past_moves, player_won_last_move, player_play_over_game)
        
        bank_move = np.random.choice(game_choices)
        past_moves.append(bank_move)
        
        if player_bet > 0:
            player_won_last_move = bank_move == player_move
            if player_won_last_move:
                player_gain += player_bet
            else:
                player_gain -= player_bet

        player_gain_over_game.append(player_gain)
        player_play_over_game.append((player_move, player_bet))
    return player_gain, player_gain_over_game, player_play_over_game
def inverse_of_last_move(past_moves):
    return [x for x in game_choices if x != past_moves[-1]][0]
def try_ai(n, ai=None):
    pg, pgog, ppog = game(n) if ai == None else game(n, ai)
    print('Gains totaux:', pg)
    plt.plot(range(n), pgog, label='Gain')
#     plt.plot(range(n), ppog[1], label='Bet')
    plt.legend()
try_ai(10000)
def dad_ai(i, past_moves, player_won_last_move, player_play_over_game):
    my_move = ''
    my_bet = 0
    
    last_two_moves = past_moves[-2:]
    if len(last_two_moves) > 1 and last_two_moves[0] == last_two_moves[1]:  # last two moves are the same
        my_move = inverse_of_last_move(past_moves)
        my_bet = 1
    return my_move, my_bet
try_ai(10000, dad_ai)
def average_on_n_game(n_game, n, ai):
    total = 0
    pg_over_n_game = []
    for i in range(n_game):
        pg, pgog, ppog = game(n) if ai == None else game(n, ai)
        total += pg
        pg_over_n_game.append(pg)
    return total / n_game, pg_over_n_game
def try_average(n_game, n, ai):
    mean, pgong = average_on_n_game(n_game, n, ai)
    print('Moyenne des gains:', mean)
    plt.plot(range(n_game), pgong, label='Gain par partie')
    plt.legend()
try_average(100, 1000, dad_ai)
def ai_always_double(i, past_moves, player_won_last_move, player_play_over_game):
    if player_won_last_move:
        player_bet = 1
        player_move = np.random.choice(game_choices)
    else:
        last_player_bet = player_play_over_game[-1][1]
        player_bet = last_player_bet * 2
        player_move = inverse_of_last_move(past_moves)
    return player_move, player_bet
try_ai(10000, ai_always_double)
def ai_always_double_but_limited_at_10000(i, past_moves, player_won_last_move, player_play_over_game):
    if player_won_last_move:
        player_bet = 1
        player_move = np.random.choice(game_choices)
    else:
        last_player_bet = player_play_over_game[-1][1]
        player_bet = min(last_player_bet * 2, 10000)
        player_move = inverse_of_last_move(past_moves)
    return player_move, player_bet
try_ai(10000, ai_always_double_but_limited_at_10000)
try_average(1000, 10, ai_always_double_but_limited_at_10000)
try_average(100, 10000, ai_always_double)
