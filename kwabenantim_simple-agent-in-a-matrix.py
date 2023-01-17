!pip install 'kaggle-environments>=0.1.6'
import os

import sys

import inspect

import numpy as np

from kaggle_environments import evaluate, make, utils
env = make('connectx', debug=True)
# https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0

# https://github.com/denkspuren/BitboardC4/blob/master/BitboardDesign.md



def agent(obs, config):

    from random import choice

    import numpy as np

    

    MARK_A = obs.mark

    MARK_B = 3 - MARK_A

    

    ROWS = config.rows

    COLS = config.columns

    INAROW = config.inarow

    

    MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]

    

    TOP_MASK  = int(('1'+'0'*ROWS)*COLS, 2)  # 283691315109952

    GRID_MASK = int(('0'+'1'*ROWS)*COLS, 2)  # 279258638311359

        

    def get_bitmap(obs):

        board = np.asarray(obs.board).reshape(ROWS, COLS)

        board = np.insert(board, 0, 0, axis=0)

        board = np.flip(np.flipud(board).flatten(order='F'))



        str_bitboard = ''.join(map(str, np.int8(board==MARK_A).tolist()))

        str_bitmask = ''.join(map(str, np.int8(board!=0).tolist()))

        return int(str_bitboard, 2), int(str_bitmask, 2)



    def is_win(bitboard):

        for direction in [1, 7, 6, 8]:  # | - \ / 

            bb = bitboard & (bitboard >> direction)

            if (bb & (bb >> (2 * direction))) != 0: 

                return True

        return False

    

    def play(bitboard, bitmask, c):

        xboard = bitboard ^ bitmask

        xmask = bitmask | (bitmask + (1 << (c * COLS)))        

        xboard = xboard ^ xmask

        return xboard, xmask



    def play_to_win(bitboard, bitmask, c):

        xboard, xmask = play(bitboard, bitmask, c)        

        if is_win(xboard): 

            return True

        return False

    

    def valid_moves(bitmask):

        moves = []

        for c in MOVE_ORDER:

            xmask = bitmask | (bitmask + (1 << (c * COLS)))

            if (TOP_MASK & xmask) == 0:

                moves.append(c)

        return moves

        

    def count_ones(bitboard, bitmask):

        zeros = (~bitmask) & GRID_MASK

        count = 0

        for d1 in [1, 7, 6, 8]:  # | - \ / 

            d2 = 2 * d1

            d3 = 3 * d1

            bb = ((bitboard & (zeros >> d1) & (zeros >> d2) & (zeros >> d3)) | 

                  (zeros & (bitboard >> d1) & (zeros >> d2) & (zeros >> d3)) | 

                  (zeros & (zeros >> d1) & (bitboard >> d2) & (zeros >> d3)) | 

                  (zeros & (zeros >> d1) & (zeros >> d2) & (bitboard >> d3)))           

            count += bin(bb).count('1')

        return count

        

    def count_twos(bitboard, bitmask):

        zeros = (~bitmask) & GRID_MASK

        count = 0

        for d1 in [1, 7, 6, 8]:  # | - \ / 

            d2 = 2 * d1

            d3 = 3 * d1

            bb = ((bitboard & (bitboard >> d1) & (zeros >> d2) & (zeros >> d3)) | 

                  (bitboard & (zeros >> d1) & (bitboard >> d2) & (zeros >> d3)) | 

                  (bitboard & (zeros >> d1) & (zeros >> d2) & (bitboard >> d3)) | 

                  (zeros & (bitboard >> d1) & (bitboard >> d2) & (zeros >> d3)) | 

                  (zeros & (bitboard >> d1) & (zeros >> d2) & (bitboard >> d3)) | 

                  (zeros & (zeros >> d1) & (bitboard >> d2) & (bitboard >> d3)))              

            count += bin(bb).count('1')

        return count



    def count_threes(bitboard, bitmask):

        zeros = (~bitmask) & GRID_MASK

        count = 0

        for d1 in [1, 7, 6, 8]:  # | - \ / 

            d2 = 2 * d1

            d3 = 3 * d1

            bb = ((bitboard & (bitboard >> d1) & (bitboard >> d2) & (zeros >> d3)) | 

                  (bitboard & (bitboard >> d1) & (zeros >> d2) & (bitboard >> d3)) | 

                  (bitboard & (zeros >> d1) & (bitboard >> d2) & (bitboard >> d3)) | 

                  (zeros & (bitboard >> d1) & (bitboard >> d2) & (bitboard >> d3)))            

            count += bin(bb).count('1')

        return count

            

    def heuristic(bitboard, bitmask, c):

        aboard, xmask = play(bitboard, bitmask, c)

        a1 = count_ones(aboard, xmask)

        a2 = count_twos(aboard, xmask)

        a3 = count_threes(aboard, xmask)

        

        bboard, xmask = play(bitboard ^ bitmask, bitmask, c)

        b1 = count_ones(bboard, xmask)

        b2 = count_twos(bboard, xmask)

        b3 = count_threes(bboard, xmask)

                

        score = 160 * a3 + 160 * b3 + 40 * a2 + 40 * b2 + 10 * a1 + 10 * b1

        return score

        

    def act(obs):        

        bitboard, bitmask = get_bitmap(obs)

        

        # start in the middle

        if bitmask == 0: 

            return COLS//2  



        good_moves = valid_moves(bitmask)

        

        # play the only option

        if len(good_moves) == 1:

            return good_moves[0]  

        

        # play to win if you can

        for c in good_moves:

            if play_to_win(bitboard, bitmask, c):

                return c  

            

        # avoid setting up a win

        bad_moves = set()

        for c in good_moves:

            xboard, xmask = play(bitboard, bitmask, c)

            xboard_b = xboard ^ xmask

            next_moves = valid_moves(xmask)

            for cx in next_moves:

                if play_to_win(xboard_b, xmask, cx):

                    bad_moves.add(c)  

        good_moves = list(set(good_moves) - bad_moves)

        

        # block a win if you can

        bitboard_b = bitboard ^ bitmask

        for c in good_moves:

            if play_to_win(bitboard_b, bitmask, c):

                return c  

            

        # play a heuristic move if you can

        best_score = -float('inf')

        best_moves = []

        for c in good_moves:

            score = heuristic(bitboard, bitmask, c)   

            

            if score > best_score:

                best_score = score

                best_moves = [c]

            elif score == best_score:

                best_moves.append(c)

                

        if best_moves:

            return choice(best_moves)

        

        # random fallback

        return choice(valid_moves(bitmask))  

    

    return act(obs)
# https://www.kaggle.com/hubcity/a-more-useful-negamax-opponent-connectx



def negarand(obs, config):

    from random import random

    from kaggle_environments import make

    

    env = make('connectx')    

    if random() < 0.2:

        return env.agents.random(obs, config)

    return env.agents.negamax(obs, config)
config = env.configuration

obs = env.train([None, 'random']).reset()

obs.mark = 2

board = np.asarray([[0, 0, 0, 0, 0, 0, 0],

                    [0, 0, 0, 0, 0, 0, 0],

                    [0, 0, 0, 0, 0, 0, 0],

                    [0, 0, 0, 0, 0, 0, 0],

                    [0, 0, 0, 0, 0, 0, 0],

                    [0, 0, 0, 1, 0, 0, 0]])

obs.board = board.flatten().tolist()
# %%timeit -r5 -n5

move = agent(obs, config)
# %%timeit -r5 -n5

move = negarand(obs, config)
# %%timeit -r5 -n5

move = env.agents.negamax(obs, config)
env.reset()

env.run([agent, agent])

env.render(mode='ipython', width=500, height=450)
def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)
rewards = evaluate('connectx', [agent, 'random'], num_episodes=100)

print('Agent v Random:', mean_reward(rewards))



rewards = evaluate('connectx', ['random', agent], num_episodes=100)

print('Random v Agent:', mean_reward(rewards))
rewards = evaluate('connectx', [agent, 'negamax'], num_episodes=100)

print('Agent v NegaMax:', mean_reward(rewards))



rewards = evaluate('connectx', ['negamax', agent], num_episodes=100)

print('NegaMax v Agent:', mean_reward(rewards))
rewards = evaluate('connectx', [agent, negarand], num_episodes=100)

print('Agent v NegaRand:', mean_reward(rewards))



rewards = evaluate('connectx', [negarand, agent], num_episodes=100)

print('NegaRand v Agent:', mean_reward(rewards))
# https://www.kaggle.com/ajeffries/connectx-getting-started



with open('submission.py', 'w') as file:

    file.write(inspect.getsource(agent))



out = sys.stdout

submission = utils.read_file('/kaggle/working/submission.py')

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make('connectx', debug=True)

env.run([agent, agent])

print('Success!' if env.state[0].status == env.state[1].status == 'DONE' else 'Failed...')