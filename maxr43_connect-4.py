import os

import numpy as np

import matplotlib.pyplot as plt

from random import choice

from tensorflow.keras import layers, models, metrics

from kaggle_environments import make, evaluate


# Creates a model

model = models.Sequential()

model.add(layers.Conv2D(32, (2,2), activation='relu', data_format='channels_first', input_shape=(1,6,7)))

model.add(layers.Conv2D(64, (2,2), activation='relu'))

model.add(layers.Conv2D(64, (2,2), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dense(7,activation='softmax'))





model.compile(

    optimizer='adam',

    loss='mse',

    metrics=[metrics.MeanSquaredError()]



)



model.summary()


# reads and forms the data

def read_dataset(dataset_path):

    with open(dataset_path, 'rb') as file:

        size = os.fstat(file.fileno()).st_size

        data = np.load(file)

        while file.tell() < size:

            data = np.vstack(data, np.load(file))

        return data



states = read_dataset('../input/connect-4-training-data/training_data_states.npy')

preds = read_dataset('../input/connect-4-training-data/training_data_preds.npy')



states = np.expand_dims(states, axis=1)



history = model.fit(states, preds, epochs=40, verbose=1)

plt.plot(history.history['mean_squared_error'])
#This agent checks for different valid moves using the dat a set

def agent(obs):

    board = obs.board

    valid_moves = [i for i in range(7) if board[i] == 0]

    

    state = np.array(board, dtype=np.int8).reshape(1,1,6,7)

    preds = model(state)

    

    preds = np.squeeze(preds)

    next_move = np.argmax(preds)

    next_move = int(next_move)

    

    if next_move in valid_moves:

        return next_move

    else:

        return choice(valid_moves)
#this evaluates the results of each game to give us how good the machine is

evaluation_result = evaluate('connectx', [agent, 'random'], num_episodes=1000)



wins = 0

for score in evaluation_result:

    if score == [1,-1]:

        wins += 1



        print('Neural network\'s win rate against a random agent in 1000 games is:', str(wins/10)+'%')
# model = Sequential([

#     Flatten(input_shape=(6,7)),

#     Dense(10),

#     Dense(7, activation='softmax')

# ])



# model.summary()



# board_state = np.arange(42).reshape(1,6,7)



# move_probability = model(board_state)[0]

# tf.argmax(move_probability).numpy()
# from typing import Tuple

# from kaggle_environments import make, evaluate

# import numpy as np

# import random



# class Agent:

#     def __init__(self, board, player):

#         self.board    = board

#         self.T        = board.T 

#         self.stones   = np.count_nonzero(board)

#         self.player   = player

#         self.opponent = 1 if player == 2 else 2

    

#     def play(self, move, color):

#         self.board[move] = color

        

#     def revert(self, move):

#         self.board[move] = 0

        

#     def valid_moves(self):

#         return [

#             (5 - np.count_nonzero(self.T[column]), column) \

#             for column in range(7) \

#             if np.count_nonzero(self.T[column]) < 6

#         ]

    

#     @staticmethod

#     def windows(arr, col):

#         '''

#         Returns consecutive subarrays of size 4

#         that contains the entry at index `col` in 1d array `arr`,

#         '''

#         assert arr.ndim == 1 and len(arr) > 4 and col in range(len(arr))



#         if col < 4:

#             end = min(col, len(arr)-4) + 1

#             return (arr[i:i+4] for i in range(end))

        

#         start = col-3

#         end   = min(col, len(arr)-4) + 1

#         return (arr[i:i+4] for i in range(start, end))

    

#     def is_col_connect_four(self, move, color):

#         '''

#         Check if there is any connected-4 in the column of `move`.

#         '''

#         row, col = move

#         return (self.T[col, row:row+4] == color).all() \

#             if row <= 2 \

#             else False

#     def is_row_connect_four(self, move, color):

#         '''

#         Check if there is any connected-4 in the row of `move`.

#         '''

#         row, col = move

#         row = self.board[row]

#         return any((arr == color).all() for arr in self.windows(row, col))

    

#     def is_diags_connect_four(self, move, color):

#         '''

#         Check if there is any connected-4 in both diagonals that passed through `move`.

#         '''

#         row, col     = move

#         major_offset = col - row

#         minor_offset = row + col - 6

    

#     # make sure major diagonal is at least of length 4

#         if major_offset in range(-2, 4):

#             major = np.diagonal(self.board, offset=major_offset)

#             if len(major) == 4:

#                 if (major == color).all():

#                     return True

#             else:

#                 # column of `move` in the major diagonal

#                 major_col = min(row, col)

#                 if any((arr == color).all() for arr in self.windows(major, major_col)):

#                     return True

#         # make sure major diagonal is at least of length 4

#         if minor_offset in range(-3, 3):

#             minor = np.diagonal(np.rot90(self.board), offset=minor_offset)

#             if len(minor) == 4:

#                 if (minor == color).all():

#                     return True

#             else:

#                 # column of `move` in the minor diagonal

#                 minor_col = min(row, 6-col)

#                 if any((arr == color).all() for arr in self.windows(minor, minor_col)):

#                     return True

#         return False

#     def is_winning(self, move, color):

#         self.play(move, color)

        

#         if self.is_col_connect_four(move, color) \

#         or self.is_row_connect_four(move, color) \

#         or self.is_diags_connect_four(move, color):

#             self.revert(move)

#             return True

        

#         self.revert(move)

#         return False

    

#     def is_double_threat(self, move, color):

#         prev_threats = set()

#         for each_move in self.valid_moves():

#             if self.is_winning(each_move, color):

#                 prev_threats.add(each_move)

                

#         self.play(move, color)

        

#         curr_threats = set()

#         for each_move in self.valid_moves():

#             if self.is_winning(each_move, color):

#                 curr_threats.add(each_move)

                

#         self.revert(move)

        

#         return len(curr_threats) - len(prev_threats) >= 2



# def MOST_POWERFUL_AI(observation):

#     game_board = observation.board

#     game_board = np.array(game_board, dtype=np.int8).reshape(6,7)

    

#     player_color = observation.mark

#     opponent_color = 1 if player_color == 2 else 2

#     agent = Agent(game_board, player_color)

    

#     for color in [player_color, opponent_color]: 

#         for column in range(7):

#             for move in agent.valid_moves():

#                 if agent.is_winning(move, color):

#                     return move[1]

#     bad_moves = []



#     for move in agent.valid_moves():

#         agent.play(move, player_color)

#         for next_move in agent.valid_moves():

#             if agent.is_winning(next_move, opponent_color):

#                 bad_moves.append(move)

#                 break

#         agent.revert(move)

    

#     #print('At step',np.count_nonzero(game_board), 'bad moves are:', bad_moves)

#     good_moves = set(agent.valid_moves()) - set(bad_moves)

#     if good_moves:

        

#         for each_move in good_moves:

#             if agent.is_double_threat(each_move, player_color):

#                 return each_move[1]

            

#         for each_move in good_moves:

#             if agent.is_double_threat(each_move, opponent_color):

#                 #print('Found Double Threat of opponent at', each_move)

#                 return each_move[1]

            

#         return random.choice(list(good_moves))[1]

    

#     return random.choice([column for column in range(7) if game_board[0, column] == 0])



# env = make("connectx", debug=True)

# env.play([None, MOST_POWERFUL_AI])

# # env.run(['negamax', MOST_POWERFUL_AI])

# # env.render(mode="ipython")
# position = Tuple[int, int]



# class Board:

#     def __init__(self, data):

#         self.data = data

#         self.indices = np.arange(42, dtype=np.int8).reshape(6,7)

#         self.rotated = np.rot90(self.indices)

        

#     def row(self, index: int):

#         return self.data[index]

    

#     def col(self,index: int):

#         return self.data.T[index]

    

#     def rows(self, start: int, end: int):

#         return self.data[start:end+1]

    

#     def cols(self,start: int,end: int):

#         return self.data.T[start:end+1]

    

#     def play(self,move: position,color: int):

#         self.data[move] = color

     

#     def remove(self, move: position):

#         self.data[move] = 0

        

#     def possible_move_at(self, col: int):

        

#         row = 5 - np.count_nonzero(self.col(col))

#         if row < 0:

#             return None

        

#         return row, col

#     def possible_moves(self):

#         return [

#             self.possible_move_at(col) for col in range(7) \

#             if self.possible_move_at(col) != None

#         ]

        

#     def is_winning_at_col(self, move: position, color: int):

#         row, col = move

#         if row <= 2:

#             self.play(move,color)

#             colum_array = self.col(col)

#             stones = colum_array[row: row+4]

#             if (stones == color).all():

#                 self.remove(move)

#                 return True

#         self.remove(move)

#         return

    

#     def is_winning_at_row(self, move: position, color: int):

#         table = {

#             0: [[0, 1, 2, 3]],

#             1: [[0, 1, 2, 3], [1, 2, 3, 4]],

#             2: [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],

#             3: [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],

#             4: [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],

#             5: [[2, 3, 4, 5], [3, 4, 5, 6]],

#             6: [[3, 4, 5, 6]]

#         }

        

#         self.play(move, color)

        

#         row_idx, col_idx = move

#         segment_indices = table[col_idx]

#         row = self.row(row_idx)

        

        



            

#         self.remove(move)

#         return False

    

#     def is_winning_at_diagonals(self, move: position, color: int):

#         five_stones_table = {

#             0: [[0, 1, 2, 3]],

#             1: [[0, 1, 2, 3], [1, 2, 3, 4]],

#             2: [[0, 1, 2, 3], [1, 2, 3, 4]],

#             3: [[0, 1, 2, 3], [1, 2, 3, 4]],

#             4: [[1, 2, 3, 4]]

#         }

#         six_stones_table = {

#             0: [[0, 1, 2, 3]],

#             1: [[0, 1, 2, 3], [1, 2, 3, 4]],

#             2: [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],

#             3: [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],

#             4: [[1, 2, 3, 4], [2, 3, 4, 5]],

#             5: [[2, 3, 4, 5]]

#         }

#         lookup_tables = {

#             5: five_stones_table,

#             6: six_stones_table

#         }

        

#         self.play(move, color)

        

#         row, col = move

#         move_index = row * 7 + col

        

#         major_diagonal_index = np.diagonal(self.indices, offset=col-row)

#         minor_diagonal_index = np.diagonal(self.rotated, offset=col+row-6)

        

#         flat_board = self.data.reshape(42)

        

#         for diagonal in [major_diagonal_index, minor_diagonal_index]:

#             if len(diagonal) >= 4:

#                 if len(diagonal) == 4:

#                     stones = flat_board[diagonal]

#                     if (stones == color).all():

#                         self.remove(move)

#                     return True

#                 else:

#                     segment_indices_table = lookup_tables[len(diagonal)]

#                     idx = np.where(diagonal == move_index)[0][0]

#                     segment_indices = segment_indices_table[idx]

#                     for segment in segment_indices:

#                         stones = flat_board[diagonal[segment]]

#                         if (stones == color).all():

#                             self.remove(move)

#                             return True

        

        

#         self.remove(move)

#         return False

    

#     def is_winning_immediately(self, move: position, color:int) -> bool:

#         return self.is_winning_at_col(move, color)\

#         or self.is_winning_at_row(move, color)\

#         or self.is_winning_at_diagonals(move, color)

    

#     def is_losing_next_move(self, move: position, color:int) -> bool:

#         row, col = move

        

#         if row > 0:

#             next_move = row- 1, col

            

#             opponent = 1 if color ==2 else 2

#             if self.is_winning_immediately(next_move, opponent):

#                 return True

#         return False

        

    

# def analyze(board,player):

#     moves = board.possible_moves()

#     opponent = 1 if player == 2 else 2

#     for current_move in moves:

#         if board.is_winning_immediately(current_move, player):

#             return current_move[1]

        

#     for current_move in moves:

#         if board.is_winning_immediately(current_move, opponent):

#             return current_move[1]

        

#     safe_moves = set()

#     for current_move in moves:

#         if board.is_losing_next_move(current_move, player):

#             safe_moves.add(current_move)

            

#     if safe_moves:

#         return random.choice(list(safe_moves))[1]

#     return random.choice(moves)[1]



# def my_agent(observation):

#     array = observation.board

#     array = np.asarray(array,dtype=np.int8).reshape(6,7)

    

#     board = Board(data=array)    

#     player = observation.mark

    

#     return analyze(board,player)



# env = make("connectx", debug=True)

# trainer = env.train([None, "random"])

# obs = trainer.reset()

# for _ in range(1000):

#     action = my_agent(obs)

#     obs, reward, done, info = trainer.step(action)

#     if done:

#         obs = trainer.reset()





# env = make('connectx', debug=True)

# env.play([None,my_agent])

# env.render(mode='ipython',width=700,height=600)