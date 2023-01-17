# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import random



# Each move is evaluated as to how good it was given the current grid

class Game:



    def __init__(self,grid):

        self.training_history = []

        self.grid = grid



    def check_win(self, grid):

        # Finds the positions of the Xs or Os in the grid

        resX = [np.where(grid == -1), np.where(np.transpose(grid) == -1)]

        resO = [np.where(grid == 1), np.where(np.transpose(grid) == 1)]



        # Gets the coordinates of the places occupied by X or O

        zipX=list(zip(np.where(grid == -1)[0],np.where(grid == -1)[1]))

        zipO=list(zip(np.where(grid == 1)[0],np.where(grid == 1)[1]))



        # Check if the X positions result in a win

        if self._check_row_col(resX):

            return "X WIN"

        # Check if the O positions result in a win

        elif self._check_row_col(resO):

            return "O WIN"

        # Check the diagonals

        elif self._check_diagonals(zipX):

            return "X WIN"

        elif self._check_diagonals(zipO):

            return "O WIN"

        # Check if the board is full

        elif np.where(grid == 0)[0].size == 0:

            # If there are no lines of 3 for XorO and the grid contains no more empty spaces

            return "DRAW"





    # Checks that the positions give a win irrespective of whether it is Os or Xs

    def _check_row_col(self, res):

        for g in res:

            if any(sublist in np.array_str(np.array(g[0])) for sublist in ('0 0 0', '1 1 1','2 2 2')):

                if '0 1 2' in np.array_str(np.array(g[1])):

                    return True



    def _check_diagonals(self, res):

        # the diagonals

        if (0,0) in res and (1,1) in res and (2,2) in res:

            return True

        if (0,2) in res and (1,1) in res and (2,0) in res:

            return True



    # This gets the current grid

    def get_grid(self):

        return self.grid



    def make_move(self,pos,XorO):

        if self.grid[pos] == 0:

            self.grid[pos] = XorO

            return True

        else:

            return False
from keras.layers import Dense

from keras.models import Sequential

from keras.utils import to_categorical

import numpy as np





class TicTacToeModel:



    def __init__(self, numberOfInputs, numberOfOutputs, epochs, batchSize):

        self.epochs = epochs



        self.batchSize = batchSize

        self.numberOfInputs = numberOfInputs

        self.numberOfOutputs = numberOfOutputs

        self.model = Sequential()

        self.model.add(Dense(64, activation='relu', input_shape=(numberOfInputs, )))

        self.model.add(Dense(128, activation='relu'))

        self.model.add(Dense(128, activation='relu'))

        self.model.add(Dense(128, activation='relu'))

        self.model.add(Dense(numberOfOutputs, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



    def train(self, dataset):

        input = []

        output = []

        for data in dataset:

            input.append(data[1])

            output.append(data[0])



        X = np.array(input).reshape((-1, self.numberOfInputs))

        y =to_categorical(output, num_classes=3)

        

        # Train and test data split this gives 80%

        boundary = int(0.8 * len(X))

        X_train = X[:boundary]

        X_test = X[boundary:]

        y_train = y[:boundary]

        y_test = y[boundary:]

        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs, batch_size=self.batchSize)

        return self.model





    # This is to use a new model that has been loaded

    def set_model(self, model):

        self.model = model



    def predict(self, data, index):

        return self.model.predict(np.array(data).reshape(-1, self.numberOfInputs))[0][index]
import random

import copy

import numpy as np





class Agent:



    def __init__(self):

        pass



    # This returns the location of the position on the grid to place the X or O

    def set_location(self,grid,method, model):

        # Find available positions

        # select a location to place the xoro

        if method == "random":

            select = _get_random_position(grid)

        if method == "neural":

            select = _get_neural_position(grid, model)

        return select



# Static private methods

# Returns the next position of the xoro randomly from the available positions

def _get_random_position(grid):

    # Find positions the contain a 0 (blank)

    available = _get_available_positions(grid)

    return random.choice(available)



# This is where the Neural network goes.

def _get_neural_position(grid, model):

    availableMoves = _get_available_positions(grid)

    maxValue = 0

    bestMove = availableMoves[0]

    for availableMove in availableMoves:

        # get a copy of a board

        boardCopy = copy.deepcopy(grid)

        value = model.predict(boardCopy, 0)

        if value > maxValue:

            maxValue = value

            bestMove = availableMove

    selectedMove = bestMove

    return selectedMove



def _get_available_positions(grid):

    a = np.where(grid == 0)

    return list(zip(a[0], a[1]))
import copy

import numpy as np

import tensorflow as tf



history = []



# Main game loop

def run_game(player1, player2, loaded_model, iterations, print_grid, show_result):

    grid = np.full((3, 3), 0)



    # create the game

    g = Game(grid)

    agent1 = Agent()

    agent2 = Agent()



    score = 0

    moves = []

    output = 0

    # iterations = 1000

    it = 0

    X_wins = 0

    O_wins = 0

    Draws = 0

    while it < iterations:

        # The moves for each game

        for i in ["X", "O"]:

            while True:

                if i == "X":

                    loc = agent1.set_location(grid,player1,loaded_model)

                    # Make sure the move is to a blank space before exiting the loop

                    if g.make_move(loc,-1):

                        break

                if i == "O":

                    loc = agent2.set_location(grid,player2,loaded_model)

                    # Make sure the move is to a blank space before exiting the loop

                    if g.make_move(loc,1):

                        break

            if print_grid:

                print(grid)

            res = g.check_win(grid)

            last_state = grid.tolist()

            moves.append(last_state)



            # Goes here if there is a result

            if res:

                if show_result:

                    print(res)

                # X wins

                if res[:1] == 'X':

                    X_wins+=1

                    grid = np.full((3, 3), 0)

                    g = Game(grid)

                    output=-1

                # O wins

                elif res[:1] == 'O':

                    O_wins+=1

                    grid = np.full((3, 3), 0)

                    g = Game(grid)

                    output=1

                # Draw

                else:

                    Draws+=1

                    grid = np.full((3, 3), 0)

                    g = Game(grid)

                    output=0

                it += 1

                # If the game is won by less than nine moves then append the last board state to make the array 9 long.

                # Not sure how to make Keras deal with uneven data sizes

                while len(moves) < 9:

                    moves.append(last_state)

                for m in moves:

                    history.append((output,copy.deepcopy(m)))

                moves = []

                break

    return history,X_wins, O_wins, Draws
# Create some random training data

print("--- Summary ---")

history,x,o,d=run_game(player1="random", player2="random", loaded_model=None, iterations=100, print_grid=False, show_result=False)

# Get the results for these games to show later

xw=str(x)

ow=str(o)

dd=str(d)

# print out the results

print("Before training (Random vs Random)")

print("X Wins = "+xw)

print("O Wins = "+ow)

print("Draws = "+dd)
# Read in the raw data from the file

dataset = pd.read_csv('/kaggle/input/tic-tac-toe-endgame-dataset/tic-tac-toe.data')

dataset.head(5)

# Extract this data into two datasets. Map Ordinal Values To Integers for ML

X = dataset.iloc[:, 0:9]

yd = dataset.iloc[:,9:10]



# transform the 'x' into -1, 'o' into 1 and the 'b' into 0

player_dict = {

    'x':-1,

    'o':1,

    'b':0

}

X = X.replace(player_dict)



# transform the results for an X win.

outcome_dict= {

    'positive':-1,

    'negative':1

}

yd = yd.replace(outcome_dict)



print(X.head(5))

print(yd.head(5))
# Convert the data into numpy arrays for reshaping.

# Xr = an array to hold the 1D results [-1,-1,-1,-1,1,1,-1,1,-1] to a 3D array [[-1,-1,-1],[-1,1,-1],[-1,1,-1]]

Xr=[]

for a in X.values:

    Xr.append(a.reshape(-1,3))

    

history=[]



# append the game outcome to the array of endstate game positions, converting to python lists

for x in Xr:

    for y in yd.values:

        history.append((y[0],x.tolist()))



# show the first 5 rows of the transformed data

history[:5]
# Train the network using the results from the random games

#### CAUTION: If you are using the tic-tac-toe predefined training data then make the number of epochs very small, e.g. 1-10 otherwise it'll take ages.

# NOTE: epochs are the number fo times it passses through the data.

ticTacToeModel = TicTacToeModel(9, 3, 1, 32)

# For much smaller amounts of randomly generated data use the following, with 100 epochs

#ticTacToeModel = TicTacToeModel(9, 3, 100, 32)

model = ticTacToeModel.train(history)



print("Before training (Random vs Random)")

print("X Wins = "+xw)

print("O Wins = "+ow)

print("Draws = "+dd)
# Use the model - neural network vs random player



ticTacToeModel.set_model(model)

_, x, o, d = run_game(player1="neural", player2="random", loaded_model=ticTacToeModel, iterations=100, print_grid=False, show_result=False)



print("After Learning (Neural = X vs Random = O):")

print("X Wins = "+str(x))

print("O Wins = "+str(o))

print("Draws = "+str(d))