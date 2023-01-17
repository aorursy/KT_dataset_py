from itertools import groupby, chain

import numpy as np



# This is game logic written in Python. This is a modified version of

# Patrick Westerhoff's code: https://gist.github.com/poke/6934842

class Game:

    NONE = '.'

    RED = 'R'

    YELLOW = 'Y'

    DRAW = 'D'

    

    def __init__ (self, starts = 'R', cols = 7, rows = 6, requiredToWin = 4):

        """Create a new game."""

        self.cols = cols

        self.rows = rows

        self.win = requiredToWin

        self.board = [[self.NONE] * rows for _ in range(cols)]

        self.turn = starts

        self.actions = []

        for i in range(cols):

            self.actions.append(i)

        

    def diagonalsPos (self, matrix, cols, rows):

        """Get positive diagonals, going from bottom-left to top-right."""

        for di in ([(j, i - j) for j in range(cols)] for i in range(cols + rows -1)):

            yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]



    def diagonalsNeg (self, matrix, cols, rows):

        """Get negative diagonals, going from top-left to bottom-right."""

        for di in ([(j, i - cols + j + 1) for j in range(cols)] for i in range(cols + rows - 1)):

            yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]



    def insert (self, column):

        """Insert the color in the given column."""

        c = self.board[column]

        if c[0] != self.NONE:

            raise Exception('Column is full')



        i = -1

        while c[i] != self.NONE:

            i -= 1

        c[i] = self.turn

        

        if len(self.getPossibleActions()) == 0:

            winner = Game.DRAW

        else:

            winner = self.getWinner()

            if winner:

                self.actions = []

            else:

                self.actions = self.getPossibleActions()

        

        self.turn = self.YELLOW if self.turn == self.RED else self.RED

        

        return winner



    def getWinner (self):

        """Get the winner on the current board."""

        lines = (

            self.board, # columns

            zip(*self.board), # rows

            self.diagonalsPos(self.board, self.cols, self.rows), # positive diagonals

            self.diagonalsNeg(self.board, self.cols, self.rows) # negative diagonals

        )



        for line in chain(*lines):

            for color, group in groupby(line):

                if color != self.NONE and len(list(group)) >= self.win:

                    return color

                

    def getPossibleActions(self):

        actions = []

        for i, col in enumerate(self.board):

            if col[0] == self.NONE:

                actions.append(i)

        return actions

        

    def printBoard (self):

        """Print the board."""

        print('  '.join(map(str, range(self.cols))))

        for y in range(self.rows):

            print('  '.join(str(self.board[x][y]) for x in range(self.cols)))

        print()

import pandas as pd

    

# Environment converts board to so the agent can understand it.

# Agent understands a 7x6 array matrix where:

# 0 = no piece

# 1 = my piece

# 2 = opponent piece

class Environment:

    def __init__(self):

        self.game = Game()

    

    def reset(self, starts=Game.RED):

        self.game = Game(starts=starts)

    

    # Retunrns board in correct format and possible actions in this state.

    def get_obs(self, player):

        board = np.copy(self.game.board)

        df = pd.DataFrame(board)

        df = df.applymap(lambda x: self.convert(x, player))

        return df.values, self.game.getPossibleActions()



    def convert(self, x, player):

        if x == Game.NONE:

            return 0

        elif x == player:

            return 1

        else:

            return 2

    

    def step(self, action):

        assert action in self.game.getPossibleActions(),"Action " + str(action) + " not in " + str(self.game.getPossibleActions())

        win = self.game.insert(action)

        return win
import random



class Agent:

    def __init__(self, model, epsilon=0.1):

        self.model = model

        self.epsilon = epsilon

    

    # Play turn based on witch player's turn is.

    def play_turn(self, env, player, explore=True):

        board, possible_actions = env.get_obs(player)

        board = to_categorical(board, 3)

        actions = to_categorical(possible_actions, env.game.cols)

        max_val = float("-inf")

        chosen_action = -1

        # Evaluate all actions from given state and take the highest value action

        if not explore or random.random() > self.epsilon:

            boards = np.repeat(board[None,...], len(possible_actions), axis=0)

            valuePred = model.predict([boards, actions])

            for i, action in enumerate(possible_actions):

                if valuePred[i] > max_val:

                    max_val = valuePred[i]

                    chosen_action = action

        else:

            chosen_action = random.choice(possible_actions)

        # For some really small percentage of times the model gives out a value of NaN.

        # In these cases the agent is unable to choose action so a random action is taken

        if chosen_action == -1:

            chosen_action = random.choice(possible_actions)

        winner = env.step(chosen_action)

        return winner, board, to_categorical(chosen_action, env.game.cols)

    

    # Play one episode until either player wins while storing the states and actions.

    # Afterwards correcting the state, action values based on the observed plays

    def _one_episode(self, env):

        env.reset(Game.RED)

        red_states = []

        red_actions = []

        yellow_states = []

        yellow_actions = []

        play = True



        while play:

            if env.game.turn == Game.RED:

                winner, board, chosen_action = self.play_turn(env, Game.RED)

                red_states.append(board)

                red_actions.append(chosen_action)

            else:

                winner, board, chosen_action = self.play_turn(env, Game.YELLOW)

                yellow_states.append(board)

                yellow_actions.append(chosen_action)

            if winner:

                play = False



        if winner == Game.RED:

            red_reward = 1

            yellow_reward = -1

        elif winner == Game.YELLOW:

            red_reward = -1

            yellow_reward = 1

        elif winner == Game.DRAW:

            red_reward = 0

            yellow_reward = 0

        

        yellow_rewards = np.empty(len(yellow_states))

        red_rewards = np.empty(len(red_states))

        

        # Distribute the rewards

        lastIndex = len(yellow_states)-1

        for i in range(lastIndex):

            yellow_rewards[i] = i/lastIndex * yellow_reward

        lastIndex = len(red_states)-1

        for i in range(lastIndex):

            red_rewards[i] = i/lastIndex * red_reward

        

        states = np.append(yellow_states, red_states, axis=0)

        rewards = np.append(yellow_rewards, red_rewards, axis=0)

        actions = np.append(yellow_actions, red_actions, axis=0)

        return states, actions, rewards

        

    # Train the model with multiple games. The number of games is how many games

    # are played before a value update is applied. Update is done in whole state,

    # action, reward set until the set becomes bigger than 400. After that the 

    # old sets are being discarded and not used anymore in the model correction.

    def fit(self, env, games_per_update=1, updates=1, verbose=True):

        mae = []

        loss = []

        states, actions, rewards = self._one_episode(env)

        

        for j in range(updates):

            if len(states) > 400:

                states = states[len(states)-100:]

                actions = actions[len(actions)-100:]

                rewards = rewards[len(rewards)-100:]

            

            for i in range(games_per_update):

                s, a, r = self._one_episode(env)

                states = np.append(states, s, axis=0)

                actions = np.append(actions, a, axis=0)

                rewards = np.append(rewards, r, axis=0)

            

            history = self.model.fit([states, actions],

                              rewards,

                              batch_size=1,

                              epochs=1,

                              verbose=verbose)

            

            mae.append(history.history['mean_absolute_error'])

            loss.append(history.history['loss'])

        

        return mae, loss

    
import numpy as np

import matplotlib.pyplot as plt

from keras.models import Model

from keras import layers

from keras.utils import to_categorical
# Board input

input_state = layers.Input(shape=(7, 6, 3), name='state_input')

x = layers.Conv2D(32,(4,4))(input_state)

x = layers.Activation('relu')(x)

x = layers.Conv2D(8,(2,2))(x)

x = layers.Activation('relu')(x)

x = layers.Flatten()(x)

x = layers.Dense(22)(x)

x = layers.Activation('sigmoid')(x)



# Action input

input_action = layers.Input(shape=(7,), name='action_input')

y = layers.Dense(7)(input_action)

y = layers.Activation('sigmoid')(y)



# Combined to one

x = layers.concatenate([x, y])

x = layers.Dense(8)(x)

x = layers.Activation('tanh')(x)

out = layers.Dense(1, activation='tanh')(x)



model = Model(inputs=[input_state, input_action], outputs=out)

model.compile(optimizer='rmsprop',

              loss='mse',

              metrics=['mae'])

print(model.summary())
environment = Environment()

agent = Agent(model, epsilon=0.2)

mae, loss = agent.fit(environment,

                      games_per_update=1,

                      updates=1000,

                      verbose=False)

agent.model.save_weights('connect_four_agent_weights.h5')



plt.plot(mae)

plt.title('Model MAE')

plt.ylabel('mae')

plt.xlabel('Epoch')

plt.show()

plt.plot(loss)

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('Epoch')

plt.show()
play = True

env = Environment()

agent = Agent(model)

agent.model.load_weights('connect_four_agent_weights.h5')

while play:

    # Player

    env.game.printBoard()

    row = input('{}\'s turn: '.format('Red' if env.game.turn == env.game.RED else 'Yellow'))

    win = env.game.insert(int(row))

    if win:

        print('Player won')

        env.game.printBoard()

        play = False

    else:

        # AI agent

        win, b, a = agent.play_turn(env, env.game.turn, explore=False)

        if win:

            print('Ai win')

            env.game.printBoard()

            play = False