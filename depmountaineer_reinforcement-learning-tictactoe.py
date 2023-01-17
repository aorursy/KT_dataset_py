import numpy as np

import sys

from itertools import product
class TicTacToe:

    """

    Game environment for Tic Tac Toe, to be played by a pair 

    """

    def __init__(self):

        self.reset()



    def get_winner(self):

        """

        Return the winner state.

        

        draw: -1

        x wins: 0

        o wins: 1

        game still in session: None

        """

        return self._winner

    

    def is_draw(self):

        """

        Return true if game is a draw.

        """

        return self._winner < 0

    

    def is_game_over(self):

        """

        Return true if game is finished.

        """

        return self._winner is not None

    

    def num_moves(self):

        """

        Return the number of moves played in this game so far.

        """

        return len(self._history) - 1

        

    def get_state(self, which):

        """

        Return a string state code of this game for player valuation. 

        

        The state is the concatenation of board values, S=self, O=other, .=nobody,

            in order from top left to bottom right, going left to right, then top to bottom.

        

        which: for which player, 0=x, 1=o 

        """

        return self._convert_state(self._state, which)

    

    def get_history(self, which):

        """

        Return a list of all states, in order, in the game so far for valuation by a player.

        

        The state is the concatenation of board values, S=self, O=other, .=nobody,

            in order from top left to bottom right, going left to right, then top to bottom.

        

        which: for which player, 0=x, 1=o 

        """

        return [self._convert_state(s, which) for s in self._history]

    

    def get_turn(self):

        """

        Whose turn is it?

        

        return: 0 for x, 1 for o, None for game over

        """

        if self.is_game_over():

            return None

        if self._x_turn():

            return 0

        if self._o_turn():

            return 1

        return None

    

    def get_valid_moves(self, which):

        """

        Return all valid moves available.

        

        Return: list of tuples, each tuple being (which, move, next_state)

        which: 0=x, 1=o

        move: (row, col) row, col each from 0 to 2 inclusive

        next_state: unique state string of the resulting game position from which's point of view,

            concatenation of board values, S=self, O=other, .=nobody,

            in order from top left to bottom right, going left to right, then top to bottom.



        """

        moves = []

        for row, col in product(range(3), range(3)):

            move = (row, col)

            pos = self._move_to_pos(move)

            if self._is_move_valid(which, pos):

                next_state = self._convert_state(self._state[:pos] + "xo"[which] + self._state[pos+1:], which)

                moves.append((which, move, next_state))

        return moves



    def play_random_move(self, which=None, verbose=0, moves=None):

        """

        Play a random move.

        

        which: which player is moving, 0=x, 1=o.  Use default for whoever's turn it is (recommended).

        verbose: nonnegative float.  higher numbers print more information. 0 for quiet.

        moves: select only from this list of (which, move, state) triples as returned by get_valid_moves.

            default is to choose randomly from every move available.        

            

        return:  0 if x wins with this move, 1 if o wins with this move, -1 if game is drawn, None if game

            is not finished, and False in case of an error (printed if verbose>0)

        """

        if which is None:

            which = self.get_turn()

        if moves is None:

            moves = self.get_valid_moves(which)

        index = np.random.randint(0, len(moves))

        (w, move, s) = moves[index]

        return self.play(which, move, verbose=verbose)

    

        

    

    def play(self, which=None, move=None, verbose=0):

        """

        Play a specified move.

        

        which: which player is moving, 0=x, 1=o.  Use default for whoever's turn it is (recommended).

        move: (row, col) to play.  If a list of (which, move, state) triples as returned by get_valid_moves, choose

            randomly from one of these.  Default is to just play a random move.

        verbose: nonnegative float.  higher numbers print more information. 0 for quiet.

        

        return:  0 if x wins with this move, 1 if o wins with this move, -1 if game is drawn, None if game

            is not finished, and False in case of an error (printed if verbose>0)

        """

        if move is None:

            return self.play_random_move(which=which, verbose=verbose)

        if hasattr(move, '__get_item__'):

            return self.play_random_move(which=which, verbose=verbose, moves=move)

        if which is None:

            which = self.get_turn()

        move_num = len(self._history)

        pos = self._move_to_pos(move)

        if self._winner is not None:

            if verbose:

                print("Error:", "Game has ended")

            return False        

        if self._state[pos] != '.':

            if verbose:

                print("Error:", "Position already played")

            return False

        if which == 0 and not self._x_turn():

            if verbose:

                print("Error:", "Not x's turn")

            return False

        if which == 1 and not self._o_turn():

            if verbose:

                print("Error:", "Not o's turn")

            return False

        self._state = self._state[:pos] + "xo"[which] + self._state[pos+1:]

        self._history.append(self._state)

        if which == 0 and self._x_won():

            self._winner = 0

            if verbose > 0:

                print(move_num, "x is the winner")

            return 0

        if which == 1 and self._o_won():

            self._winner = 1

            if verbose > 0:

                print(move_num, "o is the winner")

            return 1

        if self._board_full():

            self._winner = -1

            if verbose > 0:

                print(move_num, "game is drawn")

            return -1

        if verbose > 1:

            print(move_num, "Move accepted, game continues")

        return None

    

    def reset(self):

        """

        Reset the game to play it again.

        """

        self._state = "."*9

        self._history = [self._state]

        self._winner = None

    

    def draw(self, file=sys.stdout):

        """

        Draw the current tic tac toe board.

        

        file: where to print to.  Default is sys.stdout

        """

        print(self._state[:3], file=file)

        print(self._state[3:6], file=file)

        print(self._state[6:], file=file)



    def draw_history(self, per_row=10, from_=0, to=None, file=sys.stdout):

        """

        Draw all boards so far.

        

        per_row: how many boards per row (default: 10 so all boards are on one row)

        from_: starting board from 0 to 9 (default: 0=initial empty board)

        to: ending board from 0 to 9 (default: last board played)

        

        file: where to print to.  Default is sys.stdout



        >>> env = TicTacToe()

        >>> env.play('x',1,1)

        >>> env.play('o',2,2)

        >>> env.play('x',2,0)

        >>> env.play('o',0,2)

        >>> env.play('x',1,2)

        >>> env.play('o',1,0)

        >>> env.play('x',0,1)

        >>> env.play('o',2,1)

        >>> env.play('x',0,0)

        >>> env.draw_history()

        ...   ...   ...   ...   ..o   ..o   ..o   .xo   .xo   xxo   

        ...   .x.   .x.   .x.   .x.   .xx   oxx   oxx   oxx   oxx   

        ...   ...   ..o   x.o   x.o   x.o   x.o   x.o   xoo   xoo

        """

        if to is None:

            to = len(self._history)

        for i in range(from_, to, per_row):

            for j in range(i, min(i+per_row, to)):

                print(self._history[j][:3] + '   ', file=file, end='')

            print(file=file)

            for j in range(i, min(i+per_row, to)):

                print(self._history[j][3:6] + '   ', file=file, end='')

            print(file=file)

            for j in range(i, min(i+per_row, to)):

                print(self._history[j][6:] + '   ', file=file, end='')

            print(file=file)

            print(file=file)

    

    def self_won(self, converted_state):

        """

        Return True if the state is that of its own player winning.

        

        Gives wrong answers for invalid states.

        """

        if converted_state[:3] == 'SSS' or converted_state[3:6] == 'SSS' or converted_state[6:] == 'SSS':

            return True

        if converted_state[0] == 'S' and converted_state[3] == 'S' and converted_state[6] == 'S':

            return True

        if converted_state[1] == 'S' and converted_state[4] == 'S' and converted_state[7] == 'S':

            return True

        if converted_state[2] == 'S' and converted_state[5] == 'S' and converted_state[8] == 'S':

            return True

        if converted_state[0] == 'S' and converted_state[4] == 'S' and converted_state[8] == 'S':

            return True

        if converted_state[2] == 'S' and converted_state[4] == 'S' and converted_state[6] == 'S':

            return True

        return False

    

    def self_lost(self, converted_state):

        """

        Return True if the state is that of its own player losing.

        

        Gives wrong answers for invalid states.

        """

        if converted_state[:3] == 'OOO' or converted_state[3:6] == 'OOO' or converted_state[6:] == 'OOO':

            return True

        if converted_state[0] == 'O' and converted_state[3] == 'O' and converted_state[6] == 'O':

            return True

        if converted_state[1] == 'O' and converted_state[4] == 'O' and converted_state[7] == 'O':

            return True

        if converted_state[2] == 'O' and converted_state[5] == 'O' and converted_state[8] == 'O':

            return True

        if converted_state[0] == 'O' and converted_state[4] == 'O' and converted_state[8] == 'O':

            return True

        if converted_state[2] == 'O' and converted_state[4] == 'O' and converted_state[6] == 'O':

            return True

        return False

    

    def self_board_full(self, converted_state):

        """

        Return True if the state is that of a full board.

        

        Gives wrong answers for invalid states.

        """

        return '.' not in converted_state



    

    def _is_move_valid(self, which, pos):

        """

        Return True if the proposed move is valid for the specified player.

        

        If it is not the player's turn or the game is over or the position is already

            taken, return False.

        

        which: which player, 0=x, 1=o

        pos: integer position = row*3 + col where move is (row, col), row, col from 0 to 2 inclusive.

        """

        if self._winner is not None:

            return False        

        if self._state[pos] != '.':

            return False

        if which == 0 and not self._x_turn():

            return False

        if which == 1 and not self._o_turn():

            return False

        return True

        

    def _x_turn(self):

        """

        Return True if it is x's turn.

        

        Does not check if game is over.

        """

        return self._state.count('x') == self._state.count('o')

    

    def _o_turn(self):

        """

        Return True if it is o's turn.

        

        Does not check if game is over.

        """

        return self._state.count('x') == self._state.count('o') + 1

        

    def _x_won(self):

        """

        Return True if x has won the game.        

        """

        if self._state[:3] == 'xxx' or self._state[3:6] == 'xxx' or self._state[6:] == 'xxx':

            return True

        if self._state[0] == 'x' and self._state[3] == 'x' and self._state[6] == 'x':

            return True

        if self._state[1] == 'x' and self._state[4] == 'x' and self._state[7] == 'x':

            return True

        if self._state[2] == 'x' and self._state[5] == 'x' and self._state[8] == 'x':

            return True

        if self._state[0] == 'x' and self._state[4] == 'x' and self._state[8] == 'x':

            return True

        if self._state[2] == 'x' and self._state[4] == 'x' and self._state[6] == 'x':

            return True

        return False

    

    def _o_won(self):

        """

        Return True if o has won the game.        

        """

        if self._state[:3] == 'ooo' or self._state[3:6] == 'ooo' or self._state[6:] == 'ooo':

            return True

        if self._state[0] == 'o' and self._state[3] == 'o' and self._state[6] == 'o':

            return True

        if self._state[1] == 'o' and self._state[4] == 'o' and self._state[7] == 'o':

            return True

        if self._state[2] == 'o' and self._state[5] == 'o' and self._state[8] == 'o':

            return True

        if self._state[0] == 'o' and self._state[4] == 'o' and self._state[8] == 'o':

            return True

        if self._state[2] == 'o' and self._state[4] == 'o' and self._state[6] == 'o':

            return True

        return False

        

    def _board_full(self):

        """

        Return true if board is currently full.

        """

        return '.' not in self._state

    

    def _convert_state(self, state, which):

        """

        Convert internal state to state from the point of view of a given player suitable for valuation.

        

        which: 0=x, 1=o

        state: internal state: concatenation of board values, x, o, or .=nobody,

            in order from top left to bottom right, going left to right, then top to bottom.

        

        returns: unique state string of the resulting game position from which's point of view,

            concatenation of board values, S=self, O=other, .=nobody,

            in order from top left to bottom right, going left to right, then top to bottom.

        """

        if which == 0:

            state = state.replace('x','S')

            state = state.replace('o','O')

        else:

            state = state.replace('o','S')

            state = state.replace('x','O')

        return state

    

    def _move_to_pos(self, move):

        """

        Convert a move=(row, col) to a position integer from 0 to 8 inclusive

        

        returns: row*3 + col

        """

        return move[0]*3 + move[1]

class Agent:

    """

    A player (one of two) of the game environment.

    

    >>> agent = Agent(win_value=1, lose_value=0, draw_value=0, unknown_value=0.5)

    

    create a new agent.

    

    win_value: value of a state in which this agent wins

    lose_value: value of a state in which this agent loses

    draw_value: value of a state in which the game is a draw

    unknown_value: initial value for a state that has not yet been valuated

    """

    def __init__(self, win_value=1, lose_value=0, draw_value=0, unknown_value=0.5):

        self._values = dict()

        self._win_value = win_value

        self._lose_value = lose_value

        self._draw_value = draw_value

        self._unknown_value  = unknown_value

        

    def Play(self, env, which=0, epsilon=0.05, accuracy=0.000001, verbose=1):

        moves = env.get_valid_moves(which)

        if np.random.rand() <= epsilon:

            return env.play_random_move(which=which, verbose=verbose)

        best_value = 0

        best_moves = []

        for (w, move, next_state) in moves:

            if np.abs(self.GetValue(env, next_state) - best_value) <= accuracy:

                best_moves.append((w, move, next_state))

            elif self.GetValue(env, next_state) > best_value:

                best_value = self.GetValue(env, next_state)

                best_moves = [(w, move, next_state)]

        return env.play_random_move(which=which, verbose=verbose, moves=best_moves)   

           

    def GetValue(self, env, state):

        if(state in self._values):

            return self._values[state]

        if env.self_won(state):

            self._values[state] = self._win_value

        elif env.self_lost(state):

            self._values[state] = self._lose_value

        elif env.self_board_full(state):

            self._values[state] = self._draw_value

        else:

            self._values[state] = self._unknown_value

        return self._values[state]        

        

    def UpdateValue(self, env, state, next_state, learning_rate=0.1):

        self._values[state] = self.GetValue(env, state)*(1 - learning_rate) + self.GetValue(env, next_state)*learning_rate

    

    def UpdateAllValuesFromHistory(self, env, which, learning_rate=0.1):

        history = env.get_history(which)

        for i in range(len(history)-2,-1,-1):

            s = history[i]

            s1 = history[i+1]

            self.UpdateValue(env, s, s1, learning_rate)        

            
class Game:

    def __init__(self, env_class, agent_class):

        self._agents = [agent_class(), agent_class()]

        self._environment = env_class()

        self._last_iter=0



    def ResetGame(self):

        self._last_iter=0

        

    def PlayGame(self, num=1, update_agents=[0, 1], learning_rate=0.1, verbose=None, progress=1000, epsilon=0.05):

        wins = [0,0]

        if(verbose is None):

            if num == 1:

                verbose = 2

            else:

                verbose = 0.5

        for iter in range(num):

            if progress > 0 and (iter + 1) % progress == 0:

                print(iter+1,"games played", end='\r')                

            self._environment.reset()

            if(verbose >= 1):

                print("Game", self._last_iter + iter + 1)

            while not self._environment.is_game_over():

                which = self._environment.get_turn()

                self._agents[which].Play(env=self._environment, which=which, verbose=verbose-1, epsilon=epsilon)

                if which in update_agents:

                    self._agents[which].UpdateAllValuesFromHistory(self._environment, which, learning_rate=learning_rate)

                if 1 - which in update_agents:

                    self._agents[1 - which].UpdateAllValuesFromHistory(self._environment, 1 - which, learning_rate=learning_rate)

            if verbose>1:

                self._environment.draw_history()

            w = self._environment.get_winner()

            if w in (0, 1):

                wins[w] += 1

        if verbose:

            print("Played games", self._last_iter+1, "through", self._last_iter + num, "    wins:", wins)

            self._environment.draw_history()

        self._last_iter += num

    

    
# create a new game manager using the tic tac toe environment class and agent class

game = Game(TicTacToe, Agent)
# play the game lots of times, updating both agents (0 and 1).  Last game is displayed in full along

# with total wins for agent 0 (X) and agent 1(O) in that order.

game.PlayGame(100000, update_agents=[0,1], learning_rate=0.01, epsilon=0.01) 
#Number of tic tac toe states discovered by agent 0 and agent 1

len(game._agents[0]._values), len(game._agents[1]._values)
## Breakdown of number of states by number of marks on tic-tac-toe board, checking both X and O reckonings

for i in range(10):

    x_count = len([x for x in game._agents[0]._values if 9 - x.count('.') == i])

    o_count = len([x for x in game._agents[1]._values if 9 - x.count('.') == i])

    print("play:", i, "states for x:", x_count, "states for o:", o_count)