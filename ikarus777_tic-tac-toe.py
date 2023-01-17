import random

class TicTacToe:
    X_MARK = 'X'
    O_MARK = 'O'

    def __init__(self, callback):
        if not callable(callback):
            raise Exception('TicTacToe need a function to retrieve the next move')
        self.callback = callback

    def _resetBoard(self):
        self.board = [
            [None, None, None],
            [None, None, None],
            [None, None, None]
        ]

    def _getMark(self, mark):
        if mark == TicTacToe.X_MARK:
            return TicTacToe.O_MARK
        else:
            return TicTacToe.X_MARK

    def _getEmpty(self):
        empty = []
        for ri, row in enumerate(self.board):
            for ci, cell in enumerate(row):
                if cell is None:
                    empty.append((ri, ci))
        return empty

    def _getRandomMove(self):
        empty = self._getEmpty()
        return random.choice(empty)

    def _playMove(self, move, mark=None):
        if not mark:
            mark = self._getMark(self.mark)
        row, col = move
        if self.board[row][col] != None:
            return -1
        self.board[row][col] = mark
        return 1 if not self._getEmpty() else 0

    def _checkBoard(self):
        b = self.board
        for i in range(3):
            if (b[i][0] and b[i][0] == b[i][1] and b[i][1] == b[i][2]): # row
                return b[i][0]
            if (b[0][i] and b[0][i] == b[1][i] and b[1][i] == b[2][i]): # column
                return b[0][i]
        if (b[0][0] and b[0][0] == b[1][1] and b[1][1] == b[2][2]): # diagonal
            return b[0][0]
        if (b[0][2] and b[0][2] == b[1][1] and b[1][1] == b[2][0]): # diagonal
            return b[0][2]
        return None

    def _printBoard(self):
        p = lambda row, col: self.board[row][col] or ' '
        print( '\n -----')
        print( '|' + p(0,0) + '|' + p(0,1) + '|' + p(0,2) + '|')
        print( ' -----')
        print( '|' + p(1,0) + '|' + p(1,1) + '|' + p(1,2) + '|')
        print( ' -----')
        print( '|' + p(2,0) + '|' + p(2,1) + '|' + p(2,2) + '|')
        print( ' -----\n')       

    def simulateGame(self, mark='X', play_first=False, verbose=False):
        self.mark = mark
        self._resetBoard()
        printBoard = lambda: self._printBoard() if verbose else None
        if not play_first:
            move = self._getRandomMove()
            self._playMove(move)
        empty = self._getEmpty()
        win = None
        while empty and not win:
            printBoard()
            move = self.callback(self.board, empty, mark)
            self._playMove(move, mark)
            win = self._checkBoard()
            if not self._getEmpty() or win:
                break
            printBoard()
            move = self._getRandomMove()
            self._playMove(move)
            empty = self._getEmpty()
            win = self._checkBoard()
        printBoard()

        if win == mark:
            return 1    # win
        elif win == self._getMark(mark):
            return -1   # lose
        else:
            return 0    # draw

    def simulate(self, n_games):
        win = 0
        for _ in range(n_games):
            mark = random.choice([TicTacToe.X_MARK, TicTacToe.O_MARK])
            play_first = random.choice([True, False])
            res = self.simulateGame(mark=mark, play_first=play_first)
            if res == 1:
                win += 1
        return win


def placeMark(board_state, empty_cells, mark):
    return random.choice(empty_cells)

if __name__ == '__main__':
    from datetime import datetime
    random.seed(datetime.now())

    n_games = 5000
    win = TicTacToe(placeMark).simulate(n_games)
    print(f'Player won {win} out of {n_games} games (win rate = {round((win/n_games) * 100, 2)}%)')
import random
from datetime import datetime

random.seed(datetime.now())

def placeMark(board_state, empty_cells, mark):
    """ Return the position to place the mark.
    Ex:
        board_state: [[X, O, X], [X, None, O], [O, None, X]]
        empty_cells: [(1, 1), (2, 1)]
        mark: 'X'
    """
    return random.choice(empty_cells)
win = TicTacToe(placeMark).simulateGame(verbose=True)
if win == 1:
    print('Won')
elif win == -1:
    print('Lost')
else:
    print('Draw') 
n_games = 5000
win = TicTacToe(placeMark).simulate(n_games)
print(f'Player won {win} out of {n_games} games (win rate = {round((win/n_games) * 100, 2)}%)')
import pandas as pd
import copy

def toStr(o):
    """ Makes list/tuple readable and clean
    """
    if isinstance(o, list):
        return str(o).translate(str.maketrans('','', '\'[]'))
    elif isinstance(o, tuple):
        return str(o).strip('()').replace(', ', '-')

def playGame(n_games):
    games = []
    logs = []
    def placeMark(board_state, empty_cells, mark):
        move = random.choice(empty_cells) # randomly choose next move from empty cells
        logs.append((copy.deepcopy(board_state), move)) # deepcopy for list of lists
        return move
    
    tic = TicTacToe(placeMark)
    for _ in range(n_games):
        logs = []
        mark = random.choice([TicTacToe.X_MARK, TicTacToe.O_MARK])
        play_first = random.choice([True, False])
        win = tic.simulateGame(mark=mark, play_first=play_first)
        for i, (board_state, move) in enumerate(reversed(logs)):
            winner = win == 1
            result = 1.0 * winner if i == 0 else .4 + .2 * winner
            games.append({
                'mark': mark,
#                 'play_first': play_first,
                'board_state': toStr(board_state),
                'move': toStr(move),
                'result': result,
            })
    return games

N_GAMES = 100000
games = playGame(N_GAMES)
df = pd.DataFrame(games)
df.head()
from sklearn.preprocessing import LabelEncoder

train = pd.DataFrame() # dataset for train the model
bs_encoder = LabelEncoder()
train['board_state'] = bs_encoder.fit_transform(df['board_state'])
mark_encoder = LabelEncoder()
train['mark'] = mark_encoder.fit_transform(df['mark'])
move_encoder = LabelEncoder()
train['move'] = move_encoder.fit_transform(df['move'])
train['result'] = df['result']
train.head()
y = train['result']
X = train.drop('result', axis=1)
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X, y)
import numpy as np

def getMoveFromPred(preds, empty):
    """ Decode and format the predicted move
    """
    p = max(preds, key=lambda x: x[0]) # get the max value for predicted result
    move_dec = move_encoder.inverse_transform([p[1]])[0] # decode from int to categorical value
    row, col = move_dec.split('-')
    return (int(row), int(col))

def placeMark(board_state, empty_cells, mark):
    """ Predict the result for each possible move
    """
    preds = []
    empty_index = move_encoder.transform([toStr(e) for e in empty_cells]) # transform empty cells to index using encoder
    for i in empty_index:
        p = np.reshape([
            bs_encoder.transform([toStr(board_state)])[0],
            mark_encoder.transform([mark])[0],
            i
        ],  (1, -1))
        preds.append((model.predict(p), i)) # predict result for each possible move and store in a list
    move = getMoveFromPred(preds, empty_cells)
    
    return move
win = TicTacToe(placeMark).simulateGame()
if win == 1:
    print('Won')
elif win == -1:
    print('Lost')
else:
    print('Draw') 
n_games = 500
win = TicTacToe(placeMark).simulate(n_games)
print(f'Player won {win} out of {n_games} games (win rate = {round((win/n_games) * 100, 2)}%)')
class TicTacToeAI:
    X_MARK = 'X'
    O_MARK = 'O'

    def __init__(self, callback_X, callback_O):
        if not callable(callback_X) or not callable(callback_O):
            raise Exception('TicTacToeAI need two functions to retrieve next moves')
        self.callback_X = callback_X
        self.callback_O = callback_O

    def _resetBoard(self):
        self.board = [
            [None, None, None],
            [None, None, None],
            [None, None, None]
        ]

    def _getEmpty(self):
        empty = []
        for ri, row in enumerate(self.board):
            for ci, cell in enumerate(row):
                if cell is None:
                    empty.append((ri, ci))
        return empty

    def _playMove(self, move, mark):
        row, col = move
        if self.board[row][col] != None:
            return -1
        self.board[row][col] = mark
        return 1 if not self._getEmpty() else 0

    def _checkBoard(self):
        b = self.board
        for i in range(3):
            if (b[i][0] and b[i][0] == b[i][1] and b[i][1] == b[i][2]): # row
                return b[i][0]
            if (b[0][i] and b[0][i] == b[1][i] and b[1][i] == b[2][i]): # column
                return b[0][i]
        if (b[0][0] and b[0][0] == b[1][1] and b[1][1] == b[2][2]): # diagonal
            return b[0][0]
        if (b[0][2] and b[0][2] == b[1][1] and b[1][1] == b[2][0]): # diagonal
            return b[0][2]
        return None

    def _printBoard(self):
        p = lambda row, col: self.board[row][col] or ' '
        print( '\n -----')
        print( '|' + p(0,0) + '|' + p(0,1) + '|' + p(0,2) + '|')
        print( ' -----')
        print( '|' + p(1,0) + '|' + p(1,1) + '|' + p(1,2) + '|')
        print( ' -----')
        print( '|' + p(2,0) + '|' + p(2,1) + '|' + p(2,2) + '|')
        print( ' -----\n')

    def _getSeq(self, play_first):
        if play_first == 'X':
            return [('X', self.callback_X), ('O', self.callback_O)]
        else:
            return [('O', self.callback_O), ('X', self.callback_X)]

    def simulateGame(self, play_first='X', verbose=False):
        self._resetBoard()
        sequence = self._getSeq(play_first)
        printBoard = lambda: self._printBoard() if verbose else None
        empty = self._getEmpty()
        win = None
        while empty and not win:
            for mark, callback in sequence:
                printBoard()
                move = callback(self.board, empty, mark)
                self._playMove(move, mark)
                win = self._checkBoard()
                empty = self._getEmpty()
                if not empty or win:
                    break
        return win if win in ['X', 'O'] else 'D'

    def simulate(self, n_games):
        win_X = 0
        win_O = 0
        for _ in range(n_games):
            play_first = random.choice(['X', 'O'])
            res = self.simulateGame(play_first=play_first)
            if res == 'X':
                win_X += 1
            elif res == 'O':
                win_O += 1
        return (win_X, win_O)


def placeMark1(board_state, empty_cells, mark):
    # X
    return random.choice(empty_cells)

def placeMark2(board_state, empty_cells, mark):
    # O
    return random.choice(empty_cells)

if __name__ == '__main__':
    from datetime import datetime
    random.seed(datetime.now())

    n_games = 5000
    win_X, win_O = TicTacToeAI(placeMark1, placeMark2).simulate(n_games)
    print(f'Player X won {win_X} out of {n_games} games (win rate = {round((win_X/n_games) * 100, 2)}%)')
    print(f'Player O won {win_O} out of {n_games} games (win rate = {round((win_O/n_games) * 100, 2)}%)')

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X.values, y.ravel())
def placeMarkRF(board_state, empty_cells, mark):
    """ Predict the result for each possible move
    """
    preds = []
    empty_index = move_encoder.transform([toStr(e) for e in empty_cells]) # transform empty cells to index using encoder
    for i in empty_index:
        p = np.reshape([
            bs_encoder.transform([toStr(board_state)])[0],
            mark_encoder.transform([mark])[0],
            i
        ],  (1, -1))
        preds.append((rf.predict(p), i)) # predict result for each possible move and store in a list
    move = getMoveFromPred(preds, empty_cells)
    
    return move
def placeMark1(board_state, empty_cells, mark):
    return random.choice(empty_cells)

def placeMark2(board_state, empty_cells, mark):
    return empty_cells[0]
n_games = 100
win_X, win_O = TicTacToeAI(placeMark1, placeMark2).simulate(n_games)
print(f'Player X won {win_X} out of {n_games} games (win rate = {round((win_X/n_games) * 100, 2)}%)')
print(f'Player O won {win_O} out of {n_games} games (win rate = {round((win_O/n_games) * 100, 2)}%)')
win_X, win_O = TicTacToeAI(placeMark1, placeMarkRF).simulate(n_games)
print(f'Player X won {win_X} out of {n_games} games (win rate = {round((win_X/n_games) * 100, 2)}%)')
print(f'Player O won {win_O} out of {n_games} games (win rate = {round((win_O/n_games) * 100, 2)}%)')
win_X, win_O = TicTacToeAI(placeMark, placeMarkRF).simulate(n_games)
print(f'Player X won {win_X} out of {n_games} games (win rate = {round((win_X/n_games) * 100, 2)}%)')
print(f'Player O won {win_O} out of {n_games} games (win rate = {round((win_O/n_games) * 100, 2)}%)')