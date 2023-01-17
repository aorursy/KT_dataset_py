# Create a 3x3 tic tac toe board of "" strings for each value
board = [''] * 9
# Create a function to display your board
def display_board(board):
        print('     |     |')
        print('  ' + board[0] + '  |  ' + board[1] + '  |  ' + board[2])
        print('     |     |')
        print('-----------------')
        print('     |     |')
        print('  ' + board[3] + '  |  ' + board[4] + '  |  ' + board[5])
        print('     |     |')
        print('-----------------')
        print('     |     |')
        print('  ' + board[6] + '  |  ' + board[7] + '  |  ' + board[8])
        print('     |     |')
        print('                 ')

display_board(board)
#Create a function to check if anyone won, Use marks "X" or "O"
def check_win(player_mark, board):
    ## If the player has won then there must be 3 consecutive Player values
    return ((board[0] == player_mark and board[1] == player_mark and board[2] == player_mark) or  # H top
            (board[3] == player_mark and board[4] == player_mark and board[5] == player_mark) or  # H mid
            (board[6] == player_mark and board[7] == player_mark and board[8] == player_mark) or  # H bot
            (board[0] == player_mark and board[3] == player_mark and board[6] == player_mark) or  # V left
            (board[1] == player_mark and board[4] == player_mark and board[7] == player_mark) or  # V centre
            (board[2] == player_mark and board[5] == player_mark and board[8] == player_mark) or  # V right
            (board[0] == player_mark and board[4] == player_mark and board[8] == player_mark) or  # LR diag
            (board[2] == player_mark and board[4] ==player_mark and board[6] == player_mark))  # RL diag


check_win('X', board)
# Create a function to check its a Draw
def check_draw(board):
        return '' not in board

# Create a Function that makes a copy of the board
def board_copy(board):
    dupeBoard = []
    for j in board:
        dupeBoard.append(j)
    return dupeBoard

#Immediate move checker
def test_win_move(move, player_mark, board):
    bCopy = board_copy(board)
    bCopy[move] = player_mark
    return check_win(bCopy, player_mark)

#Strategy if others fail
def win_strategy(board):
    # Play centre
    if board[4] == ' ':
        return 4
    # Play a corner
    for i in [0, 2, 6, 8]:
        if board[i] == ' ':
            return i
    #Play a side
    for i in [1, 3, 5, 7]:
        if board[i] == ' ':
            return i

# Agents move
def get_agent_move(board):
    # Return agent move with your strategy
    # Check Agent win and Player win
    for i in range(0, 9):
        if board[i] == ' ' and test_win_move(board, 'X', i):
            return i
    # Check player win moves
    for i in range(0, 9):
        if board[i] == ' ' and test_win_move(board, '0', i):
            return i
     
    
    # Final Strategy
    win_strategy(board)

# Assemble the game
def tictactoe():
    ### Note you need to recreate your board again here if you wish to play the game more than once
    Playing = True
    while Playing:
        InGame = True
        board = [' '] * 9
        print('Would you like to go first or second? (1/2)')
        if input() == '1':
            playerMarker = '0'
        else:
            playerMarker = 'X'
        display_board(board)

        while InGame:
            if playerMarker == '0':
                print('Player go: (0-8)')
                move = int(input())
                if board[move] != ' ':
                    print('Invalid move!')
                    continue
            else:
                move = get_agent_move(board)
            board[move] = playerMarker
            if check_win(playerMarker,board):
                InGame = False
                display_board(board)
                if playerMarker == '0':
                    print('Noughts won!')
                else:
                    print('Crosses won!')
                continue
            if check_draw(board):
                InGame = False
                display_board(board)
                print('It was a draw!')
                continue
            display_board(board)
            if playerMarker == '0':
                playerMarker = 'X'
            else:
                playerMarker = '0'

        print('Type y to keep playing')
        inp = input()
        if inp != 'y' and inp != 'Y':
            Playing = False
    
# Play!!!
tictactoe()
class Tic_Tac_Toe:
    def __init__(self):
        pass