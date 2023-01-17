# Create a 3x3 tic tac toe board of "" strings for each value
board = [' '] * 9
# Create a function to display your board
def display_board(board):
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("---+---+---")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("---+---+---")
    print(f" {board[6]} | {board[7]} | {board[8]} ")
        
        
display_board(board)
#Create a function to check if anyone won, Use marks "X" or "O"
def check_win(player_mark, board):
    win = [f'{player_mark}'] * 3
    return board[:3] == win or board[3:6] == win or board[6:9] == win or \
        [board[0], board[4], board[8]] == win or [board[2], board[4], board[6]] == win or \
        [board[0], board[3], board[6]] == win or [board[1], board[4], board[7]] == win or [board[2], board[5], board[8]] == win

check_win('X', board)
# Create a function to check its a Draw
def check_draw(board):
    return ' ' not in board
# Create a Function that makes a copy of the board
def board_copy(board):
    new_board = []
    for c in board:
        new_board += c
    return new_board

# copy = board_copy(board)
# copy[3] = 'X'
# display_board(copy)
# print(' ')
# display_board(board)
#Immediate move checker
def test_win_move(move, player_mark, board):
    copy = board_copy(board)
    copy[move] = player_mark
    return check_win(player_mark, copy)
#Strategy if others fail
def win_strategy(board):
    if board[4] == ' ':
        return 4
    for i in [0, 2, 6, 8]:
        if board[i] == ' ':
            return i
    for i in [1, 3, 5, 7]:
        if board[i] == ' ':
            return i
# Agents move
def get_agent_move(board):
    for i in range(9):
        if board[i] == ' ' and test_win_move(i, 'X', board):
            return i
    for i in range(9):
        if board[i] == ' ' and test_win_move(i, 'O', board):
            return i
    return win_strategy(board)
# Assemble the game
def tictactoe():
    playing = True
    while playing:
        in_game = True
        board = [' '] * 9
        print('Would you like to go first or second? (1/2)')
        choice = input()
        player_marker = 'O' if choice == '1' else 'X'
        display_board(board)

        while in_game:
            print('\n')
            if player_marker == 'O':
                print('Player move: (0-8)')
                move = int(input())
                if board[move] != ' ':
                    print('Invalid move')
                    continue
            else:
                move = get_agent_move(board)
            board[move] = player_marker
            if check_win(player_marker,board):
                in_game = False
                display_board(board)
                if player_marker == 'O':
                    print('O won')
                else:
                    print('X won')
                break
            if check_draw(board):
                in_game = False
                display_board(board)
                print('The game was a draw.')
                break
            display_board(board)
            if player_marker == 'O':
                player_marker = 'X'
            else:
                player_marker = 'O'
        print('Continue playing? (y/n)')
        ans = input()
        if ans not in 'yY':
            playing = False
    
# Play!!!
tictactoe()
class Tic_Tac_Toe:
    def __init__(self):
        pass