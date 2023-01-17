# Create a 3x3 tic tac toe board of "" strings for each value
board = ["*" for i in range(9)]
board

# Create a function to display your board
def display_board(board):
    print(board[0] + " | " + board[1] + " | " + board[2])
    print("___________")
    print(board[3] + " | " + board[4] + " | " + board[5])
    print("___________")
    print(board[6] + " | " + board[7] + " | " + board[8])

display_board(board)
#Create a function to check if anyone won, Use marks "X" or "O"
def check_win(player_mark, board):
    win = False
    ## If the player has won then there must be 3 consecutive Player values
    return(
        (board[0] == board[1] == board[2] == player_mark) or
        (board[0] == board[3] == board[6] == player_mark) or
        (board[1] == board[4] == board[7] == player_mark) or
        (board[2] == board[5] == board[8] == player_mark) or
        (board[3] == board[4] == board[5] == player_mark) or
        (board[6] == board[7] == board[8] == player_mark) or
        (board[0] == board[4] == board[8] == player_mark) or
        (board[2] == board[4] == board[6] == player_mark)
    )


# Create a function to check its a Draw
def check_draw(board):
        return "*" not in board

# Create a Function that makes a copy of the board
def board_copy(board):
    return board[:]
    
#Immediate move checker
def test_win_move(move, player_mark, board):
    pass
#Strategy if others fail
def win_strategy(board):
    pass
# Agents move
def get_agent_move(board):
    # Return agent move with your strategy
    for i in range(9):
        if board[i] == "*" and test_win_move(board, 'X', i):
            return i
    for i in range(9):
        if board[i] == "*" and test_win_move(board, 'O', i):
            return i
    
    return win_strategy(board)
# Assemble the game
def tictactoe():
    ### Note you need to recreate your board again here if you wish to play the game more than once
    pass
    
# Play!!!
tictactoe(board)
class Tic_Tac_Toe:
    def __init__(self):
        pass