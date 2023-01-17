# Step 1: Write a function that can print out a board. 
# Set up your board as a list, where each index 1-9 corresponds with a number on a number pad, 
# so you get a 3 by 3 board representation.

from IPython.display import clear_output

board = [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ']
def display_board(board):
    '''
    This function draw the game display.
    '''
    print('         #','# ','       #','#\n',
    '        #','#','        #','#\n',
    '   '+board[7],'   #','#   ',board[8]+' ','  #','#   ', board[9]+'\n',
    '        #','#','        #','#\n',
    '        #','#','        #','#\n',
    '_______________________________\n',
    '_______________________________\n',
    '        #','#','        #','#\n',
    '        #','#','        #','#\n',
    '   '+board[4],'   #','#   ',board[5]+' ','  #','#   ', board[6]+'\n',
    '        #','#','        #','#\n',
    '        #','#','        #','#\n',
    '_______________________________\n',
    '_______________________________\n',
    '        #','#','        #','#\n',
    '        #','#','        #','#\n',
    '   '+board[1],'   #','#   ',board[2]+' ','  #','#   ', board[3]+'\n',
    '        #','#','        #','#\n',
    '        #','#','        #','#\n')
    
    
# I imported the "clear_output"class to clean the screen, but this class works only in Jupyter Notebook.
# Let's check how this function works.

display_board(board)
# Step 2: Write a function that can take in a player input and assign their marker as 'X' or 'O'. 
# I will try using while loops to continually ask until you get a correct answer.

def player_input():
    '''
    The first part of this function is to explain how this game works.
    The first while loop is to get the "X" or "O" choice.
    The second while loop is to get the "Yes" or "No" answer.
    '''
    global marker
    print('Welcome to Tic Tac Toe!')
    print('The numbers corresponding to spaces:\n',
           '   **   **            **   **   \n',
           ' 7 ** 8 ** 9          **   **   \n',
           '   **   **            **   **   \n',
           ' ------------      ------------ \n',
           ' ------------      ------------ \n',
           '   **   **            **   **   \n',
           ' 4 ** 5 ** 6          **   **   \n',
           '   **   **            **   **   \n',
           ' ------------      ------------ \n',
           ' ------------      ------------ \n',
           '   **   **            **   **   \n',
           ' 1 ** 2 ** 3          **   **   \n',
           '   **   **            **   **   \n')
    marker = input('Player 1: Do you want to be X or O? ').upper()
    while (marker != 'X') and (marker != 'O'):
        clear_output()
        print('Do you have to choose X or O!')
        marker = input('Player 1: Do you want to be X or O? ').upper()
    choose_first()
    ready = input(print('Are you ready to play? Enter Yes or No. ')).lower()
    while (ready != 'yes') and (ready != 'no'):
        clear_output()
        print('Do you have to choose Yes or No!')
        ready = input(print('Are you ready to play? Enter Yes or No. ')).lower()
    else:
        if ready == 'yes':
            clear_output()
            display_board(board)
            return True
        elif ready == 'no':
            clear_output()
            print('Maybe next time!')
            return False
# Step 3: Write a function that takes in the board list object, a marker ('X' or 'O'), 
# and a desired position (number 1-9) and assigns it to the board.

def place_marker(board, marker, position):
    '''
    marker - "X" or "O" previously selected by the players.
    position - Selected by the player as weel.
    
    This function get the marker and put it to desired position.
    '''
    board[position] = marker
    
# Let's put some random values and see if this function will work.
place_marker(board,'$',8)
display_board(board)

# Good!
# Step 4: Write a function that takes in a board and a mark (X or O) and then checks to see if that mark has won.

def win_check(board, marker):
    '''
    This function check all the victorious combinations and return True or False.
    '''
    if (board[1]==board[2]==board[3]!=' ' or 
        board[4]==board[5]==board[6]!=' ' or board[7]==board[8]==board[9]!=' ' or 
        board[1]==board[4]==board[7]!=' ' or board[2]==board[5]==board[8]!=' ' or 
        board[3]==board[6]==board[9]!=' ' or board[3]==board[5]==board[7]!=' ' or
        board[1]==board[5]==board[9]!=' '):
        return True
        pass
    else:
        return False
# Let's test this function.

# First I will insert some combination.

place_marker(board,'$',7)
place_marker(board,'$',9)

# Now I will check the function to return "True".
win_check(board, '$')

# Now I will check the function to return "False".
place_marker(board,' ',9)
win_check(board, '$')

# Great!
# Step 5: Write a function that uses the random module to randomly decide which player goes first. 
# I may want to lookup random.randint() Return a string of which player went first.

import random

def choose_first():
    players = ['Player 1', 'Player 2']
    first = players[random.randint(0,1)]
    print(f'{first} will go first!')
# Step 6: Write a function that returns a boolean indicating whether a space on the board is freely available.

def space_check(board, position):
        if board[position] == ' ':
            return True
        else:
            return False
        
# Let's check to return "True".
space_check(board, 1)
# Let's check to return "False".
space_check(board, 8)

# Done!
# Step 7: Write a function that checks if the board is full and returns a boolean value. 
# True if full, False otherwise.

def full_board_check(board):
    full = True
    for i in board:
        if i == ' ':
            full = False
            pass
    return full
    
# Let's see weathet it's work.

full_board_check(board)

# Good!!
# Step 8: Write a function that asks for a player's next position (as a number 1-9) 
# and then uses the function from step 6 to check if it's a free position. 
# If it is, then return the position for later use.

def player_choice(board):
    '''
    This function get the number which the player choose, check if the space is empty
    and then, if is available, assign the marker in the space.
    '''
    global marker
    position = int(input('Choose your next position: (1-9) '))
    while not 1<=position<=9:
        clear_output()
        print('Choose a number between 1 and 9!')
        position = int(input('Choose your next position: (1-9) '))
        
    else:
        while space_check(board, position) != True:
            clear_output()
            print('This position is not available!')
            display_board(board)
            position = int(input('Choose your next position: (1-9) '))
        else:
            clear_output()
            place_marker(board, marker, position)
            display_board(board)
            
# I used "global marker" because the changes in this variable within this function will be relevant to another functions.
# Step 9: Write a function that asks the player if they want to play again and returns a boolean.
# True if they do want to play again.

def replay():
    '''
    This function ask if the player will want to "replay" the game: if the answer is "Yes",
    this function return True and the game will replay. If the answer is "No", the function "break"the game.
    '''
    replay = input(print('Do you want to play again? Enter Yes or No. ')).lower()
    while (replay != 'yes') and (replay != 'no'):
        clear_output()
        print('Do you have to choose Yes or No!')
        replay = input(print('Do you want to play again? Enter Yes or No. ')).lower()
    else:
        if replay == 'yes':
            clear_output()
            board = ['#',' ',' ',' ',' ',' ',' ',' ',' ',' ']
            return True
        elif replay == 'no':
            clear_output()
            print('Maybe next time!')
# Now! Let's run the game!!!!

board = ['#',' ',' ',' ',' ',' ',' ',' ',' ',' ']
while True:
    global board
    if not player_input():
        break
    
    while full_board_check(board) == False:
        player_choice(board)
        if win_check(board, marker) == True:
            print('Congratulation! You have won the game!')
            board = ['#',' ',' ',' ',' ',' ',' ',' ',' ',' ']
            break
        else:
            if marker == 'X':
                marker = 'O'
            else:
                marker = 'X'
    else:
        print("This game hadn't a winner")
    if not replay():
        break