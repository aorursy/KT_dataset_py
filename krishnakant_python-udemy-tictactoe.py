'''

IMPORTS

'''

from IPython.display import clear_output
'''

CONSTANTS

'''

WIN_MATRIX = [[1,2,3],

              [4,5,6],

              [7,8,9],

              [1,4,7],

              [2,5,8],

              [3,6,9],

              [1,5,9],

              [3,5,7]]
'''

FUNCTIONS

'''

# A function to display the board

def display_board(ip_pos):

    '''

    INPUT: 

    An array of 10 values (not 9!); 

    The first value is a dummy (index 0); 

    Indices 1 to 9 hold the symbols given as input by the players from the num-pad they are using; 

    If no number exists in the list, the default value will be a space

    

    RESULT: 

    Print a new Tic Tac Toe board with the updated symbols (X,O) input by the player; 

    Does not RETURN anything

    '''

    #print('\n'*100)   # A simple way to clear the board if we are running the program as a script

    clear_output()

    print('\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t')

    print('\t' + ip_pos[7] + '\t' + '|' + '\t' + ip_pos[8] + '\t' + '|' + '\t' + ip_pos[9] + '\t')

    print('\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t')

    print('-'*17 + '-'*15 + '-'*17)

    print('\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t')

    print('\t' + ip_pos[4] + '\t' + '|' + '\t' + ip_pos[5] + '\t' + '|' + '\t' + ip_pos[6] + '\t')

    print('\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t')

    print('-'*17 + '-'*15 + '-'*17)

    print('\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t')

    print('\t' + ip_pos[1] + '\t' + '|' + '\t' + ip_pos[2] + '\t' + '|' + '\t' + ip_pos[3] + '\t')

    print('\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t' + '|' + '\t' + ' ' + '\t')

# A function to retrieve player inputs at the start of the game (X or O selection)

def select_X_or_O():

    '''

    This function will be run at the start of the program;

    '''

    print('Welcome to the Tic Tac Toe Game! \nPresented by... eeeeh... someone...')

    player1 = ''

    player2 = ''

    

    while player1 != 'X' and player1 != 'O':

        print('Player 1: Do you want to be X or O? ')

        player1 = input()

        player1 = player1.upper()

        if player1 != 'X' and player1 != 'O':

            print('Invalid Input given...\nPlease type a valid input...')

        else:

            break

    

    if player1 == 'X':

        player2 = 'O'

    else:

        player2 = 'X'

    

    print('Player 1: You have chosen to be {}\nPlayer 2: You can play as {}'.format(player1,player2))

    

    return player1, player2
# Take inputs of location to put X/O, and check if the location is free or has already been taken

def play_round(done_list, player_ip_dict):

    '''

    INPUT: 

    done_list is the list of all indices that have been taken so far in the game; Selecting any number matching a number in done_list is invalid;

    player_dict is a dictionary which has the player number as the key and X/O as the value (only 1 player)

    '''

    loc_state = True

    loc_choice = 0

    player_num = player_ip_dict['Num']

    player_sym = player_ip_dict['Symbol']

    

    while loc_state:

        # Request Input

        print('{}: Please enter the location that you want to place {} at (1 to 9)...'.format(player_num, player_sym))

        loc_choice = input()

        

        # Digit check and conversion

        if loc_choice.isdigit():

            loc_choice = int(loc_choice)

        else:

            print('You have entered a string...\nPlease enter a number when prompted...')

            continue

        

        # Digit between 1 and 9

        if loc_choice in range(1,10):

            pass

        else:

            print('You have entered a number that is not among 1-9...\nPlease enter a number when prompted...')

            continue

        

        # Check if the location has already been taken

        if loc_choice not in done_list:

            print('{} has been selected'.format(loc_choice))

            loc_state = False

        else:

            print('That location has already been taken...\n')

            loc_state = True

            continue

    

    return loc_choice, player_sym
# Logic

def win_logic(play_list):

    win_state = False

    for i in range(len(WIN_MATRIX)):

        if set(WIN_MATRIX[i]) <= set(play_list):

            win_state = True

            break

        else:

            continue

    return win_state



def display_winner(done_list):

    '''

    

    '''

    pl1_list = done_list[::2]

    pl2_list = done_list[1::2]

    pl1_state = win_logic(pl1_list)

    pl2_state = win_logic(pl2_list)

    # Both can only be False at the same time

    if pl1_state == pl2_state:

        return 'None', pl1_state

    elif pl1_state == True:

        return 'Player 1', pl1_state

    else:

        return 'Player 2', pl2_state
'''

MAIN CODE BEGINS HERE

'''

replay = True

while replay:

    player_dict = {}

    ip_list = ['#',' ',' ',' ',' ',' ',' ',' ',' ',' ']

    done_list = []

    player_dict['Player 1'], player_dict['Player 2'] = select_X_or_O()

    itr = 0

    ip_dict = {}

    pl_winner = None

    while (itr < 9):

        # Initial Display

        if itr == 0:

            display_board(ip_list)

        itr += 1

        # Request Move

        if itr % 2 == 0:

            ip_dict['Num'] = list(player_dict.keys())[1]

            ip_dict['Symbol'] = list(player_dict.values())[1]

        else:

            ip_dict['Num'] = list(player_dict.keys())[0]

            ip_dict['Symbol'] = list(player_dict.values())[0]

        loc_ch, loc_sym = play_round(done_list, ip_dict)

        done_list.append(loc_ch)

        # Display Move

        ip_list[loc_ch] = loc_sym

        display_board(ip_list)    

        # Perform Logic to decide if we have A Winner!

        winner, win = display_winner(done_list)

        if win:

            pl_winner = winner

            break



    if not pl_winner:

        print('It was a Draw...')

    else:

        print('WE HAVE A WINNER!!!\n{} Wins...'.format(pl_winner))



    print('Do you wish to play again? (Press Q to exit, or anything else to continue)')

    replay_ch = input()

    if replay_ch.upper() != 'Q':

        replay = True

    else:

        replay = False



print('The Game has Ended...\nHope to see you soon...:D')