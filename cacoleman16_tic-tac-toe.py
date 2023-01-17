# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# ------ Global Variables --------





# game board



board = ["-", "-", "-",

         "-", "-", "-",

         "-", "-", "-"]





# Is Game on?



game_still_going = True



# Who is the winner?



winner = None



# Whose turn is it?



current_player = "X"



# ------------------ Functions ----------------------





def display_board():

    print("               ")

    print(board[0]+"|" + board[1]+"|" + board[2])

    print(board[3]+"|" + board[4]+"|" + board[5])

    print(board[6]+"|" + board[7]+"|" + board[8])

    print("               ")



# Play a game of Tic Tac Toe





def play_game():

    # Display intial board

    display_board()



    while game_still_going == True:



        handle_turn(current_player)



        check_if_game_over()



        flip_player()

    if winner == "X" or winner == "0":

        print(winner + " won.")

    elif winner == None:

        print("Tie.")





# check if the game has ended

def check_if_game_over():

    check_if_win()

    check_if_tie()



    return





def check_if_win():

    global winner



    # check rows

    row_winner = check_rows()

    # check columns

    column_winner = check_columns()

    # check diagnols

    diagnol_winner = check_diagnols()



    if row_winner:

        winner = row_winner

    elif column_winner:

        winner = column_winner

    elif diagnol_winner:

        winner = diagnol_winner

    else:

        winner = None



    return





def check_rows():



    global game_still_going



    row_1 = board[0] == board[1] == board[2] != '-'

    row_2 = board[3] == board[4] == board[5] != '-'

    row_3 = board[6] == board[7] == board[8] != '-'

   # Return if Winner X or O

    if row_1 or row_2 or row_3:

        game_still_going = False



    if row_1:

        return board[0]

    elif row_2:

        return board[3]

    elif row_3:

        return board[6]



    return





def check_columns():



    global game_still_going

    # check columns

    column_1 = board[0] == board[3] == board[6] != '-'

    column_2 = board[1] == board[4] == board[7] != '-'

    column_3 = board[2] == board[5] == board[8] != '-'



    # Return if Winner X or O

    if column_1 or column_2 or column_3:

        game_still_going = False



    if column_1:

        return board[0]

    elif column_2:

        return board[1]

    elif column_3:

        return board[2]



    return





def check_diagnols():

    global game_still_going



    # check diagonols



    diagonal_1 = board[0] == board[4] == board[8] != '-'

    diagonal_2 = board[2] == board[4] == board[6] != '-'



    # Return the winner, X or O

    if diagonal_1 or diagonal_2:

        game_still_going = False



    if diagonal_1:

        return board[0]

    elif diagonal_2:

        return board[2]



    return





def check_if_tie():

    global game_still_going

    if "-" not in board:

        game_still_going = False

    return



# becomes the other players turn





def flip_player():

    global current_player

    if current_player == 'X':

        current_player = 'O'

    elif current_player == 'O':

        current_player = 'X'

    return





def handle_turn(player):



    print(player + "'s turn.")

    position = input("Choose a position from 1-9:")



    valid = False

    while not valid:

        while position not in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:

            position = input(" Invalid input. Choose a position from 1-9: ")



        position = int(position) - 1



        if board[position] == "-":

            valid = True



        else:

            print("You can not go there! Go again")



    board[position] = player

    display_board()



    return







#

#

#

#

#

# board

# display board

# play game

# handle turn

# check win

# check rows

# check columns

# check diagnols

# check tie

# flip player