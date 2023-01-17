import numpy as np

from numba import jit
def check_winner(board):

    # Check rows for winner

    for row in range(6):

        for col in range(4):

            if (board[row][col] == board[row][col + 1] == board[row][col + 2] ==\

                board[row][col + 3]) and (board[row][col] != 0):

                return board[row][col]  #Return Number that match row



    # Check columns for winner

    for col in range(7):

        for row in range(3):

            if (board[row][col] == board[row + 1][col] == board[row + 2][col] ==\

                board[row + 3][col]) and (board[row][col] != 0):

                return board[row][col]  #Return Number that match column



    # Check diagonal (top-left to bottom-right) for winner



    for row in range(3):

        for col in range(4):

            if (board[row][col] == board[row + 1][col + 1] == board[row + 2][col + 2] ==\

                board[row + 3][col + 3]) and (board[row][col] != 0):

                return board[row][col] #Return Number that match diagonal





    # Check diagonal (bottom-left to top-right) for winner



    for row in range(5, 2, -1):

        for col in range(4):

            if (board[row][col] == board[row - 1][col + 1] == board[row - 2][col + 2] ==\

                board[row - 3][col + 3]) and (board[row][col] != 0):

                return board[row][col] #Return Number that match diagonal

    c = 0

    for col in range(7):

        if board[0][col]!=0:

            c +=1

    if c == 7:

        # This is a draw

        return 0

    

    # No winner: return None

    return 0
@jit

def check_winner_jit(board):

    # Check rows for winner

    for row in range(6):

        for col in range(4):

            if (board[row][col] == board[row][col + 1] == board[row][col + 2] ==\

                board[row][col + 3]) and (board[row][col] != 0):

                return board[row][col]  #Return Number that match row



    # Check columns for winner

    for col in range(7):

        for row in range(3):

            if (board[row][col] == board[row + 1][col] == board[row + 2][col] ==\

                board[row + 3][col]) and (board[row][col] != 0):

                return board[row][col]  #Return Number that match column



    # Check diagonal (top-left to bottom-right) for winner



    for row in range(3):

        for col in range(4):

            if (board[row][col] == board[row + 1][col + 1] == board[row + 2][col + 2] ==\

                board[row + 3][col + 3]) and (board[row][col] != 0):

                return board[row][col] #Return Number that match diagonal





    # Check diagonal (bottom-left to top-right) for winner



    for row in range(5, 2, -1):

        for col in range(4):

            if (board[row][col] == board[row - 1][col + 1] == board[row - 2][col + 2] ==\

                board[row - 3][col + 3]) and (board[row][col] != 0):

                return board[row][col] #Return Number that match diagonal

    c = 0

    for col in range(7):

        if board[0][col]!=0:

            c +=1

    if c == 7:

        # This is a draw

        return 0

    

    # No winner

    return 0
python_list = [[0 for i in range(7)] for j in range(6)]

np_arr = np.array(python_list)
%%timeit

check_winner(python_list)
%%timeit

check_winner(np_arr)
check_winner_jit(np_arr) #run it once so it compiles first
%%timeit

check_winner_jit(np_arr)