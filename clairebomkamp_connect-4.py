print('This is a cell!')
import matplotlib.pyplot as plt

import numpy as np
def drop_piece(board, column):

    row = board.shape[0] - 1

    dropped = False

    while (row >= 0) & (not dropped):

        if board[row, column] == 0:

            board[row, column] = player_number

            dropped = True

        else:

            row = row - 1

    return board, row, dropped



def four_in_a_row(board, row, column):

    

    # Get a list of horizontal, vertical, and diagonal lines involving the piece that was just added

    lines = []



    # Horizontal & vertical

    lines.append(board[:, column])

    lines.append(board[row])

    

    # Diagonal

    w = board.shape[1] - 1

    h = board.shape[0] - 1        

    distance_to_left = min([row, column]) * -1

    distance_to_right = min([h - row, w - column]) + 1    

    lines.append([board[row + n, column + n] for n in range(distance_to_left, distance_to_right)])    

    distance_to_left = min([row, w - column]) * -1

    distance_to_right = min([h - row, column]) + 1

    lines.append([board[row + n, column - n] for n in range(distance_to_left, distance_to_right)])



    # Split each list into chunks of four and check each to see if all values are a) the same and b) not zero

    fours = []

    for line in lines:

        for n in range(len(line) - 3):

            four = line[n:n+4]

            if (four[0] == four).all() & (four[0] != 0):

                return True

    return False



def show_board(board):

    plt.imshow(board, vmin = 0, vmax = 2)

    for i in range(board.shape[0]):

        for j in range(board.shape[1]):

            text = plt.gca().text(j, i, board[i, j],

                           ha="center", va="center", color="w")

    plt.gca().set_yticks([])

    plt.show()
board = np.zeros((6, 7), dtype = int)

playing = True

player_number = 1    

    

while playing:

    

    show_board(board)

    column = int(input('Player ' + str(player_number) + ', pick a column!\n'))

    board, row, dropped = drop_piece(board, column)

    

    if dropped:

        if four_in_a_row(board, row, column):

            playing = False

            show_board(board)

            print('Player ' + str(player_number) + ' wins!')        

        elif not (board == 0).any():

            playing = False

            show_board(board)

            print('Game over!')        

        player_number = 2 if player_number == 1 else 1     

    else:

        print('Sorry, that column is full. Try again!')




