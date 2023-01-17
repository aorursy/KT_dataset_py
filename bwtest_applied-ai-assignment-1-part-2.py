import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import os
def backread(im):

    return cv2.imread(im)[:,:,::-1]



edf = pd.read_pickle("../input/aiappliedtest/assignment_1/assignment_0/edf.pkl")

pic = backread("../input/aiappliedtest/assignment_1/assignment_0/connect4.png")

red = backread("../input/aiappliedtest/assignment_1/assignment_0/red.png")

yellow = backread("../input/aiappliedtest/assignment_1/assignment_0/yellow.png")

white = backread("../input/aiappliedtest/assignment_1/assignment_0/white.png")

Data = pd.read_csv("../input/aiappliedtest/assignment_1/assignment_0/Examples.csv")



def synthetic(mat):

    mpic = pic.copy()

    for i in edf.index:

        w = edf.loc[i]

        x1,y1 = w['left']

        f = w['f']

        if mat[w['x'],w['y']] == 0:    

            mpic[y1:y1+f,x1:x1+f] = white

        elif mat[w['x'],w['y']] == 1:

            mpic[y1:y1+f,x1:x1+f] = red

        elif mat[w['x'],w['y']]== 2:

            mpic[y1:y1+f,x1:x1+f] = yellow

        else:

            mpic[y1:y1+f,x1:x1+f] = 0

    return mpic



def print_board(board):

    f,ax = plt.subplots(figsize = (6,6))

    plt.imshow(synthetic(board))

    plt.show()

    

def loadfromarray(ar):

    ar = ar.values

    ar = ar[0][1:43]

    try:

        ar = ar.reshape(6,7).astype('float')

        return ar

    except:

        ar = ar.reshape(6,7)

        return ar
redwins = loadfromarray(Data[Data.file_names == 'redwins'])

yellowwins = loadfromarray(Data[Data.file_names == 'yellowwins'])

rednext = loadfromarray(Data[Data.file_names == 'redisnext'])

yellownext = loadfromarray(Data[Data.file_names == 'yellownext'])

floating_piece = loadfromarray(Data[Data.file_names == 'floating_piece'])

proportion = loadfromarray(Data[Data.file_names == 'invalid_proportion'])

outof = loadfromarray(Data[Data.file_names == 'corrupt_'])

nonnum = loadfromarray(Data[Data.file_names == 'corrupt'])
"""

Create a function isCorrupt that classifies any board that does that has non 

numeric values or values that are not in range as corrupt i.e N not in (0,1,2).

"""

def isCorrupt(board):

    pass
isCorrupt(nonnum)
isCorrupt(outof)
isCorrupt(yellowwins)
def Isfloating(board):

    board = np.flip(board)

    for i in range(len(board)):

        if i < len(board)-1:

            for j in np.where(board[i] == 0)[0]:

                #-------------------------------------------------------------------------------------

                """

                Complete the code below

                The Function should return True if such a condition exists refer to part 1 

                """

                #Your Code Goes Here

                #-------------------------------------------------------------------------------------

    return False
Isfloating(floating_piece)
Isfloating(proportion)
def IsProportional(board):

    red_pieces = len(board[np.where(board == 1)])

    #-------------------------------------------------------------------------------------

    """

    Insert code that counts the number of Yellow pieces

    """

    # Your Code Goes Here

    #yellow_pieces = 

    

    

    #-------------------------------------------------------------------------------------

    

    #Condition: if there the difference between the counts is more than 2, return false

    """

    Write code to check 1st condition

    """

    # Your Code Goes Here

    #if 

        #return False

        

    #-------------------------------------------------------------------------------------

    

    

    

    """

    Remove the comments from the code below once you complete the part above

    """

    #Condition: As red always starts, yellow count can never exceeed red count

    #elif yellow_pieces > red_pieces:

    #    return False
IsProportional(proportion)
def WhosTurn(board):

    """

    Write code to output whos turn is it based on the board

    

    Logic: If there are equal pieces, than its reds turn, if red has more pieces its reds turn

    

    Also if the board is empty its Reds turn

    

    """

    red_pieces = len(board[np.where(board == 1)])

    """

    Insert code that counts the number of Yellow pieces

    """

    # Your Code Goes Here

    #yellow_pieces = 

    

    

    #-------------------------------------------------------------------------------------

    

    """

    Write code that executes the logic described above

    

    """

    

    # Your Code Goes Here

    
WhosTurn(rednext)
WhosTurn(yellownext)
def check_win(board, piece):

    

    board = np.flip(board)

    ROW_COUNT = 6

    COLUMN_COUNT = 7

    

    #-------------------------------------------------------------------------------------

    # Check horizontal locations for win

    for c in range(COLUMN_COUNT-3):

        for r in range(ROW_COUNT):

            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:

                return True

     

    

    

    

    #-------------------------------------------------------------------------------------

    """

    Write the code to see if any game was won, with respect to each column

    """

    # Check vertical locations for win

            

     

    #Your Code Goes Here

    

    

            

    #-------------------------------------------------------------------------------------

    """

    Remove the comments once you have finished the code above. The code below return true if the game is won

    by a player 'piece' = (1,2); wiht 4 in a row connected diaganolly. Use the same method to complete the 

    functions that are above.

    

    # Check positively sloped diaganols

    for c in range(COLUMN_COUNT-3):

        for r in range(ROW_COUNT-3):

            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:

                return True

                

                

    # Check negatively sloped diaganols

    for c in range(COLUMN_COUNT-3):

        for r in range(3, ROW_COUNT):

            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:

                return True

    """

            
check_win(redwins,1)
check_win(yellowwins,2)
test_data = pd.read_csv("../input/aiappliedtest/assignment_1/assignment_0/AssignmentData.csv")

test = pd.read_csv("../input/aiappliedtest/assignment_1/assignment_0/TestFile.csv")

valid = pd.read_csv("../input/aiappliedtest/Validation.csv")
def classifier(B):

    """

    Build a classifier

    """
#Test you code with the validation set

wcount = 0

for i in valid.file_names:

    board = loadfromarray(valid[valid.file_names == i])

    if valid[valid.file_names == i].state.values[0] != classifier(board):

        print('You scored '+valid[valid.file_names == i].file_names[0]+' as '+str(classifier(board))+' the ground truth is '+ str(valid[valid.file_names == i].state.values[0]))     

        wcount += 1

if wcount == 0:

    print('100% correct')
#insert a board you got wrong below to print the board

# example board = loadfromarray(valid[valid.file_names == 'board_7'])

board = loadfromarray(valid[valid.file_names == ''])

print_board(board)
Name = None # Replace None with your name
# Creates the final submission file that you will submit to Kaggle

for i in test_data.file_names:

    board = loadfromarray(test_data[test_data.file_names == i])

    id_ = test[test.file_names == i].index[0]

    test.at[id_,'state'] = int(classifier(board))

test.to_csv("Submission.csv", index = False)