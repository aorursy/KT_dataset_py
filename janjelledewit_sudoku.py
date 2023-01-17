import numpy as np



def sudo_check( sudo ):

    ''''

    The goal of this function is to check wether or not a given sudoku,

    represented by an 9 x 9 NumPy array, satisfies all sudoku constraints.

    Specifically that in each row, column and each of the nine 

    non-overlapping 3 x 3 block, the integers 1 to 9 occur only once.

    '''

    # 

    def check_doubles( array ):

        ''' 

        The goal of this subfunction is to check if the given array holds any 

        duplicates.

        

        Note that digit '0' is not a real digit, but a way to denote a

        blank space i. e. an unknown digit. Duplicates of the digit '0' do 

        not count. 

        '''

        used = []

        for x in array:

            if x == 0:

                continue

            elif x in used:

                return False

            else:

                used += [x]

        return True

    #

    for i in range( 9 ):

        # First we check the rows and columns for duplicate digits.

        row = sudo[i]

        col = sudo[: , i]

        if check_doubles(row) == False or check_doubles(col) == False:

            return False

    #

    for hori in [0, 3, 6]:

        for verti in [0, 3, 6]:

            # Now we look at the 3 x 3 blocks.

            # (hori, verti) is the top left corner of each block.

            block = sudo[ hori : hori + 3 , verti : verti + 3 ]

            block_array = block.reshape(9)

            #Now we can use check_doubles as before

            if check_doubles(block_array) == False:

                return False

    return True
def sudo_solve( sudo, count=0 ):

    '''

    This function will return a solved sudoku if possible, and also say how

    many possible solutions can be found:

        - if count == 0, it is impossible to solve this sudoku.

        - if count == 1, there is a unique solution.

        - if count == 2, there is more than unique solution.

    If count == 0, the original sudoku will be returned. Otherwise a solved 

    sudoku is returned alongside the value of count.

    '''

    # First we need to find the first blank space.

    complete = True

    all_entries = [ (row, col) for row in range(9) for col in range(9) ]

    for entry in all_entries:

        if sudo[ entry ] == 0:

            complete = False

            break

    # If complete is still True, there are no blank spaces.

    if complete == True:

        return sudo, count + 1

    # Else, we can start guessing.

    for guess in range(1, 10):

        sudo[ entry ] = guess

        if sudo_check(sudo) == False:

            # Our guess was clearly false.

            continue

        else:

            # Our guess could lead to a solution.

            sudo, count = sudo_solve(sudo, count=count)

            if count == 2:

                # This sudoku does not have a unique solution.

                return sudo, 2

    # We are out of guesses. This spot is made blank again, and we go  

    # one space back.

    sudo[ entry ] = 0

    return sudo, count
def sudo_gen( sudo=np.zeros( (9,9), dtype=int), first_blank=0 ):

    '''

    The goal of this function is to generate a random solved sudoku from 

    scratch. The keyword arguments are need for recursion only and should not

    be defined when calling this function from outside.

    Note that first_blank is a number between 0 and 80, corresponding to the 

    position of the first blank spot if we start counting first left to right

    and from up to down.

    '''

    row = first_blank // 9

    col = first_blank % 9

    options = np.random.permutation( [1, 2, 3, 4, 5, 6, 7, 8, 9] )

    for guess in options:

        sudo[row, col] = guess

        if sudo_check( sudo ) == False:

            # Our guess was clearly false.

            continue

        elif first_blank == 80:

            # We filled in the last blank, and we are done.

            return sudo, True

        else:

            # Our guess could work. On to the next blank.

            sudo, accept = sudo_gen(sudo=sudo, first_blank=(first_blank + 1) )

            if accept == True:

                # A solution has been found.

                return sudo, True

    # None of the possible guesses work. Make this space blank again and go 

    # one space back.

    sudo[row, col] = 0

    return sudo, False
from time import time

from random import choice



def puzzle_gen( clues ):

    '''

    This function is a puzzle generator, and an overarching function using 

    both sudo_gen() and sudo_solve(). It returns both a solved sudoku, and a 

    puzzle containing the number of clues specified in the input.

    '''

    if clues <= 33:

        print('ERROR: The value of clues should be higher than 33')

    # We generate a solved sudoku

    start = time()

    solved, temp = sudo_gen()

    end = time()

    print("Generate solved sudoku time:", end - start)

    # We make fields blank, and check the solution still is unique.

    start = time()

    count = 1

    puzzle = solved.copy()

    # The variable options lists all positions where there is no blank yet.

    options = [ (row, col) for row in range(9) for col in range(9) ]

    while len(options) > clues:

        entry = choice( options )

        puzzle[ entry ] = 0

        temp, count = sudo_solve( puzzle.copy() )

        if count == 2:

            puzzle[ entry ] = solved[ entry ]

        else:

            options.remove( entry )

    end = time()

    print("Generate puzzle time:", end - start)

    return puzzle, solved



puzzle, solved = puzzle_gen(36)

print(puzzle)

print(solved)