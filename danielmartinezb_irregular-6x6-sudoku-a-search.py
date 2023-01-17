from simpleai.search import SearchProblem, astar

print('imports ready!')
initial_board_str = '''0-0-0-0-1-0

0-3-0-0-2-0

0-2-0-0-0-0

0-0-0-0-5-0

0-0-0-1-0-0

0-6-5-2-0-0'''



initial_group_str = '''1-1-1-1-1-2

4-4-3-1-2-2

4-3-3-3-2-2

4-4-3-3-2-6

4-5-6-6-6-6

5-5-5-5-5-6'''



print('board loaded')
def string_to_list(string_):

    '''Convert puzzle string to list.

       Returns a list'''

    return [row.split('-') for row in string_.split('\n')]



def list_to_string(list_):

    '''Convert puzzle list to string.

       Returns a string'''

    return '\n'.join(['-'.join(row) for row in list_])



print('ok')
def find_actual_position(board):

    '''Find the location of the actual position piece in the puzzle.

       Returns a tuple: row, column'''

    rows = string_to_list(board)

    for ir, row in enumerate(rows):

        for ic, element in enumerate(row):

            if element == 'X':

                return ir, ic

print('ok')
def possible_numbers_in_row(board, row_):

    '''Find all valid numbers inside a row.

       Returns a list of strings''' 

    rows = string_to_list(board)

    elements = []

    for ir, row in enumerate(rows):

        for ic, element in enumerate(row):

            if ir == row_:

                elements.append(element)    

    result = []

    for number in range(1,7):

        if str(number) not in elements:

            result.append(str(number))

    return result

print('ok')
def possible_numbers_in_column(board, column):

    '''Find all valid numbers inside a column.

       Returns a list of strings''' 

    rows = string_to_list(board)

    elements = []

    for ir, row in enumerate(rows):

        for ic, element in enumerate(row):

            if ic == column:

                elements.append(element)

    result = []

    for number in range(1,7):

        if str(number) not in elements:

            result.append(str(number))

    return result



print('ok')
def possible_numbers_in_group(board, groups, group):

    '''Find all valid numbers inside a group.

       Returns a list of strings''' 

    board_list = string_to_list(board) 

    rows = string_to_list(groups)

    elements = []

    for ir, row in enumerate(rows):

        for ic, gr in enumerate(row):

            if gr == str(group):

                elements.append(board_list[ir][ic])

    result = []

    for number in range(1,7):

        if str(number) not in elements:

            result.append(str(number))

    return result



print('ok')
def check_if_complete(board, groups):

    '''Returns true if there is no empty cells in a board.

    '''

    rows = string_to_list(board)    

    for ir, row in enumerate(rows):

        for ic, element in enumerate(row):

            if element == '0' or element == 'X':

                return False

    return True  

print('ok')
def find_next_actual(board, groups):

    '''Find the next cell to expand based on the number of posible numbers in the cell.

        Returns a tuple: row, column'''

    min_posibilities = 16

    next_actual_row = None

    next_actual_col = None



    groups_list = string_to_list(groups) 

    rows = string_to_list(board)

    for ir, row in enumerate(rows):

        for ic, element in enumerate(row):

            if element == '0':

                nums_in_rows = possible_numbers_in_row(board, int(ir))

                nums_in_cols = possible_numbers_in_row(board, int(ic))

                nums_in_gr = possible_numbers_in_group(board, groups, int(groups_list[ir][ic]))

                posibilites = [value for value in nums_in_cols if value in nums_in_rows and value in nums_in_gr]

                if len(posibilites) < min_posibilities:

                    next_actual_row = ir 

                    next_actual_col = ic

                    min_posibilities = len(posibilites)

    return next_actual_row, next_actual_col

print('ok')
class IrregularSudokuProblem(SearchProblem):

    def actions(self, state):

        actual_row, actual_col = find_actual_position(state)

        nums_in_rows = possible_numbers_in_row(state, actual_row)

        nums_in_cols = possible_numbers_in_column(state, actual_col)

        nums_in_gr = possible_numbers_in_group(state, initial_group_str, int(groups_list[actual_row][actual_col]))

        

        posibilites = [value for value in nums_in_cols if value in nums_in_rows and value in nums_in_gr]

        return posibilites



    def result(self, state, action):

        actual_row, actual_col = find_actual_position(state)

        state_list = string_to_list(state)

        state_list[actual_row][actual_col] = action



        if not check_if_complete(list_to_string(state_list), initial_group_str):

            next_actual_row, next_actual_col = find_next_actual(state, initial_group_str)

            state_list[next_actual_row][next_actual_col] = 'X'



        return list_to_string(state_list)



    def is_goal(self, state):

        return check_if_complete(state,initial_group_str)



    def cost(self, state, action, state2):

        return 1

    

    def heuristic(self, state):

        # how many empty cells are we to complete?

        h = 0

        rows = string_to_list(state)

        for ir, row in enumerate(rows):

            for ic, element in enumerate(row):

                if element == '0':

                    h = h + 1

        return h 

print('problem definition done!')
#global variable definition

groups_list = string_to_list(initial_group_str)



#initial state definition

initial_state_list = string_to_list(initial_board_str)

next_actual_row, next_actual_col = find_next_actual(initial_board_str, initial_group_str)

initial_state_list[next_actual_row][next_actual_col] = 'X'



initial_state = list_to_string(initial_state_list) 



#instance of the problem with the initial state

my_problem = IrregularSudokuProblem(initial_state=initial_state)



# A* implementation and results

result = astar(my_problem)

for action, state in result.path():

    print('Insert number', action)

print('-------------')

print('FINAL STATE')

print('-------------')

print(state)