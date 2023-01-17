import random
#sort goal according to row

state_library = dict()

class State:

    def __init__(self, n=3, k=6, board=0):

        

        if board == 0:

            self.n = n

            board = [[] for i in range(n)]

            self.ok_col = set(range(n))

            num_bag = set(range(1, k+1))

            for i in range(k):

                selector1 = random.choice(list(num_bag))

                num_bag.remove(selector1)

                selector2 = random.choice(list(self.ok_col))

                board[selector2].append(selector1)

                if len(board[selector2]) >= n:

                    self.ok_col.remove(selector2)

            # empty list evaluate to False

            self.board = tuple(tuple(i) for i in board)

            self.nonempty_col = set(i for i in range(n) if self.board[i])

        else:

            self.board = tuple(tuple(i) for i in board)

            self.n = len(self.board)

            self.ok_col = set(i for i in range(self.n) if len(self.board[i])< self.n)

            self.nonempty_col = set(i for i in range(self.n) if self.board[i])



    def hashing(self):

        board = self.board

        hash_key = board.__hash__()

        self.hash = hash_key

        self.snapshot = board

            



    def find_legal(self):

        legal = []

        for i in self.nonempty_col:

            for j in self.ok_col:

                if i != j:

                    legal.append((i,j))

        return legal

    

    

    

    def find_heuristic(self, goal):

        # define 'goal cell' as a number cell that is specified in one of the goal components

        # define 'goal position' as a position specified in one of the goal components

        # first, filter out the goal components that are fulfilled

        # a goal component is fulfilled if corresponding goal cell is in the goal position, 

        # and there is no unfulfilled goal cell or goal beneath it

        # after that, for each column with at least one unfulfilled goal cell or goal position, 

        # find the lowest goal cell or goal position, count the cells above it, add the count to the heuristic, 

        # to represent effort to remove obstacles and move goal cell to the position

        # the resulting heuristic should be admissable, as it represent moves needed in perfect conditions

        # heuristic is 0 if and only if goal state is reached

        heuristic = 0

        # assume all goals are unfulfilled

        u_goal = set(goal)

        # i for columns and j for rows

        for i in range(self.n):

            for j in range(self.n):

                match_ele = [g for g in u_goal if (g[1] == j and g[2] == i)]

                if match_ele:

                    goal_cell = match_ele[0][0]

                    if len(self.board[i])< j+1:

                        break

                    elif self.board[i][j] == goal_cell:

                        u_goal.remove(match_ele[0])

                    else:

                        break

        goal_cell_bag = set(x[0] for x in u_goal)

        for i in range(self.n):

            goal_pos_i = set(g[1] for g in u_goal if g[2] == i)

            goal_cell_i = set(j for j in range(len(self.board[i])) if self.board[i][j] in goal_cell_bag)

            len_i = len(self.board[i])

            cell_pos_i = goal_pos_i.union(goal_cell_i)

            if cell_pos_i:

                if len(self.board[i]) > min(cell_pos_i):

                    heuristic += len_i - min(cell_pos_i)

            #if goal_cell_i and not goal_pos_i:

            #    heuristic += len_i - min(goal_cell_i)

            #elif not goal_cell_i and goal_pos_i:

            #    heuristic += 0.5*np.abs(len_i - min(goal_pos_i))

            #elif goal_cell_i and goal_pos_i:

            #    if min(goal_cell_i) <= min(goal_pos_i):

            #        heuristic += len_i - min(goal_cell_i)

            #    else:

            #        heuristic += 0.5*(len_i - min(goal_pos_i)) + 0.5*(len_i - min(goal_cell_i))

            

            #if min_goal_cell_i <= min_goal_cell_i:

            #    heuristic += len_i - min_goal_cell_i

            #elif min_goal_cell_i

            

            #cell_pos_i = goal_pos_i.union(goal_cell_i)

            #if cell_pos_i and len(self.board[i]) > min(cell_pos_i):

            #    heuristic += len(self.board[i]) - min(cell_pos_i)

            #print(i)

            #print(goal_pos_i)

            #print(goal_cell_i)

        

        self.heuristic = heuristic

                



        

        

        
# solve the game with A* algorithm

# put initial state to open

# heuristic value need to be known for all states in open

# for states in open set, find the state with lowest heuristic value

# put the child states of the state to open, and put the state to closed

# repeat until a state with heuristic 0 is found





class Solver:

    def __init__(self, init_state, goal = []):

        self.state_library = {}

        self.state = init_state

        self.depth = 0

        self.goal = goal

        self.open = set()

        self.closed = set()

        init_hash = self.snapshot(state=self.state, depth=self.depth, mother=0)

        self.open.add(init_hash)

        self.goal_found = False

        

    # states are hashed for easy reference and avoid repeated states

    def snapshot(self, state, depth, mother):

        state.hashing()

        state.find_heuristic(self.goal)

        if state.hash not in self.state_library:

            self.state_library[state.hash] = {'image': state.snapshot, 'depth': depth, 'heuristic': state.heuristic, 'mother': mother}

        #elif depth + 1 < self.state_library[state.hash].depth:

        #    self.state_library[state.hash.depth] = depth + 1

        return state.hash

    

    def close_current(self):

        children = []

        legal = self.state.find_legal()

        for move in legal:

            new_board = list(list(i) for i in self.state.board)

            #print(new_board)

            #print(self.state.board)

            new_board[move[1]].append(new_board[move[0]].pop())

            new_state = State(board=new_board)

            new_hash = self.snapshot(state=new_state, depth = self.depth + 1, mother = self.state.hash)

            if new_hash not in self.closed:

                self.open.add(new_hash)

            

        self.open.remove(self.state.hash)

        self.closed.add(self.state.hash)

        

    def open_next(self):

        open_states = list(self.open)

        next_state = open_states[0]

        minimum = self.state_library[open_states[0]]['depth'] + self.state_library[open_states[0]]['heuristic']

        m_depth = self.state_library[open_states[0]]['depth']

        for state in open_states:

            if self.state_library[state]['depth'] + self.state_library[state]['heuristic'] < minimum:

                next_state = state

                minimum = self.state_library[state]['depth'] + self.state_library[state]['heuristic']

                m_depth = self.state_library[state]['depth']

            elif self.state_library[state]['depth'] + self.state_library[state]['heuristic'] == minimum:

                if self.state_library[state]['depth'] > m_depth:

                    next_state = state

                    minimum = self.state_library[state]['depth'] + self.state_library[state]['heuristic']

                    m_depth = self.state_library[state]['depth']

                    

        if self.state_library[next_state]['heuristic'] == 0:

            self.goal_found = True

        self.state = State(board=self.state_library[next_state]['image'])

        self.state.hashing()

        self.depth = self.state_library[next_state]['depth']

        

    def print_sequence(self):

        hashkey = self.state.hash

        sequence = []

        # trace nodes back to the orginal node

        while (hashkey != 0):

            sequence.insert(0, hashkey)

            hashkey = self.state_library[hashkey]['mother']

        

        for hashkey in sequence:

            print(self.state_library[hashkey]['image'])

        

        return sequence

            

    
# perform iterations of the A* algorithm, at the end print the best path

def main(sol):

    while not sol.goal_found:

        sol.close_current()

        sol.open_next()

    return sol.print_sequence()
# testing the algorithm with a random n=3, k=6 board, with one atom goal

random.seed(43)

goal = [(1,1,1)]

first_run = Solver(init_state=State(), goal=goal)

main(first_run)
# testing the algorithm with a random n=7, k=35 board, with three conjunctive atom goals

random.seed(43)

goal = [(1,2,3), (2,2,2), (4,0,1)]

second_run = Solver(init_state=State(n=7, k=35), goal=goal)

main(second_run)
# testing the algorithm with a random n=9, k=40 board, with five conjunctive atom goals

random.seed(43)

goal = [(1,2,3), (2,2,2), (4,0,1)]

third_run = Solver(init_state=State(n=9, k=40), goal=goal)

main(third_run)
# User can input the size of the board, number of cells and conjunctive goals

# Limitation: 

# For larger n and large k relative to n^2-n, the heuristic might be less effective, it might take a very long time to reach the goal.

# The algorithm works better for smaller row numbers in goal

print('Please input the size of the board (3~9):')

while True:

    n = input()

    try:

        n = int(n)

    except:

        print('Please input a valid integer.')

    else:

        if n in range(3,10):

            break

        else:

            print('The integer is out of range, please try again.')



print('Please input the number of cells (n~n^2-n)')

while True:

    k = input()

    try:

        k = int(k)

    except:

        print('Please input a valid integer.')

    else:

        if k in range(n,n**2-n+1):

            break

        else:

            print('The integer is out of range, please try again.')

goal = []

goal_count = 1

while True:

    

    print(f'Please input goal {goal_count}, or press enter to start the algorithm')

    print('The goal should be typed as three integers divided by space')



    try:

        input_string = input()

        if not input_string:

            if goal_count == 1:

                print('Please give at least one goal.')

                continue

            else:

                print('Understood, initiate algorithm.')

                break

        a,b,c = input_string.split()

        a = int(a)

        b = int(b)

        c = int(c)

    except:

        print('Input invalid, please try again')

    else:

        if a not in range(1,k+1) or b not in range(0,k) or c not in range(0,k):

            print('Input out of range, please try again')

            continue

        

        goal_cells = set(g[0] for g in goal)

        goal_pos = set((g[1],g[2]) for g in goal)

        goal_row_sum = sum([g[1] + 1 for g in goal])

        if a in goal_cells:

            print('Duplicated goal cell, please try again')

            continue

        if (b,c) in goal_pos:

            print('Position already occupied by another goal, please try again')

            continue

        if b + 1 + goal_row_sum > k:

            print('Row sum exceeds total number of cells, please try again')

            continue

        goal.append((a,b,c))

        goal_count += 1



user_run = Solver(init_state=State(n=n, k=k), goal=goal)

_ = main(user_run)


