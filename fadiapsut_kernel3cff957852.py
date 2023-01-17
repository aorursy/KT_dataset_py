# Implementation of a depth-first search for a solution to the she goat-wolf-cabbage-farmer puzzle.

entity = ['man', 'goat', 'wolf', 'cabbage']

path = []

steps = 0





# Defines who can eat whom

def eats(x, y):

    if x == 'goat' and y == 'cabbage':

        return True

    elif x == 'goat' and y == 'wolf':

        return True

    else:

        return False





# Defines if a pair of entities is safe to be left alone on one side of the river.

def safe_pair(x, y):

    if eats(x, y) or eats(y, x):

        return False

    else:

        return True





# Returns the state of the symbol who is in the dictionary.

def state_of(who, state):

    try:

        return state[who]

    except KeyError:

        state[who] = False

        return False





# Verifies if the state defined as an dictionary is safe.

def safe_state(state):

    if state_of('man', state) == state_of('goat', state):

        return True

    elif state_of('goat', state) == state_of('wolf', state):

        return False

    elif state_of('goat', state) == state_of('cabbage', state):

        return False

    else:

        return True





# Moves the entity from one side to the other in the sate.

def move(who, state):

    if state[who] == 'leftt':

        state[who] = 'right'

    else:

        state[who] = 'leftt'

    return state





# Tests if the state has reached the goal. This is the case if all are on the other side.

def goal_reach(state):

    if not state:

        return False

    return (state_of('man', state) == 'right' and

            state_of('goat', state) == 'right' and

            state_of('wolf', state) == 'right' and

            state_of('cabbage', state) == 'right')





# Checks if child is a safe state to move into, and if it is, it adds it to the list of states.

def check_add_child(child, list_states):

    if safe_state(child):

        list_states.append(child)

    return list_states





# The state of everything based on the farmer location.

def expand_states(state):

    children = []

    child = state.copy()

    move('man', child)

    check_add_child(child, children)

    for ent in entity:

        if state_of(ent, state) == state_of('man', state):

            child = state.copy()

            move('man', child)

            move(ent, child)

            check_add_child(child, children)

    return children





# Searches for a solution from the initial state.

def search_sol(state):

    path.append(state)

    next = state.copy()

    while next and not goal_reach(next):

        global steps

        steps += 1;

        nl = expand_states(next)

        next = {}

        for child in nl:

            if not (child in path):

                next = child

                path.append(next)

                break

    return next





# Initialization of the global variables.

initial_state = {}

initial_state['man'] = 'leftt'

for e in entity:

    initial_state[e] = 'leftt'



# Construct the full solution now.

print("Current state:")

print("{'man': 'leftt', 'goat': 'leftt', 'wolf': 'leftt', 'cabbage': 'leftt'}")



# Construct the full solution after evaluating the previous statements.

print("\nSearching for a solution:")

print(search_sol(initial_state))



# Evaluate the variable path to see the solution backwards.

print("\nThe full path is:")

for s in path:

    print(s)

# The optimal solution "no. of steps"

print("\nThe number of steps the farmer did to reach the other side is:", steps)