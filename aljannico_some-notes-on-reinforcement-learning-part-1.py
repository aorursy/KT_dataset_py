import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import os

# print(os.listdir("../input"))



# The following code tries to be as close as possible to

# the pseudo code in paper 1.

# Exception: we count the leafes of the game tree that

# are explored 

leaf_count = 0

# (and we generate some debug output, if requested)



mm = {}

def minimax(s,d,p):

    if s in mm: return mm[s]

    if debug: print(' '*(depth-d),"MM: ",s,p,W(s))

    if d <= 0 or W(s) != "ongoing":

        global leaf_count

        leaf_count += 1

        b = E(s)

    elif p == "max":

        b = float("-inf")

        # Hopelessly inefficient way to evaluate all "next" feasible actions

        for a in A:

            if V(s,a,"max"):

                b = max(b,minimax(T(s,a),d-1,"min"))

    else:

        b = float("inf")

        # Hopelessly inefficient way to evaluate all "next" feasible actions

        for a in A:

            if V(s,a,"min"):

                b = min(b,minimax(T(s,a),d-1,"max"))

    mm[s] = b # store result for this state

    return b

    

### Jetzt brauchen wir die Funktionen aus der Formalisierung

def E(s):

    '''

    Evaluation function

    @s: state to eval

    We return one of the values 100, -100, 0

    '''

    if W(s) == 'min':

        return -100 # min wins!

    if W(s) == 'max':

        return 100 # max wins!

    return 0 # Otherwise, answer as if this would be a draw

    # da wir Tic-Tac-Toe vollständig lösen, benötigen wir

    # keine Bewertung für Nicht-Ende-Zustände



# We assume a 3x3 board represented as a one-dimensional array:

#  0 1 2

#  3 4 5

#  6 7 8

# Initially, the board is empty (represented with zeroes)

empty_board = ( 0,0,0,\

                0,0,0, \

                0,0,0 )

# We represent pieces of max as 1 and pieces of min as -1

# We do not try to identify draws early

# A winning state for max is a board with 3 ones horizontally, vertically or diagonally

# A winning state for max is a board with 3 minus-ones horizontally, vertically or diagonally

# A draw is a no-win board with no more empty space

# Note: it is logically impossible to have a situation, where both players win (this

# cannot be produced from legal moves)

W_store = {} # this is a version of Winner checking with memoization (could be

             # obfuscated with a dedorator)

def W(s):    

    '''

    Winner determination

    We return the winning player, "draw" or "ongoing"

    "draw" is defined as: there is no more space left, it

    it not clever in the sense that it would see early

    that winning is already impossible for either player

    '''

    global W_store # keep the result of your analysis, trades space for time

    if s in W_store: return W_store[s]

    p = 0

    # Horizontal wins

    if s[0] == s[1] == s[2] and s[0]: p=s[0]

    if s[3] == s[4] == s[5] and s[3]: p=s[3]

    if s[6] == s[7] == s[8] and s[6]: p=s[6]

    # Vertical wins     

    if s[0] == s[3] == s[6] and s[0]: p=s[0]

    if s[1] == s[4] == s[7] and s[1]: p=s[1]

    if s[2] == s[5] == s[8] and s[2]: p=s[2]

    # Diagonal wins    

    if s[0] == s[4] == s[8] and s[0]: p=s[0]

    if s[2] == s[4] == s[6] and s[2]: p=s[2]

    if p == 1: 

        result = "max"

    elif p == -1: 

        result = "min"

    elif sum(map(abs,s)) == 9:  

        result = "draw"  # no more free space

    else:

        result = "ongoing"

    W_store[s] = result # memoize the result

    return result
# Let's do some testing, before we continue:

b1 = ( 0,0,0,   0,0,0,   0,0,0)

b2 = ( 1,1,1,  -1,-1,0,  1,-1,0)

b3 = ( 1,1,-1,  1,-1,0, -1,-1,1)

b4 = ( 1,1,-1,  -1,-1,1,  1,1,-1)



# Minimalistic tests

assert W(b1)=='ongoing'

assert W(b2)=="max"

assert W(b3)=="min"

assert W(b4)=="draw"

# .. or this way

assert W((0, 1, 0, 0, 0, 0, -1, 0, 0)) == "ongoing"
## We need more!

## Let's codify actions as in the formalization, there is no

## differentiation among players, and as we want to follow

## the formalization as close as possible, the action list is

## rather stupid...placing a piece (represented

## as either 1 or -1) on an square of the board

## (represented as an index taken from {0,...,8})

## Action = {0,...,8} x {-1,1}

A = [(i,p) for i in set([0,1,2,3,4,5,6,7,8]) for p in set([1,-1])]



print("Set of possible actions: ",A)



## Now, we have to validate potential actions...

def V(s,a,p):

    '''

    We return either true of false.

    We return false, if we try to set a foreign piece (player 1

    sets a -1 piece, or vice versa) or if we try to set a piece

    to an index which is occupied already. Otherwise, we return

    true.

    Remark: The validity check is in a sense incomplete, as it does 

    not check if the number of pieces of each player is balanced, 

    that is: assuming max always starts, the number of pieces of max 

    on the board prior to the action max takes must equal the number 

    of min pieces, and the number of min pieces before min takes an

    action must be (number of max pieces - 1). This can be ignored 

    as the mechanics of the implemented game take care of this 

    constraints implicitly.

    '''

    idx,piece = a

    # Do not touch the pieces of the other player...

    if piece == -1 and p == 'max': return False

    if piece == 1  and p == 'min': return False

    # Is there space on the board?

    if s[idx]: return False # No...

    return True # ok, no more constraints to be checked



print("Setting a piece of player max by player min to position 2 on Board",b1," is valid?",V(b1,(2,1),'min')) 

print("Setting a piece of player max by player max to position 2 on Board",b1," is valid?",V(b1,(2,1),'max'))

print("Setting a piece of player max by player max to position 2 on Board",b2," is valid?",V(b2,(2,1),'max'))

print("Setting a piece of player min by player min to position 2 on Board",b2," is valid?",V(b2,(5,-1),'min'))

# Note: the following is an example for the incompleteness of the validity check

# see the remark above.

print("Setting a piece of player max by player max to position 2 on Board",b2," is valid?",V(b2,(5,1),'max'))



# we return a COPY of the input state!

def T(s,a):

    '''

    Transition function: applying an action to a state to

    returns the new state (without validity check)

    '''

    new_s = list(s)

    idx,piece = a

    new_s[idx] = piece

    return tuple(new_s) # we return an immutable tuple - which is hashable

    

print("\nThe new board state after setting a piece of player max to position 2 on board b1: ",T(b1,(2,1))) 
# Now, we can play!

# If you set debug to True, you will see the complete game tree...this will

# slow down your browser significantly (as the result pane will become

# pretty filled-up... ;)

debug = True # USE False fro higher d



# depth gives a depth limit

# depth == 100 is (by far) large enough to be irrelevant, of course,

# more than 9 actions are not possible in Tic-Tac-Toe, so 9 will be the max depth

# achievable (yes, right, d limits the depth)

# depth = 100 # produces complete game tree

depth = 3 # for demonstration purposes



leaf_count = 0 # see above, we count the visited leaves



print(minimax(b1,d=depth,p="max"))  # for demonstration purposes, not the complete game tree

# TO TRAVEL THE COMPLETE GAME TREE, set d higher, see below

# print(minimax(b1,d=100,p="max"))  

print("\nVisited Leaves: ",leaf_count)
# Solving the game!

debug = False

depth = 100

leaf_count = 0 

mm = {} # Clear memory

print("Resultat optimalen Tic-Tac-Toe-Spielens: ",minimax(b1,d=depth,p="max"))

print("\nVisited Leaves: ",leaf_count)
# Zustandsdefinitionen s. oben

V_table = {}  # space to store probabilities



def Val(s,p_id):

    '''

    @s: state

    @p: player who asks

    '''

    # Setting pl (player) and op (opponent) id strings

    if p_id == 'max': 

        pl,op = 'max','min'

    else:

        pl,op = 'min','max'

    if debug: print("Looking for ",s," in Val:",s in V_table)

    v = 0.5 # Default value in ongoing games which have not already be values

    if s not in V_table:

        w = W(s)

        if debug: print("Result: ",w,"(",pl,",",op,")")

        if w == pl: 

            v = 1.0

        elif w == op: # or w == 'draw':

            v = 0.0            

        elif w == 'draw': 

            v = 0.5

        # store only if value is not 0.5!

        # if not v == 0.5: 

        #    if debug: print("Storing ",v," for ",s)

        V_table[s] = v # faster with storage? (could be removed!)

    else:

        v = V_table[s]

    return v



import random



class Player():

    

    def __init__(self,id,epsilon=0.1):

        self.id = id

        self.epsilon = epsilon

        self.history = [] # Moves played

        

    def move(self,s):

        # Decide how to move in state s

        pmoves = []

        for a in A:

            if V(s,a,self.id):

                pmoves += [(a,Val(T(s,a),self.id))]

        if debug: print(pmoves)

        # Now, decide for a move

        # Either, take on of the best possible moves (exploitation)

        # or "any" other move

        move_type = 1 # 1 for greedy (exploit), 0 for non greedy (explore)

        if random.random() < self.epsilon:

            # Explore

            # Two options: 

            # (A) we remove all moves with maximal value

            # and the choose one of the remaining moves

            # As it may happen that we remove all possible moves,

            # (because they all have the same evaluation),

            # we keep the first move in the list

            # save_move = pmoves[0] # store a "default move"

            # ...

            # (B) we simply chose "any" move

            # We go for B initially this is also

            # done in the implementation referenced above

            # This is justifiable, as we can't

            # differentiate between 

            # "good because (often) explored" / "Bad because (often) explorer"

            # "still good because not yet explored (enough)" / 

            # "still bad because not yet explored (enough)"

            # to differentiate here is a "no brainer" (compare MCTS)

            # see later

            action,_ = random.choice(pmoves)

            move_type = 0

            if debug: print("Explore action ",action)

        else:

            # Exploit 

            # (we should shuffle among all "best" moves!)

            _,max_value = max(pmoves,key=lambda x:x[1])

            max_actions = [(act,val) for (act,val) in pmoves if val == max_value]

            action,_ = random.choice(max_actions) # Actions with maximal value

            if debug: print("Maximal value choices:", max_actions)

            if debug: print("Exploit action ",action)

        # Store the move and it's type

        self.history.append((s,action,move_type))

        return action

    

    def clear(self):

       self.history = []

            

        

debug = True



p1 = Player('max')

p2 = Player('min')

def game(p1,p2):

    s = empty_board # start fresh with an empty board

    players = [p1,p2]

    p1.clear()

    p2.clear()

    turn = 0

    

    while turn < 10: 

        if debug: print(s)

        player = players[turn % 2]

        if debug: print("Turn of ",player.id)

        a = player.move(s)

        if debug: print("Action of the player: ",a)

        s = T(s,a)

        # Are we done?

        w = W(s)

        if debug: print("Result: ",w)

        if w != 'ongoing':

            if debug: print("Game ended with ",w," State: ",s)

            return w            

        turn += 1

        

result = game(p1,p2)
# Some magic to extend classes on the fly

# adapted from: https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6

# extended to being able to use self or cls, could be done nicer, I am sure

from functools import wraps 

def add_method(cls,type='o'):

    def decorator(func):

        @wraps(func) 

        def wrapper(self, *args, **kwargs): 

            if type == 'o':

                return func(self, *args, **kwargs)

            if type == 'c':

                return func(cls, *args, **kwargs)

            if type == 'n':

                return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)

        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func

        return func # returning func means func can still be used normally

    return decorator



# Convinience names

def add_inst_method(cls):

    return add_method(cls,'o')

def add_cls_method(cls):

    return add_method(cls,'c')

def add_static_method(cls):

    return add_method(cls,'n')

# Now, we can use a decorator add_method to add instance methods to a class

# "on the fly"



class AAA:

    test1 = "huhu"

    pass



@add_static_method(AAA) 

def test(test2="hihi"):

    print(AAA.test1+test2)

    

@add_inst_method(AAA) # or add_method(A,'o') (o for object)

def set_v(self,value):

    self.v = value

# setattr(A,'set_v',set_v)



@add_inst_method(AAA)

def get_v(self):

    return self.v

# setattr(A,'get_v',get_v)



a = AAA()

a.test()

a.set_v("toll")

a.get_v()
@add_method(Player)

def backup(self,result,step_size=0.05):

    ''' 

    Learn from the history!

    There are two possible outcomes (from the perspective

    of the current player):

    (A) Either, we have won (then the resulting state from the

    last action is a win) or not (we lost or it was a draw)

    This is the case when our last move resulted immediately

    in a draw or the other player had one more finishing move.

    Either way: if the next state following from our last action

    is not a win, the winning probability should be zero...

    well, not really true, as we collapse a set of potential 

    follow-up states into one result - which is cool for 

    MiniMax, but not for RL...hm, on the other hand, as we do 

    not SET the value of the prior state to the value WIN, following

    the  above argument should result in a "sensibly" average value,

    as non-wins and wins will have an effect in the value in

    proportion to their occurences. Ok. Let's to it then.

    

    Ok, actually there are two cases here:

    we are the player that ended the game (in win or draw), then

    the follow-up state in our last move is either a win or a draw

    or we are NOT the ending player, in which case the follow-up

    state in our last move was "ongoing" which implies that we

    did not win (as we value loss and draw equally "bad", leabing this

    undecided does no harm - we could as well integrate the final

    payoff into the history mechanism, should we wish to differentiate

    between the two cases).

    

    (B) We keep track of our local (decision) history only.

    Assume we are X. We are in a state s_x (where X has to move).

    We want to choose the best possible follow-up state (exploitation).

    We decide by looking at the values of the follow-up states (which

    are all states where O has to move). We choose and move, and then

    O decides upon the next state, then we have to choose again.

    In back-up, we could basically do two things:

    (1) we keep track of a global state history and propagate value

    over all states. Then, each agent has to have his own Val-Table (with

    redundant information)

    (2) we propage back up with only one table by leaving out the

    states where we have to move (we require values for the RESULTS

    of our moves, that is the follow-up states). Then we can use ONE

    Val table for both players (as we learn against ourselves this is

    even more plausible since we will learn ONE (universal) policy for 

    X and O in parallel)

    We realize (2) below.

    X starts, states: []/X - [X]/O - [XO]/X - [XOX]/O ...

    X has to have values for all states where it can moves to (i.e. all 

    states marked by /O, where O should move.

    

    Practically, we have to propagate the back-up as follows:

    We keep track of the states and actions, where and with which

    we move, for example [XO]/X above, and implicitly [XOX]/O, as T(s,a).

    Now, we want to value [XOX]/O, and the predecessor of [XO]/X, but

    not [XO]/X. 

    '''

    def update(next_val,s,a):

        td_error = next_val - Val(T(s,a),self.id)

        new_val = Val(T(s,a),self.id) + step_size * td_error

        if debug: print("Update ",T(s,a)," from ",Val(T(s,a),self.id)," to ",new_val)

        V_table[T(s,a)] = new_val



    

    if debug: print("History ",self.id,":",self.history,len(self.history))

    i = len(self.history)-1

    while i >= 1:

        incomplete = False # did our last move finish the game or is the history incomplete?

        if i == len(self.history)-1:

            if self.id == result: # ... und gerade gewinnen

                next_val = 1.0 # not used

            elif result == 'draw': # draw

                next_val = 0.5 # 0.0 if we follow Sutton/Barto, try it out yourself, min will lose finally

                # as it will not learn to minimize its regret! (there is no winning strategy for min if the

                # competitor learns)

                incomplete = True

            else: # we lost...

                next_val = 0.0

                incomplete = True



        if incomplete: 

            s,a,move_type = self.history[i]

            # learning from a non-finishing last move does no harm, this test could be left out

            if move_type: 

                update(next_val,s,a)

        

        s_next,a_next,move_type_next = self.history[i]

        s,a,move_type = self.history[i-1]

        if debug: print("S/T(S)/S_Next/T(S_Next):",s,T(s,a),s_next,T(s_next,a_next),"Move Type: ",move_type_next)

        if debug: print("Vals: ",

              Val(s,self.id),Val(T(s,a),self.id),Val(s_next,self.id),Val(T(s_next,a_next),self.id))

        if move_type_next:  

            next_val = Val(T(s_next,a_next),self.id)

            # Temporal difference learning

            update(next_val,s,a)

        i -= 1

        

debug = True

p1.backup(result)

debug = True

p2.backup(result)

# Saving and restoring the policy (one file for both players!)

import pickle 



# Versioning of the files would be nice

def save_policy(postfix):

    global V_table

    with open('policy_%s.bin' % postfix, 'wb') as f:

        pickle.dump(V_table, f)



def load_policy(postfix):

    global V_table

    with open('policy_%s.bin' % postfix, 'rb') as f:

        V_table = pickle.load(f)



print(p1.history)

print(p2.history)

V_table

        

# save_policy("very_short") # save policy to file
def play_cycle(p1,p2,result,step_size=0.1):    

    res = game(p1,p2)

    result[res] += 1

    p1.backup(res,step_size) # Learn!

    p2.backup(res,step_size)



def play_random(step_size=0.15):

    p1.epsilon = 1.0

    p2.epsilon = 1.0

    result = {'max':0,'min':0,'draw':0}

    for j in range(10):

        for i in range(200):

            play_cycle(p1,p2,result,step_size)

        print("Epsilon: ",p1.epsilon," - Round ",i+j*200,":",result," V_table size: ",len(V_table))        

    

def play_scheduled(schedule):

    '''

    Play in a schedule way:

    (epochs,games_per_epoch,((p1.epsilon,p2.epsilon),...))

    The epsilon-pairs reflect the epoch count

    '''

    games,epsilons = schedule

    epochs = len(epsilons)

    print("Playing ",games," games per epoch for",epochs,"epochs:")

    results = []

    for j in range(epochs):

        p1.epsilon = epsilons[j][0]

        p2.epsilon = epsilons[j][1]

        # Result per Epochs

        result = {'max':0,'min':0,'draw':0}

        for i in range(games):            

            play_cycle(p1,p2,result)

            if i%1000 == 0: 

                print(i+j*games," - Epsilon: ",p1.epsilon,",",p2.epsilon," - Round ",i+j*games,":",result," V_table size: ",len(V_table))

        print("Epoch finished - Epsilon: ",p1.epsilon,",",p2.epsilon," - Round ",i+j*games,":",result," V_table size: ",len(V_table))

        results.append(result)

    return results



# Convinience function to show start of tree (first three level)

def show_tree(data):

    s = [0,0,0,0,0,0,0,0,0]

    for i in range(9):

        s_new = s[:]

        s_new[i] = 1

        if tuple(s_new) in data:

            print(s_new,":",data[tuple(s_new)])

        else:

            print(s_new,": 0.5")

        for j in range(9):

            s_nnew = s_new[:]

            if i != j:

                s_nnew[j] = -1

                if tuple(s_nnew) in data:

                    print("   ",s_nnew,":",data[tuple(s_nnew)])

                else:

                    print(s_nnew,": 0.5")



import pprint



debug = False

V_table = {} # clear the V_table

random.seed(1000)



# Initial schedule (pure random exploration for p2, thus no learning takes place)

# play_scheduled((4000,((0.30,1),(0.20,1),(0.10,1),(0.05,1))))



# Fixed play

# play_scheduled((15000,((0,0),(0,0),(0,0),(0,0),(0,0))))

# pure random play

# play_scheduled((15000,((1,1),(1,1),(1,1),(1,1),(1,1))))



print("Round 0:",play_scheduled((7000,((0.2,0.2),(0.1,0.1),(0.05,0.05),(0.01,0.01)))))

save_policy("short") # save policy to file



# Test against a random p1 with fixed p2 policy

print("Round 1:",play_scheduled((10000,((1,0.0),(1,0.0),(1,0.0),(1,0.0),(1,0.0),(1,0.0)))))



# Test against a random p1 with fixed p2 policy, allow some more exploration

print("Round 2:",play_scheduled((10000,((1,0.3),(1,0.2),(1,0.1),(1,0.05),(1,0.01),(1,0.0)))))



# %timeit -n 1 -r 2 play_scheduled((2,200,((0.01,0.01),(0.01,0.01))))

# And finish it trying to re-learn draw-ing

print("Round 3:",play_scheduled((12000,((0.2,0.01),(0.1,0.01),(0.05,0.01),(0.01,0.01),(0.01,0.01)))))

save_policy("long") # save policy to file
# for key in sorted(V_table.keys()):

#    print("%s: %s" % (key, V_table[key]))
result = {'max':0,'min':0,'draw':0}

debug = True

play_cycle(p1,p2,result)

print("\n",result,"\n")

show_tree(V_table)
class MiniMax_Player:

    def __init__(self,id):

        self.id = id

    

    # Decide how to move in state s

    def move(self,s):

        # Determine possible moves

        pmoves = []

        for a in A:

            if V(s,a,self.id):

                value = minimax(T(s,a),100,self.id)

                if value == 0:

                    val = 0.5 # this will be a draw

                elif (value==100 and self.id=='max') or (value==-100 and self.id=='min'):

                    val = 1 # we will win

                else:

                    val = 0 # we will loose

                pmoves += [(a,val)]

        if debug: print(pmoves)

        # Choose one of the best moves (hm, the randomness here might be important

        # in some cases because of potential errors of the other player - note that

        # this can only IMPROVE our result, as the value we compute is

        # a lower bound resulting from optimal play of the opponent)

        _,max_value = max(pmoves,key=lambda x:x[1])

        max_actions = [(act,val) for (act,val) in pmoves if val == max_value]

        action,_ = random.choice(max_actions) # Actions with maximal value

        if debug: print("Maximal value choices:", max_actions)

        if debug: print("Chosen action ",action)

        return action

    

    def clear(self):

        pass # nothing here to clear
debug = False



print("Games of exploiting RL max player versus Minimax Player min:")

result = {'max':0,'min':0,'draw':0}

for j in range(100):

    res = game(Player('max',epsilon=0),MiniMax_Player('min'))

    result[res] += 1

print("Results: ",result)



print("Games of  Minimax Player max versus exploiting simple RL min player:")

load_policy("short")

result = {'max':0,'min':0,'draw':0}

for j in range(100):

    res = game(MiniMax_Player('max'),Player('min',epsilon=0))

    result[res] += 1

print("Results: ",result)



print("Games of  Minimax Player max versus exploiting improved RL min player:")

load_policy("long")

result = {'max':0,'min':0,'draw':0}

for j in range(100):

    res = game(MiniMax_Player('max'),Player('min',epsilon=0))

    result[res] += 1

print("Results: ",result)
# Show some infos about mm, the valuations in the solved tree



print(len(mm))

print((0,0,0,0,0,0,0,0,0),":",mm[(0,0,0,0,0,0,0,0,0)])

show_tree(mm)



def show_children(s,data,id):

    print(s,":",data[s])

    for i in range(9):

        s_new = list(s)

        if s[i]==0:

            s_new[i]=id

            s_new = tuple(s_new)

            if s_new in data:

                print("  ",s_new,":",data[s_new])

            else:

                print("State ",s_new," has not been considered before! Impossible to reach?")

            

show_children((1,-1,1,-1,0,0,0,0,0),mm,1)

show_children((1,-1,1,-1,0,0,1,0,0),mm,-1)

show_children((1,-1,1,-1,-1,0,1,0,0),mm,1)

show_children((1,-1,1,-1,-1,1,1,0,0),mm,-1)

show_children((1,-1,1,-1,-1,1,1,0,-1),mm,1)