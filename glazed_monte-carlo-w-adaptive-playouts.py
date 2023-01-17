agent_as_a_string = """

import time

import numpy as np

from random import choice

from math import sqrt

from numba import njit

from numba import prange



# The following several functions are run with the Numba JIT compiler

# resulting in a dramatic speed increase.



@njit() # Numba function

def win_for_tup_jit(b, tup, I, color):

    # b[] holds the board state.

    # tup[] holds 4 board positions that are in a row.

    # I is one of those positions.

    # Would placing a color piece on I, result in 4 in a row?

    for x in tup:

        if (not ((b[x] == color) or (b[x] == 0))):

            return False

        if (not ((b[x] == color) or (x == I))):

            return False

    return True  



@njit()

def winning_move_jit(b, tupCounts, spaceTups, x_in, color_in):

    # Would placing a color_in piece on I, result in 4 in a row?

    # b[] holds the board state.

    # tupCounts[] and spaceTups[,,] are lookup tables to help examine

    # different groups of 4 positions in a row.

    tup = np.zeros(4, np.int32)

    x = int(x_in)

    color = int(color_in)

    for n in range(tupCounts[x]):# for each tup

        for z in range(4):

            tup[z] = spaceTups [x, n, z]

        if (win_for_tup_jit(b, tup, x, color)):

            return(True)

    return False   



@njit()

def lowest_empty_row_jit(b, j):

    # Return the row for a stone placed in column j.

    # (My coordinates are upside down wrt the contest. )

    r = 6

    c = 7

    for i in range(r): # rows

        x = (i * c) + j

        if (b[x] == 0):

            return i

        # Paronoid check to avoid race conditions if Numba makes this parallel.

        #if ((b[x] == 0) and (i == 0)):

        #    return i

        #if ((b[x] == 0) and (i > 0) and (b[x - c] != 0)):

        #    return i

    return(r)



@njit()

def get_sensible_moves_jit(b, tupCounts, spaceTups, color, otherColor):

    # Return a list of moves worth considering, plus the status of the board.

    # Is there a winning or forced move? Return a list of length one.

    # Also return a status flag.

    # status 0 : tie game, no legal moves

    # status 1 : ongoing game

    # status 2 : Color will win by playing the (single) move in obviousMoves[]

    r = 6 # number of rows

    c = 7 # number of columns

    N = 42 # number of board spaces

    legalMoves = [np.int32(x) for x in range(0)] # weird hack so numba can guess what type is in the list

    for j in range(c):

        i = lowest_empty_row_jit(b, j)

        if (i < r): # legal move

            x = (i * c) + j

            if (winning_move_jit(b, tupCounts, spaceTups, x, color)):

                #print("win")

                obviousMoves = [x]

                return (obviousMoves, 2)

            legalMoves.append(x)

    

    if (len(legalMoves) == 0):

        return (legalMoves, 0)  # tie game

    

    if (len(legalMoves) == 1):

        return (legalMoves, 1)  # ongoing game

    

    for x in legalMoves:

        if (winning_move_jit(b, tupCounts, spaceTups, x, otherColor)):

            #print("forced")

            obviousMoves = [x]

            return (obviousMoves, 1)

    

    lm = legalMoves.copy()

    for x in lm:

        if (x + c < N):

            b[x] = color # temporarily place stone here

            if ((len(legalMoves) > 1) and (winning_move_jit(b, tupCounts, spaceTups, x + c, otherColor))):

                legalMoves.remove(x)

            b[x] = 0 # remove temporarily stone

    return (legalMoves, 1)  # no obvious move





@njit()

def friendly_tupple_jit(b, tup, x, color):

    # tup[] holds 4 board positions that are in a row.

    # x is one of those positions.

    # If there are no othercolor pieces in tup, how many color pieces are there?

    count = np.int32(0)

    nope = False

    if (b[x] != 0):

        return (-1)

    for i in range(4):

        z = tup[i]

        if ((b[z] != color) and (b[z] != 0)):

            nope = True

        if (b[z] == color):

            count += 1

    if (nope): return(-1)

    return count



@njit()

def calcMoveFeatures_jit(b, tupCounts, spaceTups, x, color):

    # calculate features for a move at position x and return in feat[] 

    # tupCounts[] and spaceTups[,,] are lookup tables to help examine

    # different groups of 4 positions in a row.

    c = 7

    r = 6

    feat = np.zeros(22, np.int32)

    tup = np.zeros(4, np.int32)

    #self.feat.fill(0)

    otherColor = 2

    if (color == 2):

        otherColor = 1

    i = x // c

    j =  x % c

    #feat = feat * 0  # clear feat[]

    feat[0] = min(j, (c - 1) - j)  # distance from edge

    tups = tupCounts[x]

    for n in range(tups):

        #tup = self.spaceTups [x, n, :]

        for z in range(4): 

            tup[z] = spaceTups [x, n, z]

        count = friendly_tupple_jit(b, tup, x, color)

        if (count > -1):

            feat[count + 1] += 1

        count = friendly_tupple_jit(b, tup, x, otherColor)

        if (count > 0):

            feat[count + 4] += 1

            

    if (i >= r - 1):

        return feat # we're on the top row, so leave all other features at zero

    I = i + 1 # looking at space above x

    b[x] = color  # temporarily put friendly stone on space x

    xp = (I * c) + j

    tups = tupCounts[xp]

    for n in range(tups):

        #tup = self.spaceTups [xp, n, :]

        for z in range(4):

            tup[z] = spaceTups [xp, n, z]

        count = friendly_tupple_jit(b, tup, xp, color)

        if (count > -1):

            feat[count + 8] += 1

        count = friendly_tupple_jit(b, tup, xp, otherColor)

        if (count > 0):

            feat[count + 11] += 1

        

    b[x] = 0  # remove friendly stone from space x

    

    if (i >= r - 2):

        return feat # we're on the next-to top row, so leave all other features at zero 

    I = i + 2 # looking at space above x

    b[x] = color  # temporarily put friendly stone on space x

    b[xp] = otherColor  # temporarily put enemy stone on space xp

    xpp = (I * c) + j

    tups = tupCounts[xpp]

    for n in range(tups):

        #tup = self.spaceTups [xpp, n, :]

        for z in range(4):

            tup[z] = spaceTups [xpp, n, z]

        count = friendly_tupple_jit(b, tup, xpp, color)

        if (count > -1):

            feat[count + 15] += 1

        count = friendly_tupple_jit(b, tup, xpp, otherColor)

        if (count > 0):

            feat[count + 18] += 1

    

    b[x] = 0  # remove friendly stone from space x

    b[xp] = 0  # remove enemy stone from space xp

    return feat

    

@njit()          

def calc_meta_features_jit(feat, x):

    #calculate meta-features for a move at position x and return in metaFeat[]

    

    # all binary features

    #metaFeat = metaFeat * 0  # clear metaFeat[]

    metaFeat = np.zeros((4 + 6 + (21 * 3)), np.int32)

    #self.metaFeat .fill(0)

    c = 7

    i = x // c

    n = 0

    y = feat [0] # distance from edge -> 4 possibilities, 4 'binary' variables

    metaFeat [y] = 1 # only 1 can be non-zero

    n += 4

    # row -> 6 (essentially boolean) parameters

    metaFeat [n + i] = 1 # only 1 can be non-zero

    n += 6

    for f in range(1, len(feat)):

        if (feat[f] == 0):

            metaFeat[n] = 1

        elif (feat[f] == 1):

            metaFeat[n + 1] = 1

        elif (feat[f] > 1):

            metaFeat[n + 2] = 1

        n += 3

    return metaFeat



@njit()

def linear_move_scores_jit(b, tupCounts, spaceTups, wts, moves, color):

    # For every move in the list, calculate a score and return them in scores[]

    # moves[] holds the list of moves.

    # b[] holds the board state.

    # wts[] holds the linear weights for the features.

    # tupCounts[] and spaceTups[,,] are lookup tables to help examine

    # different groups of 4 positions in a row.

    #min_score = 0.05

    scores = [np.float64(x) for x in range(0)] # weird hack so numba can guess what type belongs in the list

    total = 0.0

    for i in prange(len(moves)):

        x = moves[i]

        feat = calcMoveFeatures_jit(b, tupCounts, spaceTups, x, color) # fills feat

        metaFeat = calc_meta_features_jit(feat, x)         # fills metafeat

        score = np.sum (metaFeat * wts) 

        scores.append(score)

        total += score

    #scores = scores / np.sum(scores)

    for i in range(len(scores)):

        scores[i] /= total

    #for sc in scores:

    #    if (sc < min_score): sc = min_score

    return scores





@njit()  

def choose_linear_move_jit(b, tupCounts, spaceTups, wts, color):

    # Get the sensible_moves[], score them, return the move with the highest score.

    # score is a weighted sum of features.

    #print("choose_linear_move_jit start")

    otherColor = 2

    if (color == 2):

        otherColor = 1

    sensible_moves, status = get_sensible_moves_jit(b, tupCounts, spaceTups, color, otherColor) # also sets self.status

    #print("choose_linear_move_jit calculated")

    if (len(sensible_moves) == 1):   # If it's a win, there will only be one move, so it returns w/ correct status.

        return(sensible_moves[0], status)

    if (len(sensible_moves) == 0):

        return(-1, 0)   # tie  

    scores = linear_move_scores_jit(b, tupCounts, spaceTups, wts, np.array(sensible_moves), color)

    

    #print("choose_linear_move_jit calculated")

    x = sensible_moves[int(np.argmax(np.array(scores)))]

    #print(sensible_moves)

    #print(scores)

    #print(x)

    return (x, 1)





class GAME_MANAGER():

    

    # This class holds the board, the weights for move features and some lookup tables.

    # It has a method that calculates the tables.

    # Note: "tupples" doesn't mean Python tupples.

    

    def __init__(self):

        self.c = 7

        self.r = 6

        self.N = self.c * self.r

        self.K = 4

        self.b = np.zeros((self.N), np.int32)

        self.tupCounts = np.zeros((self.N), np.int32) # how many tupples contain this space

        self.spaceTups = np.zeros((self.N, 16, self.K), np.int32)# all the tupples 

        self.feat = np.zeros(22, np.int32)

        self.metaFeat = np.zeros((4 + 6 + (21 * 3)), np.int32)

        self.precalcTups()  # calculates tupCounts[] and spaceTups[,,]

        self.wts = np.array([\

0, 3.30366111407672, 6.48699261816764, 16.0570530870234, 

0, 6.89154554317436, 12.2291091073301, 14.9610249625214, 

8.2764687731129, 5.36422024027117, 14.3516002179586, 2.1021183007845, 

0, 0.719857647774551, 2.34487198327812, 11.2057343305379, 

0, 27.8825518166704, 43.8228479471015, 0, 

1.99103479234425, 0.726840470816211, 0.560352057248385, 4.22057034862125, 

6.83487190573809, 0, 17.491336765736, 35.714660516821, 

0, 5.87818176004432, 39.6996173910764, 0, 

3.6234074451734, 4.49879127350228, 0.806205543215103, 1.71224017973768, 

3.8749091066669, 2.53500463742859, 4.460882129649, 1.37971933373037, 

44.3252027404174, 29.7283585631117, 0, 0, 

2.19363385696125, 1.47116517810174, 11.3140778960712, 2.27101838697622, 

0, 1.74611338807736, 6.51927915172379, 0, 

2.94225634035759, 0, 3.50453812682987, 0, 

0.267382395928564, 1.98693848260618, 0, 1.58478060332119, 

4.71565983853622, 0, 17.8551290536231, 7.25762646495066, 

5.7077735522044, 1.10042450222865, 0, 8.22587152564711, 

4.70559492215376, 0.324579480451124, 16.3108479710915, 8.11547320076288, 

0, ]) 

        

    

    def precalcTups(self):

        # tupCounts[] and spaceTups[,,] are lookup tables to help examine

        # different groups of 4 positions in a row.

        #tupCounts *= 0

        # horizontal

        tup = np.zeros((self.K), np.int32)

        for j in range (0, (self.c - self.K) + 1):

            for i in range(self.r):

                # fill tup[]

                for h in range(self.K):

                    tup[h] = (i * self.c) + j + h

                for h in range(self.K):

                    x = tup [h]

                    n = self.tupCounts [x]

                    for z in range(self.K):

                        self.spaceTups [x, n, z] = tup [z]

                    self.tupCounts [x] = n + 1

        # vertical

        for j in range (self.c):

            for i in range(0, (self.r - self.K) + 1):

                # fill tup[]

                for h in range(self.K):

                    tup[h] = ((i + h) * self.c) + j

                for h in range(self.K):

                    x = tup [h]

                    n = self.tupCounts [x]

                    for z in range(self.K):

                        self.spaceTups [x, n, z] = tup [z]

                    self.tupCounts [x] = n + 1

        # diagonal up, up

        for j in range (0, (self.c - self.K) + 1):

            for i in range(0, (self.r - self.K) + 1):

                # fill tup[]

                for h in range(self.K):

                    tup[h] = ((i + h) * self.c) + j + h

                for h in range(self.K):

                    x = tup [h]

                    n = self.tupCounts [x]

                    for z in range(self.K):

                        self.spaceTups [x, n, z] = tup [z]

                    self.tupCounts [x] = n + 1

        # diagonal something, something...

        for j in range (0, (self.c - self.K) + 1):

            for i in range(self.r-1, self.r - self.K, -1):

                # fill tup[]

                for h in range(self.K):

                    tup[h] = ((i - h) * self.c) + j + h

                for h in range(self.K):

                    x = tup [h]

                    n = self.tupCounts [x]

                    for z in range(self.K):

                        self.spaceTups [x, n, z] = tup [z]

                    self.tupCounts [x] = n + 1

        #print("precalcTups",self.tupCounts)

        return

     

    #@njit()       

    def reset_b(self, B):

        for i in range(len(B)):

            self.b[i] = B[i]





# This class uses all moves as first (AMAF) which is similar to RAVE in MCTS

# There are 42 positions on the board. 

# raveS[] holds the score for each position.

# raveV[] holds the number of visits for each position.

# This class holds methods for UCT-like move selection using RAVE plus policy scores.

class AMAF():

    def __init__(self, N):

        self.N = N

        self.raveS = np.zeros (N, int)  # myColor AMAF

        self.raveV = np.zeros (N, int)

        

    def clearit(self):

        self.raveS.fill(0)

        self.raveV.fill(0)

   

    def amaf_scores(self, moves):

        #return amaf scores for moves 

        scores = []

        for x in moves:

            scores.append( self.raveS[x] / max(1, self.raveV[x]) )

        return(scores)

    

    def PUCT_scores(self, Cpuct, moves, scores):

        if (len(moves) == 0) : print("PUCT_scores BADNESS! length 0")

        if (len(scores) == 0) : print("scores PUCT_scores BADNESS! length 0")

        encounters = 0

        pscores = scores.copy()

        for x in moves:

            encounters += self.raveV[x]

        if (encounters == 0):

            return (pscores)

        for k in range(len(moves)):

            x = moves[k]

            u = self.raveS[x] / max(1, self.raveV[x])

            u += scores[k] * Cpuct * sqrt(encounters) / (self.raveV[x] + 1)

            pscores[k] = u

        return(pscores)

    

    def PUCT_no_policy(self, Cpuct, moves):

        # Returns a move using UCT-like algorithm.

        # Used when no policy scores are avaulable.

        # moves[] holds the moves to be considered.

        if (len(moves) == 0) : print("PUCT_no_policy BADNESS! length 0")

        encounters = 0

        best_score = 0

        best = moves[0]

        for x in moves:

            encounters += self.raveV[x]

        if (encounters == 0):

            return (choice(moves))

        for x in moves:

            u = self.raveS[x] / max(1, self.raveV[x])

            if (u == 2):

                return(x)  # perfect score, so why explore

            u += (1.0 /len(moves) ) * Cpuct * sqrt(encounters) / (self.raveV[x] + 1)

            if (u >= best_score):

                best_score = u

                best = x

        #print("amaf_puct", x)

        return(best)

    

    def PUCT(self, Cpuct, moves, scores):

        # Returns a move using an UCT-like algorithm.

        # scores[] holds the policy scores.

        # moves[] holds the moves to be considered.

        if (len(moves) == 0) : print("PUCT BADNESS! length 0")

        if (len(scores) == 0) : print("scores PUCT BADNESS! length 0")

        encounters = 0

        best_score = 0

        best = moves[0]

        for x in moves:

            encounters += self.raveV[x]

        if (encounters == 0):

            return (choice(moves))

        for k in range(len(moves)):

            x = moves[k]

            u = self.raveS[x] / max(1, self.raveV[x])

            if (u == 2):

                return(x)  # perfect score, so why explore

            u += scores[k] * Cpuct * sqrt(encounters) / (self.raveV[x] + 1)

            if (u >= best_score):

                best_score = u

                best = x

        #print("amaf_puct", x)

        return(best)

    

    

    def reinforce(self, moves, reward):

        # moves holds a list of all the moves (for this color) in one playout.

        for k in moves:

            self.raveV[k] += 1  

            self.raveS[k] += reward





class BRAIN():

    # This class chooses a move using MonteCarlo with adaptive playouts.



    def __init__(self):

        self.gm = GAME_MANAGER() # holds the board and some tables

        self.colorAMAF = AMAF(self.gm.N) # AMAF object for color moves in playouts

        self.otherAMAF = AMAF(self.gm.N) # AMAF object for otherColor moves in playouts

        

     

    def brain_linear_move(self, b, color, V = False):

        # Pick a move just using heuristics. (Not called)

        ts = time.time()

        self.gm.reset_b(b)

        otherColor = 1 + (2 - color)

        moves, status = get_sensible_moves_jit(self.gm.b, self.gm.tupCounts, self.gm.spaceTups, color, otherColor)

        linear_move_scores_jit(self.gm.b, self.gm.tupCounts, self.gm.spaceTups, self.gm.wts, np.array(moves), otherColor) 

       

        x, status = choose_linear_move_jit(self.gm.b, self.gm.tupCounts, self.gm.spaceTups, self.gm.wts, color)

        if (V): print(" JIT tm",time.time() - ts)   

   

        return(x, status)

    

    

    def clearAMAF(self):

        self.colorAMAF.clearit()

        self.otherAMAF.clearit()

        

            

    def MC_adaptive_move(self, root, color, start, time_limit, POLICY):

        # Choose a move using Monte Carlo playouts.

        # First check that there's not only one real choice.

        # Use all time available.

        # Use adaptive playouts.

        # Play move with most visits at the root

        # Only reinforce discretionary moves.

        # if (POLICY): calculate and use heuristic move scores ('priors') in playouts

        

        self.clearAMAF()

        showit = True

        

        CpuctRoot = 7.0 # exploration term for root

        Cpuct = 4.0     # exploration term for all other nodes

        otherColor = 2

        if (color == 2):

            otherColor = 1

        

        self.gm.b = root.copy()   # Set b in game_manager (This is the root board state)

        # Get sensible moves 

        legalMoves, status = get_sensible_moves_jit(self.gm.b, self.gm.tupCounts, self.gm.spaceTups, color, otherColor)

        # If the move is obvious, we return it along with the status

        if (status == 2):

            print("win")

        #    return(legalMoves[0], status)

        if (len(legalMoves) == 1):

            print("forced move")

            return(legalMoves[0], status)

        if (len(legalMoves) == 0):

            print("tie")

            return(-1, 0)   # tie

        

        # There must be 2 or more sensible moves to choose from

        use_policy = POLICY

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # run MonteCarlo playouts for all 'sensible' moves

        # 

        

        rootMoves = legalMoves.copy()

      

        training_visits = np.zeros(7)

        root_visits = np.zeros(len(rootMoves))

        root_rewards = 0.0

        if (use_policy):

            root_scores = linear_move_scores_jit(self.gm.b, self.gm.tupCounts, self.gm.spaceTups, self.gm.wts, np.array(rootMoves), color) 

        else:

            root_scores = np.ones(len(rootMoves))

            root_scores = root_scores / np.sum(root_scores)

        # run the playouts

        end = time.time()

        reps = 0 

        

        # Run as many playouts as time permits (within reason).

        while ((reps == 0) or ((end-start < time_limit) and (reps < 30000))):

            # Run a single Monte carlo playout - with adaptive playouts using AMAF and priors

            use_policy = POLICY

            reps += 1

            self.gm.b = root.copy() # reset board to root state

            movesColor = [] # moves made by color this playout

            movesOther = [] # moves made by otherColor this playout

            # make a move at the root, a color move

            if (use_policy):

                j = self.colorAMAF.PUCT(CpuctRoot, rootMoves, root_scores)

            else:

                j = self.colorAMAF.PUCT_no_policy(CpuctRoot, rootMoves)

            training_visits[j % 7] += 1 # keep track of starting column

            #print("root move", j)

            self.gm.b[j] = color

            movesColor.append(j)

            cnt = 1

            status = 1

            while (status == 1):

                # ~~~~~~~~~~~~~~~~~~~ otherColor moves ~~~~~~~~~~~~~~~

                moves, status = get_sensible_moves_jit(self.gm.b, self.gm.tupCounts, self.gm.spaceTups, otherColor, color)

                if (status == 2):

                    result = otherColor

                    x = moves[0]

                    self.gm.b[x] = otherColor

                    movesOther.append(x)

                if (status == 0):

                    result = 0

                if (status == 1):

                    if (len(moves) == 1):

                        x = moves[0]

                        #movesOther.append(x) #sort of a forced move so better not to use it

                    else:

                        if (use_policy):

                            scores = linear_move_scores_jit(self.gm.b, self.gm.tupCounts, self.gm.spaceTups, self.gm.wts, np.array(moves), otherColor) 

                            x = self.otherAMAF.PUCT(Cpuct, moves, scores)

                        else:

                            x = self.otherAMAF.PUCT_no_policy(Cpuct, moves)

                        movesOther.append(x)    # use this move to reinforce RAVE

                    self.gm.b[x] = otherColor

                    cnt += 1

                    # ~~~~~~~~~~~~~~~~~~~ color moves ~~~~~~~~~~~~~~~

                    moves, status = get_sensible_moves_jit(self.gm.b, self.gm.tupCounts, self.gm.spaceTups, color, otherColor)

                    if (status == 2):

                        result = color

                        x = moves[0]

                        self.gm.b[x] = color

                        movesColor.append(x)

                    if (status == 0):

                        result = 0

                    if (status == 1):

                        if (len(moves) == 1):

                            x = moves[0]

                            #movesColor.append(x) #sort of a forced move so better not to use it

                        else:

                            if (use_policy):

                                scores = linear_move_scores_jit(self.gm.b, self.gm.tupCounts, self.gm.spaceTups, self.gm.wts, np.array(moves), color) 

                                x = self.colorAMAF.PUCT(Cpuct, moves, scores)

                            else:

                                x = self.colorAMAF.PUCT_no_policy(Cpuct, moves)

                            movesColor.append(x)

                        self.gm.b[x] = color

                        cnt += 1

           

            reward = 0

            if (result == color):

                reward = 2

            elif (result == 0):

                reward = 1

            root_rewards += reward

            # reinforce rave (AMAF)

            self.colorAMAF.reinforce(movesColor, reward)

            self.otherAMAF.reinforce(movesOther, 2 - reward)

            end = time.time()

            if (end-start > time_limit):

                break

            

        #print(scores)

        end = time.time()

        #print("root_scores", root_scores)

        for k in range(len(rootMoves)):   # for display/diagnostics

            x = rootMoves[k]

            root_scores[k] = 100*self.colorAMAF.raveS[x] / max(1, self.colorAMAF.raveV[x])

        #print("root_scores", root_scores)

        j = int(np.argmax(training_visits)) # the column

        i = lowest_empty_row_jit(root, j) # the row

        x = j + i * self.gm.c

        root_rewards /= reps  # average reward for color

        #print (root_visits, 'reps', reps, end-start, x % 7)

        #print (training_visits, 'reps', reps, end-start, x % 7)

        # end of Monte Carlo block

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #b = root.copy() # reset board to root state

        if (showit): 

            print (root_visits, "root_rewards", root_rewards, 'stone_count', np.sum(root != 0), 'playouts', reps, "***")

            #net_move_scores   (b, rootMoves, color, otherColor, showit = True)

        

        status = 1 # ongoing game

        # return column and status

        return (x, status)  

   

brain = BRAIN() # The object that holds everything





def mirror_top_to_bottom(B):

   # Change board to my (upside down) internal coordinates 

    b = np.zeros((brain.gm.N), int)

    for i in range(6): # rows of b

        for j in range(7):

            x = (i * 7) + j

            mx = ((5 - i) * 7) + j # mirrored top to bottom

            b[mx] = B[x]

    return(b)    





    

def my_agent(observation, configuration):

    # calls brain to make a move 

    bail_early = False

    start = time.time()

    #The amount of thinking time per move. Add small margin.

    time_limit = configuration.actTimeout - 0.25

    

    # My coordinates are upside down, so I must flip the input

    b = mirror_top_to_bottom(np.array(observation.board))

    

    

    stone_count = np.sum(b != 0)

    if (stone_count <= 1): time_limit = configuration.agentTimeout / 2

    color = observation.mark 

    

    

    if (stone_count <= 1): 

       print('time_limit', time_limit)

        

        

    rootMove, status = brain.MC_adaptive_move(b, color, start, time_limit, POLICY = True)

    

    

    #x, status = brain.brain_linear_move(b, color) just uses heuristics, no search

    if (bail_early and ((status == 2) or (stone_count > 40))): 

        print('About to win (or tie); cant have that, bailing')

        return(-1)   # Cause an error so as to fail validation and not waste a submission slot.

    

    x = rootMove # x is the board position (one of 42)

    x = x % brain.gm.c  # We must return the column

    

    

    print('time_limit', time_limit, "my duration",time.time() - start, 'x', x)

    return(x)

"""

agent_file = open("submission.py", "w")

agent_file.write(agent_as_a_string)

agent_file.close()