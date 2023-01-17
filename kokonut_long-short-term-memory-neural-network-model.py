import numpy as np
import pandas as pd
from random import randint
import time
import tensorflow as tf

print('Done!')
class Board:
    
    valid_play = [11,12,13,21,22,23,31,32,33]
    
    def __init__(self):
        self.play = [0]*9 #List of plays coordinate, ordered
        self.winner = '.' #State of the game. 'X' if P1 won, 'O' if P2 won, '+' if draw, '.' if no winner yet
        self.state = ['.']*9 #State of the board, for display purposes
        self.verbose = False #If we want to print the board play by play
        
    def play2state(self):
        #rewrite self.state after a play
        current = -1
        poss = ['.','O','X']
        for i in self.play:
            if i != 0:
                j = (i//10 - 1)*3 + (i%10) - 1
                self.state[j] = poss[current]
            current = current * (-1)
    
    def newplay(self,coord):
        #Add new play (if valid) and update other variables
        if self.winner != '.':
            print("Erreur : partie déjà terminée")
        elif (not coord in self.valid_play):
            print("Erreur : coup hors tableau.")
        elif (coord in self.play):
            print("Erreur : coup déjà joué.")
        else:
            self.play[self.play.index(0)] = coord
            self.play2state()
            if self.verbose:
                self.display()
            self.status()
            
    def available(self):
        #Return list of available move to pick from
        t = self.valid_play.copy()
        for i in self.play:
            if i != 0:
                t.remove(i)
        return t
    
    def print(self):
        #Debugging function
        print("Play : {}.\nState : {}.\nWinner = {}.".format(self.play, self.state, self.winner))
        
    def display(self):
        #Print the board state
        print("")
        for i in range(9):
            print(self.state[i],end='')
            if (i+1)%3 == 0:
                print('\n',end='')
            else:
                print(' ',end='')
        print("")
    
    def status(self):
        #Check the board, if there is a winner
        if ((self.state[0] == self.state[4] == self.state[8]) or (self.state[2] == self.state[4] == self.state[6])) and (self.state[4] != '.'):
            self.winner = self.state[4]
        else:
            for i in range(3):
                if (self.state[i] == self.state[i+3] == self.state[i+6]) and (self.state[i] != '.'):
                    self.winner = self.state[i]
                elif (self.state[3*i] == self.state[3*i+1] == self.state[3*i+2]) and (self.state[3*i] != '.'):
                    self.winner = self.state[3*i]
        if self.winner != '.':
            if self.verbose:
                print("Partie terminée ! Victoire des", self.winner, '!')
        elif (self.winner == '.') and (len(self.available())==0):
            self.winner = '+'
            if self.verbose:
                print("Partie terminée ! Match nul !")
   
def randomIA(B):
    #Make a random move on a Board B
    if B.winner != '.':
        return None
    poss = B.available()
    L = len(poss)-1
    new = poss[randint(0,L)]
    B.newplay(new)
    
def autogame(B):
    #Playing random moves to get a valid board
    while B.winner == '.':
        randomIA(B)

def simu(nsim=10):
    #Simulating nsim games and exporting their results (plays, winner)
    #With outputs encoded for a LSTM neural network
    dataset = []
    win = []
    for i in range(nsim):
        B = Board()
        autogame(B)
        dataset.append([[x] for x in B.play])
        foo = [0]*3
        if B.winner == 'X':
            foo[0] = 1
        elif B.winner == 'O':
            foo[2] = 1
        elif B.winner == '+':
            foo[1] = 1
        win.append(foo)
    return dataset, win

print('Done!')
a,b = simu(5) #Simulate 5 games
print(np.squeeze(a)) #List of plays
print(b) #List of end-game status
tf.reset_default_graph() #This need to be run everytime you need to reset the neural network
n1 = 10000 #Training sample size
n2 = 1000 #Testing sample size
train_input, train_output = simu(n1)
test_input, test_output = simu(n2)

print('Done!')
data = tf.placeholder(tf.float32, [None, 9, 1])
target = tf.placeholder(tf.float32, [None, 3])

num_hidden = 18 #LSTM parameter
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

print('Done!')
batch_size = int(n1/10)
no_of_batches = 10
epoch = 200

T1 = time.time()
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr += batch_size
        sess.run(minimize,{data: inp, target: out})
    if (i+1)%20 == 0:
        print("Epoch - {}/{}".format(i+1,epoch))
T2 = time.time()

print('Runing time : {:.1f}s.'.format(T2-T1))
incorrect = sess.run(error,{data: test_input, target: test_output})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
draws_index = [(test_output[i][1] == 1) for i in range(n2)]
print('Wrong prediction : {:.3f}. Draw rate : {:.3f}.'.format(incorrect, sum(draws_index)/n2))
inp_draws = [test_input[i] for i in range(n2) if draws_index[i]]
out_draws = [test_output[i] for i in range(n2) if draws_index[i]]
err_draws = sess.run(error,{data: inp_draws, target: out_draws})
print('Error on draws : {:3.1f}%.'.format(100*err_draws))

inp_nodraws = [test_input[i] for i in range(n2) if (not draws_index[i])]
out_nodraws = [test_output[i] for i in range(n2) if (not draws_index[i])]
err_nodraws = sess.run(error,{data: inp_nodraws, target: out_nodraws})
print('Error on non-draws : {:3.1f}%.'.format(100*err_nodraws))