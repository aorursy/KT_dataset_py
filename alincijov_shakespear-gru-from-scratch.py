import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os
np.random.seed(0)
df = pd.read_csv('../input/shakespeare-plays/Shakespeare_data.csv')

df.head()
data = ' '.join(df['PlayerLine'].astype(str))

data[:100]
chars = sorted(list(set(data)))

data_size, vocab_size = len(data), len(chars)

print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }

ix_to_char = { i:ch for i,ch in enumerate(chars) }
def sigmoid(input, deriv=False):

    if deriv:

        return input*(1-input)

    else:

        return 1 / (1 + np.exp(-input))



def tanh(input, deriv=False):

    if deriv:

        return 1 - input ** 2

    else:

        return np.tanh(input)
def softmax(input):

    out = np.exp(input - np.max(input))

    return out / out.sum()
N, h_size, o_size = vocab_size, vocab_size, vocab_size

seq_length = 25

learning_rate = 1e-1
Wz = np.random.rand(h_size, N) * 0.1 - 0.05

Uz = np.random.rand(h_size, h_size) * 0.1 - 0.05

bz = np.zeros((h_size, 1))



Wr = np.random.rand(h_size, N) * 0.1 - 0.05

Ur = np.random.rand(h_size, h_size) * 0.1 - 0.05

br = np.zeros((h_size, 1))



Wh = np.random.rand(h_size, N) * 0.1 - 0.05

Uh = np.random.rand(h_size, h_size) * 0.1 - 0.05

bh = np.zeros((h_size, 1))



Wy = np.random.rand(o_size, h_size) * 0.1 - 0.05

by = np.zeros((o_size, 1))



beta = [Wy,  Wh,  Wr,  Wz,  Uh,  Ur,  Uz,  by,  bh,  br,  bz]
n, p = 0, 0

smooth_loss = -np.log(1.0/vocab_size)*seq_length



mdWy, mdWh, mdWr, mdWz = np.zeros_like(Wy), np.zeros_like(Wh), np.zeros_like(Wr), np.zeros_like(Wz)

mdUh, mdUr, mdUz = np.zeros_like(Uh), np.zeros_like(Ur), np.zeros_like(Uz)

mdby, mdbh, mdbr, mdbz = np.zeros_like(by), np.zeros_like(bh), np.zeros_like(br), np.zeros_like(bz)



gamma = [mdWy,mdWh,mdWr,mdWz,mdUh,mdUr,mdUz,mdby,mdbh,mdbr,mdbz]
def sample(h, seed_ix, n):

    # Initialize first word of sample ('seed') as one-hot encoded vector.

    x = np.zeros((vocab_size, 1))

    x[seed_ix] = 1

    ixes = [seed_ix]

    

    for t in range(n):

        # Calculate update and reset gates

        z = sigmoid(np.dot(Wz, x) + np.dot(Uz, h) + bz)

        r = sigmoid(np.dot(Wr, x) + np.dot(Ur, h) + br)

        

        # Calculate hidden units

        h_hat = tanh(np.dot(Wh, x) + np.dot(Uh, np.multiply(r, h)) + bh)

        h = np.multiply(z, h) + np.multiply((1 - z), h_hat)

        

        # Regular output unit

        y = np.dot(Wy, h) + by

        

        # Probability distribution

        p = softmax(y)



        # Choose next char according to the distribution

        ix = np.random.choice(range(vocab_size), p=p.ravel())

        x = np.zeros((vocab_size, 1))

        x[ix] = 1

        ixes.append(ix)

    

    return ixes
def forward(inputs, alpha):

    x, z, r, h_hat, h, y, p, loss = alpha

    

    for t in range(len(inputs)):

        # Set up one-hot encoded input

        x[t] = np.zeros((vocab_size, 1))

        x[t][inputs[t]] = 1

        

        # Calculate update and reset gates

        z[t] = sigmoid(np.dot(Wz, x[t]) + np.dot(Uz, h[t-1]) + bz)

        r[t] = sigmoid(np.dot(Wr, x[t]) + np.dot(Ur, h[t-1]) + br)

        

        # Calculate hidden units

        h_hat[t] = tanh(np.dot(Wh, x[t]) + np.dot(Uh, np.multiply(r[t], h[t-1])) + bh)

        h[t] = np.multiply(z[t], h[t-1]) + np.multiply((1 - z[t]), h_hat[t])

        

        # Regular output unit

        y[t] = np.dot(Wy, h[t]) + by

        

        # Probability distribution

        p[t] = softmax(y[t])

        

        # Cross-entropy loss

        loss = -np.sum(np.log(p[t][targets[t]]))

        

    return x, z, r, h_hat, h, y, p, loss
def backward(inputs, alpha):

    

    x, z, r, h_hat, h, y, p, loss = alpha

    

    dWy, dWh, dWr, dWz = np.zeros_like(Wy), np.zeros_like(Wh), np.zeros_like(Wr), np.zeros_like(Wz)

    dUh, dUr, dUz = np.zeros_like(Uh), np.zeros_like(Ur), np.zeros_like(Uz)

    dby, dbh, dbr, dbz = np.zeros_like(by), np.zeros_like(bh), np.zeros_like(br), np.zeros_like(bz)

    dhnext = np.zeros_like(h[0])

    

    # Backward prop

    for t in reversed(range(len(inputs))):

        # ???loss/???y

        dy = np.copy(p[t])

        dy[targets[t]] -= 1

        

        # ???loss/???Wy and ???loss/???by

        dWy += np.dot(dy, h[t].T)

        dby += dy

        

        # Intermediary derivatives

        dh = np.dot(Wy.T, dy) + dhnext

        dh_hat = np.multiply(dh, (1 - z[t]))

        dh_hat_l = dh_hat * tanh(h_hat[t], deriv=True)

        

        # ???loss/???Wh, ???loss/???Uh and ???loss/???bh

        dWh += np.dot(dh_hat_l, x[t].T)

        dUh += np.dot(dh_hat_l, np.multiply(r[t], h[t-1]).T)

        dbh += dh_hat_l

        

        # Intermediary derivatives

        drhp = np.dot(Uh.T, dh_hat_l)

        dr = np.multiply(drhp, h[t-1])

        dr_l = dr * sigmoid(r[t], deriv=True)

        

        # ???loss/???Wr, ???loss/???Ur and ???loss/???br

        dWr += np.dot(dr_l, x[t].T)

        dUr += np.dot(dr_l, h[t-1].T)

        dbr += dr_l

        

        # Intermediary derivatives

        dz = np.multiply(dh, h[t-1] - h_hat[t])

        dz_l = dz * sigmoid(z[t], deriv=True)

        

        # ???loss/???Wz, ???loss/???Uz and ???loss/???bz

        dWz += np.dot(dz_l, x[t].T)

        dUz += np.dot(dz_l, h[t-1].T)

        dbz += dz_l

        

        # All influences of previous layer to loss

        dh_fz_inner = np.dot(Uz.T, dz_l)

        dh_fz = np.multiply(dh, z[t])

        dh_fhh = np.multiply(drhp, r[t])

        dh_fr = np.dot(Ur.T, dr_l)

        

        # ???loss/???h??????????

        dhnext = dh_fz_inner + dh_fz + dh_fhh + dh_fr

        

    return dWy, dWh, dWr, dWz, dUh, dUr, dUz, dby, dbh, dbr, dbz
def train(inputs, targets, hprev):

    # Initialize variables

    # x, z, r, h_hat, h, y, p, loss

    alpha = {}, {}, {}, {}, {-1: hprev}, {}, {}, 0

    sequence_loss = 0



    # Forward prop

    alpha = forward(inputs, alpha)

    sequence_loss += alpha[-1]



    # Backward prop

    grads = backward(inputs, alpha)



    # alpha[4] - h

    return sequence_loss, grads, alpha[4][len(inputs) - 1]
def update(grads, beta):

    for param, dparam, mem in zip(beta,grads,gamma):

        # clip the array to be between -5 and 5

        np.clip(dparam, -5, 5, out=dparam)

        mem += dparam * dparam

        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    

    return beta
for r in range(100001):

    # Reset memory if appropriate

    if p + seq_length + 1 >= len(data) or n == 0:

        hprev = np.zeros((h_size, 1))

        p = 0

    

    # Get input and target sequence

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]

    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]



    # Sample from model and print result

    if n % 20000 == 0:

        sample_ix = sample(hprev, inputs[0], 1000)

        txt = ''.join(ix_to_char[ix] for ix in sample_ix)

        print('----\n%s\n----' % (txt, ))



    # Get gradients for current model based on input and target sequences

    loss, grads, hprev = train(inputs, targets, hprev)

    smooth_loss = smooth_loss * 0.999 + loss * 0.001



    # Print loss information

    if n % 20000 == 0:

        print('iter %d, loss: %f, smooth loss: %f' % (n, loss, smooth_loss))



    # Update model with adagrad (stochastic) gradient descent

    beta = update(grads, beta)



    # Prepare for next iteration

    p += seq_length

    n += 1