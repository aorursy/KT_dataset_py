import numpy as np # linear algebra

from utils import *

import random

import pprint

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = open('../input/dinosaur-island/dinos.txt', 'r').read()

data= data.lower()

chars = list(set(data))

data_size, vocab_size = len(data), len(chars)

print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
chars = sorted(chars)

print(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }

ix_to_char = { i:ch for i,ch in enumerate(chars) }

pp = pprint.PrettyPrinter(indent=4)

pp.pprint(ix_to_char)
def clip(gradients, maxValue):

    '''

    Clips the gradients' values between minimum and maximum.

    

    Arguments:

    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"

    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    

    Returns: 

    gradients -- a dictionary with the clipped gradients.

    '''

    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)

    for gradient in [dWax, dWaa, dWya, db, dby]:

        np.clip(gradient, -maxValue, maxValue, out=gradient)    

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients

# Test with a maxvalue of 10

mValue = 10

np.random.seed(3)

dWax = np.random.randn(5,3)*10

dWaa = np.random.randn(5,5)*10

dWya = np.random.randn(2,5)*10

db = np.random.randn(5,1)*10

dby = np.random.randn(2,1)*10

gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}

gradients = clip(gradients, mValue)

print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])

print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])

print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])

print("gradients[\"db\"][4] =", gradients["db"][4])

print("gradients[\"dby\"][1] =", gradients["dby"][1])
# Test with a maxValue of 5

mValue = 5

np.random.seed(3)

dWax = np.random.randn(5,3)*10

dWaa = np.random.randn(5,5)*10

dWya = np.random.randn(2,5)*10

db = np.random.randn(5,1)*10

dby = np.random.randn(2,1)*10

gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}

gradients = clip(gradients, mValue)

print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])

print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])

print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])

print("gradients[\"db\"][4] =", gradients["db"][4])

print("gradients[\"dby\"][1] =", gradients["dby"][1])

del mValue # avoid common issue
matrix1 = np.array([[1,1],[2,2],[3,3]]) # (3,2)

matrix2 = np.array([[0],[0],[0]]) # (3,1) 

vector1D = np.array([1,1]) # (2,) 

vector2D = np.array([[1],[1]]) # (2,1)

print("matrix1 \n", matrix1,"\n")

print("matrix2 \n", matrix2,"\n")

print("vector1D \n", vector1D,"\n")

print("vector2D \n", vector2D)
print("Multiply 2D and 1D arrays: result is a 1D array\n", 

      np.dot(matrix1,vector1D))

print("Multiply 2D and 2D arrays: result is a 2D array\n", 

      np.dot(matrix1,vector2D))
print("Adding (3 x 1) vector to a (3 x 1) vector is a (3 x 1) vector\n",

      "This is what we want here!\n", 

      np.dot(matrix1,vector2D) + matrix2)
print("Adding a (3,) vector to a (3 x 1) vector\n",

      "broadcasts the 1D array across the second dimension\n",

      "Not what we want here!\n",

      np.dot(matrix1,vector1D) + matrix2)
def sample(parameters, char_to_ix, seed):

    """

    Sample a sequence of characters according to a sequence of probability distributions output of the RNN



    Arguments:

    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 

    char_to_ix -- python dictionary mapping each character to an index.

    seed -- used for grading purposes. Do not worry about it.



    Returns:

    indices -- a list of length n containing the indices of the sampled characters.

    """

    # Retrieve parameters and relevant shapes from "parameters" dictionary

    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    vocab_size = by.shape[0]

    n_a = Waa.shape[1]

    # Step 1: Create the a zero vector x that can be used as the one-hot vector 

    # representing the first character (initializing the sequence generation). (≈1 line)

    x = np.zeros(( vocab_size, 1))

    # Step 1': Initialize a_prev as zeros (≈1 line)

    a_prev = np.zeros(( n_a, 1))

    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)

    indices = []

    # idx is the index of the one-hot vector x that is set to 1

    # All other positions in x are zero.

    # We will initialize idx to -1

    idx = -1 

    # Loop over time-steps t. At each time-step:

    # sample a character from a probability distribution 

    # and append its index (`idx`) to the list "indices". 

    # We'll stop if we reach 50 characters 

    # (which should be very unlikely with a well trained model).

    # Setting the maximum number of characters helps with debugging and prevents infinite loops. 

    counter = 0

    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):

        # Step 2: Forward propagate x using the equations (1), (2) and (3)

        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)

        z = np.dot(Wya, a) + by

        y = softmax(z)

        # for grading purposes

        np.random.seed(counter+seed) 

        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y

        # (see additional hints above)

        idx = np.random.choice(list(range(vocab_size)), p = y.ravel())

        # Append the index to "indices"

        indices.append(idx)

        # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.

        # (see additional hints above)

        x = np.zeros((vocab_size, 1))

        x[idx] = 1

        # Update "a_prev" to be "a"

        a_prev = a

        # for grading purposes

        seed += 1

        counter +=1

    if (counter == 50):

        indices.append(char_to_ix['\n'])

    return indices

np.random.seed(2)

_, n_a = 20, 100

Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)

b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)

parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

indices = sample(parameters, char_to_ix, 0)

print("Sampling:")

print("list of sampled indices:\n", indices)

print("list of sampled characters:\n", [ix_to_char[i] for i in indices])
def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):

    """

    Execute one step of the optimization to train the model.

    

    Arguments:

    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.

    Y -- list of integers, exactly the same as X but shifted one index to the left.

    a_prev -- previous hidden state.

    parameters -- python dictionary containing:

                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)

                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)

                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)

                        b --  Bias, numpy array of shape (n_a, 1)

                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    learning_rate -- learning rate for the model.

    

    Returns:

    loss -- value of the loss function (cross-entropy)

    gradients -- python dictionary containing:

                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)

                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)

                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)

                        db -- Gradients of bias vector, of shape (n_a, 1)

                        dby -- Gradients of output bias vector, of shape (n_y, 1)

    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)

    """

    # Forward propagate through time (≈1 line)

    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    # Backpropagate through time (≈1 line)

    gradients, a = rnn_backward(X, Y, parameters, cache)

    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)

    gradients = clip(gradients, maxValue=5)

    # Update parameters (≈1 line)

    parameters = update_parameters(parameters, gradients, learning_rate)    

    return loss, gradients, a[len(X)-1]

np.random.seed(1)

vocab_size, n_a = 27, 100

a_prev = np.random.randn(n_a, 1)

Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)

b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)

parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

X = [12,3,5,11,22,3]

Y = [4,14,11,22,25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)

print("Loss =", loss)

print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])

print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))

print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])

print("gradients[\"db\"][4] =", gradients["db"][4])

print("gradients[\"dby\"][1] =", gradients["dby"][1])

print("a_last[4] =", a_last[4])
def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27, verbose = False):

    """

    Trains the model and generates dinosaur names. 

    

    Arguments:

    data -- text corpus

    ix_to_char -- dictionary that maps the index to a character

    char_to_ix -- dictionary that maps a character to an index

    num_iterations -- number of iterations to train the model for

    n_a -- number of units of the RNN cell

    dino_names -- number of dinosaur names you want to sample at each iteration. 

    vocab_size -- number of unique characters found in the text (size of the vocabulary)

    

    Returns:

    parameters -- learned parameters

    """

    # Retrieve n_x and n_y from vocab_size

    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters

    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss)

    loss = get_initial_loss(vocab_size, dino_names)

    # Build list of all dinosaur names (training examples).

    with open("../input/dinosaur-island/dinos.txt") as f:

        examples = f.readlines()

    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all dinosaur names

    np.random.seed(0)

    np.random.shuffle(examples)

    # Initialize the hidden state of your LSTM

    a_prev = np.zeros((n_a, 1))

    # Optimization loop

    for j in range(num_iterations):

        # Set the index `idx` (see instructions above)

        idx = j % len(examples)

        # Set the input X (see instructions above)

        single_example = examples[idx] 

        single_example_chars = [c for c in single_example]

        single_example_ix = [char_to_ix[c] for c in single_example_chars]

        X = [None]+ [single_example_ix]

        X = [None] + [char_to_ix[ch] for ch in examples[idx]]; 

        # Set the labels Y (see instructions above)

        ix_newline = char_to_ix["\n"]

        Y = X[1:]+[ix_newline]

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters

        # Choose a learning rate of 0.01

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)

        # debug statements to aid in correctly forming X, Y

        if verbose and j in [0, len(examples) -1, len(examples)]:

            print("j = " , j, "idx = ", idx,) 

        if verbose and j in [0]:

            print("single_example =", single_example)

            print("single_example_chars", single_example_chars)

            print("single_example_ix", single_example_ix)

            print(" X = ", X, "\n", "Y =       ", Y, "\n")

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.

        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly

        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print

            seed = 0

            for name in range(dino_names):

                # Sample indices and print them

                sampled_indices = sample(parameters, char_to_ix, seed)

                print_sample(sampled_indices, ix_to_char)

                seed += 1  # To get the same result (for grading purposes), increment the seed by one. 

            print('\n')

    return parameters

parameters = model(data, ix_to_char, char_to_ix, verbose = True)