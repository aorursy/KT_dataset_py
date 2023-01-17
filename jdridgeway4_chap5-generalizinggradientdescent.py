def weighted_sum(a,b):

    assert(len(a) == len(b))

    output = 0

    for i in range(len(a)):

        output += (a[i] * b[i])

        

    return output



def neural_network(input_val, weights):

    pred = weighted_sum(input_val, weights)

    return pred





def ele_mul(number, vector):

    output = [0,0,0]

    assert(len(output) == len(vector))

    for i in range(len(vector)):

        output[i] = number * vector[i]

    

    return output





#input variables

toes = [8.5, 9.5, 9.9, 9.0]

wlrec = [0.65, 0.8, 0.8, 0.9]

nfans = [1.2, 1.3, 0.5, 1.0]



win_or_lose_binary = [1,1,0,1]

true = win_or_lose_binary[0]

alpha = 0.01

weights = [0.1, 0.2, -.1]

input_val = [toes[0], wlrec[0], nfans[0]]



for i in range(3):

    # do you prediction

    pred = neural_network(input_val, weights)

    error = (pred - true) ** 2

    delta = pred - true

    weight_deltas = ele_mul(delta, input_val)

    

    print("Iteration: {}".format(i+1))

    print("Pred: {}".format(pred))

    print("Error: {}".format(error))

    print("Delta: {}".format(delta))

    print("Weights: {}".format(weights))

    print("Weight_Deltas: {}".format(weight_deltas))

    print("Prediction: {}".format(pred))

    print()

    

    # see page 247 (in the ebook version)

    # after you calculate the `weight_deltas` for each of the input values

    # you're saying here

    # "I want you to modify each of the original weight values to be the that specific weight_delta

    # (for that input node) multiplied by the alpha (so we don't overshoot, think about the parabola)

    # and substract by the original weight"

    # then these new weights will be used in making the next prediction

    for k in range(len(weights)):

        weights[k] -= alpha * weight_deltas[k]
# This code is the same as above except that you are freezing(making the first weight_delta 0) and incresing

# the alpha to 0.3



#input variables

toes = [8.5, 9.5, 9.9, 9.0]

wlrec = [0.65, 0.8, 0.8, 0.9]

nfans = [1.2, 1.3, 0.5, 1.0]



win_or_lose_binary = [1,1,0,1]

true = win_or_lose_binary[0]

alpha = 0.3

weights = [0.1, 0.2, -.1]

input_val = [toes[0], wlrec[0], nfans[0]]



for i in range(3):

    # do you prediction

    pred = neural_network(input_val, weights)

    error = (pred - true) ** 2

    delta = pred - true

    weight_deltas = ele_mul(delta, input_val)

    #freezing weight_delta[0]

    weight_deltas[0] = 0 

    

    print("Iteration: {}".format(i+1))

    print("Pred: {}".format(pred))

    print("Error: {}".format(error))

    print("Delta: {}".format(delta))

    print("Weights: {}".format(weights))

    print("Weight_Deltas: {}".format(weight_deltas))

    print("Prediction: {}".format(pred))

    print()

    

    for k in range(len(weights)):

        weights[k] -= alpha * weight_deltas[k]
def neural_network(input_val, weights):

    pred = ele_mul(input_val, weights)

    return pred





def ele_mul(number, vector):

    output = [0,0,0]

    assert(len(output) == len(vector))

    for i in range(len(vector)):

        output[i] = number * vector[i]

    

    return output



weights = [0.3, 0.2, 0.9]

wlrec = [0.65, 1.0, 1.0, 0.9]

hurt = [0.1, 0.0, 0.0, 0.1]

win = [1, 1, 0, 1]

sad = [0.1, 0.0, 0.1, 0.2]

input_val3 = wlrec[0]

true = [hurt[0], win[0], sad[0]]

# the original alpha in the book is alpha = 0.1 

# I increased it to increase the learning rate to the max of 1 with fewer iterations for the predictions to be

# close to the true values

alpha = 1

error = [0,0,0]

delta = [0,0,0]



# I added this for loop to see how many iterations it would take

for k in range(10):

    pred = neural_network(input_val3, weights)



    for i in range(len(true)):

        error[i] = (pred[i] - true[i]) ** 2

        delta[i] = pred[i] - true[i]



    weight_deltas = ele_mul(input_val3, delta)



    for i in range(len(weights)):

        weights[i] -= (weight_deltas[i] * alpha)



    print("Step: {}".format(k))

    print("Weights: {}".format(weights))

    print("Weight Deltas: {}".format(weight_deltas))

    print("Prediction: {}".format(pred))

    print()
import numpy as np



def weighted_sum(a,b):

    assert(len(a) == len(b))

    output = 0

    for i in range(len(a)):

        output += (a[i] * b[i])

        

    return output



def neural_network(input_val, weights):

    pred = vect_mat_mul(input_val, weights)

    return pred





def ele_mul(number, vector):

    output = [0,0,0]

    assert(len(output) == len(vector))

    for i in range(len(vector)):

        output[i] = number * vector[i]

    

    return output



def vect_mat_mul(vect, matrix):

    assert(len(vect) == len(matrix))

    output = [0,0,0]

    for i in range(len(vect)):

        output[i] = weighted_sum(vect, matrix[i])

    return output





#this code for this function was copied from this link

# https://integratedmlai.com/basic-linear-algebra-tools-in-pure-python-without-numpy-or-scipy/

# I don't know why Andrew didn't include this in ebook version

def zeros_matrix(rows, cols):

    """

    Creates a matrix filled with zeros.

        :param rows: the number of rows the matrix should have

        :param cols: the number of columns the matrix should have

 

        :return: list of lists that form the matrix

    """

    M = []

    while len(M) < rows:

        M.append([])

        while len(M[-1]) < cols:

            M[-1].append(0.0)

 

    return M



def outer_prod(vec_a, vec_b):

    out = zeros_matrix(len(vec_a), len(vec_b))

    

    for i in range(len(vec_a)):

        for j in range(len(vec_b)):

            out[i][j] = vec_a[i] * vec_b[j]

    

    return out



#defining variables

toes = [8.5, 9.5, 9.9, 9.0]

wlrec = [0.65, 0.8, 0.8, 0.9]

nfans = [1.2, 1.3, 0.5, 1.0]



hurt = [0.1, 0.0, 0.0, 0.1]

win = [1, 1, 0, 1]

sad = [0.1, 0.0, 0.1, 0.2]



alpha = 0.7



input_val4 = [toes[0], wlrec[0], nfans[0]]

true = [hurt[0], win[0], sad[0]]

weights = [[0.1, 0.1, -0.3],

          [0.1, 0.2, 0.0],

          [0.0, 1.3, 0.1]]

error =[0,0,0]

delta = [0,0,0]



for iteration in range(3):

    pred = neural_network(input_val4, weights)

    

    for i in range(len(true)):

        error[i] = (pred[i] - true[i]) ** 2

        delta[i] = pred[i] - true[i] 

        

    weight_deltas = outer_prod(input_val4, delta)

    

    for new_i in range(len(weights)):

        for j in range(len(weights[0])):

            weights[new_i][j] -= alpha * weight_deltas[new_i][j]

            

    print("Step: {}".format(iteration))

    print("Weights: {}".format(weights))

    print("Weight Deltas: {}".format(weight_deltas))

    print("Prediction: {}".format(pred))

    print()

    
