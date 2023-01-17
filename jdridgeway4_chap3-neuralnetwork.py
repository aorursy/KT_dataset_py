weight = 0.1
def neural_network(input_val, weight):

    # you could also picture this like the formula

    # y       =   m     *    x    

    prediction = weight * input_val

    return prediction
number_of_toes = [8.5, 9.5, 10, 9]

input_val = number_of_toes[0]

pred = neural_network(input_val, weight)

print(pred)
def elementwise_multiplication(vec_a, vec_b):

    # multiplying the values in two different vectors (assuming they have the same length)

    

    if (len(vec_a) == len(vec_b)):

        final_vector = [a*b for a,b in zip(vec_a, vec_b)]

    else:

        final_vector = 0

    

    return final_vector
def elementwise_addition(vec_a, vec_b):

    # adding the values in two different vectors (assuming they have the same length)

    

    if (len(vec_a) == len(vec_b)):

        final_vector = [a+b for a,b in zip(vec_a, vec_b)]

    else:

        final_vector = 0

        

    return final_vector
def vector_sum(vec_a):

    # the summation of all the values in a vector

    

    return sum(vec_a)
def vector_average(vec_a):

    # the average of all the values in a vector

    

    return sum(vec_a)/len(vec_a)
def dot_product(vec_a, vec_b):

    # the dot product is the summation of multiplying two or more vectors

    

    final_vector = elementwise_multiplication(vec_a, vec_b)

    return vector_sum(final_vector)
# practice

a = [0,1,0,1]

b = [1,0,1,0]

c = [0,1,1,0]

d = [.5,0,.5,0]

e = [0,1,-1,0]



[dot_product(a,b), dot_product(b,c), dot_product(b,d), dot_product(c,c), dot_product(d,d), dot_product(c,e)]
weights = [0.1, 0.2, 0]
def neural_network2(input_val, weights):

    

    pred = dot_product(input_val, weights)

    return pred
# each index in the vectors below corresponds to one game (4 games in total)

# i.e. toes[0], wlrec[0], nfans[0] => 1st game

# toes[1], wlrec[1], nfans[1] => 2nd game

# etc.



toes = [8.5, 9.5, 9.9, 9.0]

wlrec = [0.65, 0.8, 0.8, 0.9]

nfans = [1.2, 1.3, 0.5, 1.0]



input_val = [toes[0], wlrec[0], nfans[0]]

pred = neural_network2(input_val, weights)

print(pred)
def neural_network_numpy(input_val, weights):

    pred = input_val.dot(weights)

    return pred
import numpy as np



weights3 = np.array([0.1, 0.2, 0])



toes = np.array([8.5, 9.5, 9.9, 9.0])

wlrec = np.array([0.65, 0.8, 0.8, 0.9])

nfans = np.array([1.2, 1.3, 0.5, 1.0])



#we are just predicting the first game

input_val3 = np.array([toes[0], wlrec[0], nfans[0]])

pred = neural_network_numpy(input_val3, weights3)

print(pred)
def vector_mat_mul(vect, matrix):

    assert(len(vect) == len(matrix))

    output = [0,0,0]

    

    for i in range(len(vect)):

        output[i] = dot_product(vect, matrix[i])

        

    return output



def neural_network(input_val, weights):

    pred = vector_mat_mul(input_val, weights)

    return pred



            #toes wins #fans

weights = [[0.1, 0.1, -0.3], #hurt?

          [0.1, 0.2, 0.0], #win?

          [0.0, 1.3, 0.1]] #sad?



toes = [8.5, 9.5, 9.9, 9.0]

wlrec = [0.65, 0.8, 0.8, 0.9]

nfans = [1.2, 1.3, 0.5, 1.0]



#input_values are just the values of one game

input_values = [toes[0], wlrec[0], nfans[0]]

pred = neural_network(input_values, weights)

print(pred)
def neural_network(input_val, weights):

    hid = vector_mat_mul(input_val, weights[0])

    pred = vector_mat_mul(hid, weights[1])

    return pred



ih_wgt = [[0.1, 0.2, -0.1],

         [-0.1, 0.1, 0.9],

         [0.1, 0.4, 0.1]]



hp_wgt = [[0.3, 1.1, -0.3],

         [0.1, 0.2, 0.0],

         [0.0, 1.3, 0.1]]



weights = [ih_wgt, hp_wgt]



toes = [8.5, 9.5, 9.9, 9.0]

wlrec = [0.65, 0.8, 0.8, 0.9]

nfans = [1.2, 1.3, 0.5, 1.0]



input_val = [toes[0], wlrec[0], nfans[0]]

pred = neural_network(input_val, weights)

print(pred)
import numpy as np



def neural_network_stacked(i_val, weights):

    hid = i_val.dot(weights[0])

    pred = hid.dot(weights[1])

    return pred



ih_wgt = np.array([[0.1, 0.2, -0.1],

         [-0.1, 0.1, 0.9],

         [0.1, 0.4, 0.1]]).T



hp_wgt = np.array([[0.3, 1.1, -0.3],

         [0.1, 0.2, 0.0],

         [0.0, 1.3, 0.1]]).T



weights2 = [ih_wgt, hp_wgt]



toes = np.array([8.5, 9.5, 9.9, 9.0])

wlrec = np.array([0.65, 0.8, 0.8, 0.9])

nfans = np.array([1.2, 1.3, 0.5, 1.0])



i_val = np.array([toes[0], wlrec[0], nfans[0]])

pred = neural_network_stacked(i_val, weights2)

print(pred)