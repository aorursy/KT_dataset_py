import numpy as np

#Grafik çizdirme kütüphanesi

import matplotlib.pyplot as plt



import os #Sistem 

import warnings #uyarılar

#print(os.listdir("../input/"))

warnings.filterwarnings("ignore")


# The training data.

X = np.array([

    [1, 1],

    [0, 1],

    [1, 0],

    [0, 0]

])



# The labels for the training data.

y = np.array([

    [0],

    [1],

    [1],

    [0]

])
W13 = 0.5

W14 = 0.9

W23 = 0.4

W24 = 1.0

W35 = -1.2

W45 = 1.1

Q3 = 0.8

Q4 = -0.1

Q5 = 0.3

learning_rate = 0.1
def sigmoid(x):

    return 1/(1 + np.exp(-x)) 
z = X[0][0]*W13 + X[0][1]*W23-Q3

z2 = X[0][0]*W14 + X[0][1]*W24-Q4



Y3 = sigmoid(z)

Y4 = sigmoid(z2)
z_next = Y3*W35 + Y4*W45 - Q5



Y5 = sigmoid(z_next)
e = y[0]-Y5

print(e)
G5 = Y5*(1-Y5)*e

print(G5)
delta_W35 = learning_rate*Y3*G5

delta_W45 = learning_rate*Y4*G5

delta_Q5 = learning_rate*Q4*G5



G3 = Y3*(1-Y3)*G5*W35

G4 = Y4*(1-Y4)*G5*W45
delta_W13 = learning_rate*X[0][0]*G3

delta_W23 = learning_rate*X[0][1]*G3

delta_Q3 = learning_rate*(-1)*G3

delta_W14 = learning_rate*X[0][0]*G4

delta_W24 = learning_rate*X[0][1]*G4

delta_Q4 = learning_rate*(-1)*G4
W13 = W13 + delta_W13

W14 = W14 + delta_W14

W23 = W23 + delta_W23

W24 = W24 + delta_W24

W35 = W35 + delta_W35

W45 = W45 + delta_W45

Q5 = Q5 + delta_Q5

Q4 = Q4 + delta_Q4

Q3 = Q3 + delta_Q3
print("W13",W13)

print("W14",W14)

print("W23",W23)

print("W24",W24)

print("W35",W35)

print("W45",W45)

print("Q5",Q5)

print("Q4",Q4)

print("Q3",Q3)
def next_fit(x1,x2,y,pr:str):

    W13 = 0.5

    W14 = 0.9

    W23 = 0.4

    W24 = 1.0

    W35 = -1.2

    W45 = 1.1

    Q3 = 0.8

    Q4 = -0.1

    Q5 = 0.3

    learning_rate = 0.1

    

    e_count = []

    i_count = []

    

    for i in range(500):

        z = x1*W13 + x2*W23-Q3

        z2 = x1*W14 + x2*W24-Q4

        

        Y3 = sigmoid(z)

        Y4 = sigmoid(z2)

        

        z_next = Y3*W35 + Y4*W45 - Q5

    

        Y5 = sigmoid(z_next)

        

        e = y-Y5

        if i%20 == 0:

            print("error",e)

        e_count.append(e)

        i_count.append(i)

        

        G5 = Y5*(1-Y5)*e

        

        delta_W35 = learning_rate*Y3*G5

        delta_W45 = learning_rate*Y4*G5

        delta_Q5 = learning_rate*Q4*G5

        

        G3 = Y3*(1-Y3)*G5*W35

        G4 = Y4*(1-Y4)*G5*W45

        

        

        delta_W13 = learning_rate*x1*G3

        delta_W23 = learning_rate*x2*G3

        delta_Q3 = learning_rate*(-1)*G3

        delta_W14 = learning_rate*x1*G4

        delta_W24 = learning_rate*x2*G4

        delta_Q4 = learning_rate*(-1)*G4

        

        W13 = W13 + delta_W13

        W14 = W14 + delta_W14

        W23 = W23 + delta_W23

        W24 = W24 + delta_W24

        W35 = W35 + delta_W35

        W45 = W45 + delta_W45

        Q5 = Q5 + delta_Q5

        Q4 = Q4 + delta_Q4

        Q3 = Q3 + delta_Q3

        

#        print("--------------  ",str(pr),"  --------------")

#        print("W13",W13)

#        print("W14",W14)

#        print("W23",W23)

#        print("W24",W24)

#        print("W35",W35)

#        print("W45",W45)

#        print("Q5",Q5)

#        print("Q4",Q4)

#        print("Q3",Q3)

    plt.xlabel("iteration")

    plt.ylabel("error")

    plt.scatter(i_count,e_count);

    

    return Y5,e

    

one_input_Y5, one_input_error = next_fit(X[0][0],X[0][1],y[0],"0")

#two_input_Y5, two_input_error = next_fit(X[1][0],X[1][1],y[1],"1")

#three_input_Y5, three_input_error = next_fit(X[2][0],X[2][1],y[2],"2")

#four_input_Y5, four_input_error = next_fit(X[3][0],X[3][1],y[3],"3")
two_input_Y5, two_input_error = next_fit(X[1][0],X[1][1],y[1],"1")
three_input_Y5, three_input_error = next_fit(X[2][0],X[2][1],y[2],"2")
four_input_Y5, four_input_error = next_fit(X[3][0],X[3][1],y[3],"3")
print("X1 = ",str(X[0][0])," X2 = ",str(X[0][1])," Y_actual_output = ",str(y[0])," Y5 = ",str(one_input_Y5)," error = ",str(one_input_error))

print("X1 = ",str(X[1][0])," X2 = ",str(X[1][1])," Y_actual_output = ",str(y[1])," Y5 = ",str(two_input_Y5)," error = ",str(two_input_error))

print("X1 = ",str(X[2][0])," X2 = ",str(X[2][1])," Y_actual_output = ",str(y[2])," Y5 = ",str(three_input_Y5)," error = ",str(three_input_error))

print("X1 = ",str(X[3][0])," X2 = ",str(X[3][1])," Y_actual_output = ",str(y[3])," Y5 = ",str(four_input_Y5)," error = ",str(four_input_error))
summ_error = 0



summ_error = summ_error + one_input_error**2

summ_error = summ_error + two_input_error**2

summ_error = summ_error + three_input_error**2

summ_error = summ_error + four_input_error**2



summ_error = summ_error / 4

print(summ_error)