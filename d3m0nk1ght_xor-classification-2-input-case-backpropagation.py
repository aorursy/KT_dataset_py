import numpy as np



def nonlin(x):

     return 1/(1+np.exp(-x))

    

def deriv(x):

     return x*(1-x)

    

X = np.array([[0,0],

            [0,1],

            [1,0],

            [1,1]])

                

y = np.array([[0],

			[1],

			[1],

			[0]])



np.random.seed(1)



# randomly initialize our weights with mean 0

syn0 = 2*np.random.random((2,4)) - 1

syn1 = 2*np.random.random((4,1)) - 1





def forward(x,w0,w1):

	# Feed forward through layers 0, 1, and 2

    l0 = x

    l1 = nonlin(np.dot(l0,w0))

    #bias = np.ones((len(l1),1))

    #l1=np.concatenate((bias,l1),axis=1)

    l2 = nonlin(np.dot(l1,w1))

    return l1,l2

    #bias = np.ones((slen(l2),1))

    

def backprop(Y,l0,l1,l2,w0,w1):

    #l2=np.concatenate((bias,l2),axis=1)

    # how much did we miss the target value?

    l2_error = Y - l2

    # in what direction is the target value?

    # were we really sure? if so, don't change too much.

    l2_delta = l2_error*deriv(l2)

    # how much did each l1 value contribute to the l2 error (according to the weights)?

    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?

    # were we really sure? if so, don't change too much.

    l1_delta = l1_error * deriv(l1)

    w1 += l1.T.dot(l2_delta)

    w0 += l0.T.dot(l1_delta)

    return l2_error



for j in range(60000):

       layer1,layer2=forward(X,syn0,syn1)

       l2e=backprop(y,X,layer1,layer2,syn0,syn1)

       if (j% 10000) == 0:

        print("Error:" + str(np.mean(np.abs(l2e))))

o=[]

for l in layer2:

    if(float(l)>0.5):

        o.append("1")

    elif(float(l)<0.5):

        o.append("0")

    else:

        o.append(l)

o=np.array(o)

print("Inputs:"+str(X))

print("Desired outputs:"+str(y))

print("Predicted outputs:"+str(o))



        



    

    
