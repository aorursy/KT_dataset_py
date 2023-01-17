from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

import numpy as np



# Create a dataset for the XOR function

x = np.array([

    [0,0],

    [1,0],

    [0,1],

    [1,1]

])



y = np.array([

    0,

    1,

    1,

    0

])



# Build the network

# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



done = False

cycle = 1



while not done:

    print("Cycle #{}".format(cycle))

    cycle+=1

    model = Sequential()

    model.add(Dense(2, input_dim=2, activation='relu')) 

    model.add(Dense(1)) 

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x,y,verbose=0,epochs=10000)



    # Predict

    pred = model.predict(x)

    

    # Check if successful.  It takes several runs with this small of a network

    done = pred[0]<0.01 and pred[3]<0.01 and pred[1] > 0.9 and pred[2] > 0.9 

    print(pred)
# Dump weights

for layerNum, layer in enumerate(model.layers):

    weights = layer.get_weights()[0]

    biases = layer.get_weights()[1]

    

    for toNeuronNum, bias in enumerate(biases):

        print(f'{layerNum}B -> L{layerNum+1}N{toNeuronNum}: {bias}')

    

    for fromNeuronNum, wgt in enumerate(weights):

        for toNeuronNum, wgt2 in enumerate(wgt):

            print(f'L{layerNum}N{fromNeuronNum} -> L{layerNum+1}N{toNeuronNum} = {wgt2}')
input0 = 0

input1 = 1



hidden0Sum = (input0*1.3)+(input1*1.3)+(-1.3)

hidden1Sum = (input0*1.2)+(input1*1.2)+(0)



print(hidden0Sum) # 0

print(hidden1Sum) # 1.2



hidden0 = max(0,hidden0Sum)

hidden1 = max(0,hidden1Sum)



print(hidden0) # 0

print(hidden1) # 1.2



outputSum = (hidden0*-1.6)+(hidden1*0.8)+(0)

print(outputSum) # 0.96



output = max(0,outputSum)



print(output) # 0.96