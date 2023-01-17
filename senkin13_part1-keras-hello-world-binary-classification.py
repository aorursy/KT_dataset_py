from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import plot_model
import numpy as np



X = np.array([

    [0,1,-2,-1],

    [1,3,1,4],

    [3,-1,-1,2],

    [2,0,2,1],

    [4,-3,0,-3]

]    

)



y = np.array([

    [False],

    [True],

    [False],

    [True],

    [False]

]    

)



print ('X:',X.shape)

print ('y:',y.shape)
model = Sequential()
"""

演習:unitsを変更してみてください

"""

model.add(Dense(units=8,input_dim=X.shape[1]))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',  metrics=['accuracy'])
model.summary()
plot_model(model, to_file='helloworld.png')