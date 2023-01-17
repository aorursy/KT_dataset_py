from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import plot_model
import numpy as np



X = np.array([

    [0,1,-2,-1,10],

    [1,3,1,4,2],

    [3,-1,-1,2,3],

    [2,0,2,1,4],

    [4,-3,0,-3,5]

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
"""

演習:epochsとbatch_sizeを変更してみてください

"""

history = model.fit(X, y, epochs=20, batch_size=5)
from sklearn.metrics import accuracy_score



y_pred = model.predict(X, batch_size=5)

print (y_pred)



print ('Accuracy:', accuracy_score(y,np.round(y_pred)))