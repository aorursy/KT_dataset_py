!git clone https://gitlab.com/pw_neural_nets/numpy_ann.git

!mv numpy_ann/* .
import numpy as np

import pandas as pd



from mlp import MLP
train_data = pd.read_csv('../input/digit-recognizer/train.csv')

X_train = np.array(train_data.iloc[:,1:].T).reshape(784, -1)

Y_train = np.array(train_data.iloc[:,0]).reshape(1, -1)
X_train.shape
test_data = pd.read_csv('../input/digit-recognizer/test.csv')

X_test = np.array(test_data.T).reshape(784, -1)
# number of layers

# number of neurons

# activation function



mlp = MLP()

mlp.add_layer(784, activation="sigmoid")

mlp.add_layer(784, activation="sigmoid")
# bias

# batch

# iterations

# learning rate

# momentum

# mode

# epochs



mlp.train(

    X=X_train,

    Y=Y_train,

    bias=True,  

    batch_size=64,

    learning_rate=0.005,

    momentum_rate=0.8,

    mode="multiple_classification",

    loss="categorical_crossentropy",

    epochs=100,

    verbose=True,

    early_stopping=True,

    patience=3,

    validation_split=0.7,

)
mlp.load_best_parameters()
Y_test_hat = mlp.predict(X_test, np.zeros((1, X_test.shape[1])))
Y_test_hat
subm_dict = {

    "ImageId": list(range(1, len(Y_test_hat[0])+1)),

    "Label": Y_test_hat[0]

}
df = pd.DataFrame(subm_dict)
df.to_csv("subm.csv", index=False)