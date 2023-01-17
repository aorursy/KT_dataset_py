!pip install ultimate==2.48.1
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ultimate.mlp import MLP
from ultimate.plotting import plot_image

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 
X_test = np.array(test, np.float64)

X_train = np.array(X_train, np.float64)
Y_train = np.array(Y_train, np.float64)
X_train = (X_train / 255.0) * 2 - 1 
X_test = (X_test / 255.0) * 2 - 1

print("X_train", X_train.shape, X_train.min(), X_train.max())
print("Y_train", Y_train.shape, Y_train.min(), Y_train.max())
print("X_test", X_test.shape, X_test.min(), X_test.max())

param = {
    'layer_size': [
        (28, 28, 1),
        (28, 28, 4),
        (28, 28, 4),
        (14, 14, 8),
        (14, 14, 8),
        (7, 7, 16),
        (1, 1, 80),
        (1, 1, 10),
    ],
    'exf': [
        {'Kernel':[3,3], 'Stride':[1,1], 'Pad':[1,1], 'Turbulence':[1, 1]},
        {'Op': 'star', 'Constellation': 'aries'},
        {'Kernel':[6,6], 'Stride':[2,2], 'Pad':[2,2], 'Turbulence':[0, 0]},
        {'Op': 'star', 'Constellation': 'aries'},
        {'Kernel':[6,6], 'Stride':[2,2], 'Pad':[2,2], 'Turbulence':[0, 0]},
        {'Op': 'fc'},
        {'Op': 'fc'},
    ],
    
    'rate_init': 0.01,
    'loss_type': 'hardmax',
    
    'output_range': [-1, 1],
    'output_shrink': 0.001,
    'regularization': 1,
    'op': 'conv',

    'importance_mul': 0.0001,
    'importance_out': True,
    'verbose': 1,
    
    'rate_decay': 0.8, 
    'epoch_train': 4 * 10, 
    'epoch_decay': 4,
    
    'iteration_log': 10000,
}

mlp = MLP(param).fit(X_train, Y_train)

plot_image(mlp.feature_importances_.reshape(28, 28))

pred = mlp.predict(X_test)

pd.DataFrame({"ImageId": list(range(1,len(pred)+1)), "Label": pred}).to_csv("pred.csv", index=False, header=True)