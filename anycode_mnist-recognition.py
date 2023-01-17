!pip install ultimate==2.1.2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ultimate.mlp import MLP

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
        (14, 14, 4),
        (7, 7, 8),
        (1, 1, 10),
    ],
    'exf': [
        {'Kernel':[6,6], 'Stride':[2,2], 'Pad':[2,2]},
        {'Kernel':[6,6], 'Stride':[2,2], 'Pad':[2,2]},
        {'Kernel':[7,7], 'Stride':[1,1], 'Pad':[0,0]},
    ],
    
    'loss_type': 'hardmax',
    'output_range': [-1, 1],
    'output_shrink': 0.001,
    'regularization': 1,
    'op': 'conv',

    'importance_mul': 0.0001,
    'importance_out': True,
    'verbose': 1,
    'rate_init': 0.01, 
    'rate_decay': 0.8, 
    'epoch_train': 1 * 10, 
    'epoch_decay': 1,
}

mlp = MLP(param).fit(X_train, Y_train)

feature_importance = mlp.feature_importances_
    
print(feature_importance.shape, feature_importance.min(), feature_importance.max())

feature_importance = (feature_importance - feature_importance.min()) / (feature_importance.max() - feature_importance.min()) 
feature_importance = feature_importance.reshape(28, 28)
print(feature_importance.shape, feature_importance.min(), feature_importance.max())

import matplotlib.pyplot as plt
plt.imshow(feature_importance, cmap='gray')
plt.show()

pred = mlp.predict(X_test)

pd.DataFrame({"ImageId": list(range(1,len(pred)+1)), "Label": pred}).to_csv("pred.csv", index=False, header=True)
