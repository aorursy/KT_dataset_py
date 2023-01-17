import numpy as np

import torch
A = np.matrix([

    [0, 1, 0, 0],

    [0, 0, 1, 1], 

    [0, 1, 0, 0],

    [1, 0, 1, 0]])

X = np.matrix([

            [i, -i]

            for i in range(A.shape[0])

        ])
A*X
I = np.matrix(np.eye(A.shape[0]))

A_hat = A + I

A_hat * X
D = np.array(np.sum(A, axis=0))[0]

D = np.matrix(np.diag(D))

D**-1 * A
D**-1 * A * X
D_hat = np.array(np.sum(A_hat, axis=0))[0]

D_hat = np.matrix(np.diag(D_hat))

W = np.matrix([

             [1, -1],

             [-1, 1]

         ])

D_hat**-1 * A_hat * X 
W = np.matrix([

             [1],

             [-1]

         ])

D_hat**-1 * A_hat * X * W
torch.nn.ReLU()(torch.Tensor(D_hat**-1 * A_hat * X * W))