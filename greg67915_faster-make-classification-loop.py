from sklearn.datasets import make_classification 

import numpy as np

import pandas as pd
def gen_data_0():

    # generate dataset 

    train, target = make_classification(512, 255, n_informative=np.random.randint(33, 47), n_redundant=0, flip_y=0.08)

    train = np.hstack((train, np.ones((len(train), 1))*0))



    for i in range(1, 512):

        X, y = make_classification(512, 255, n_informative=np.random.randint(33, 47), n_redundant=0, flip_y=0.08)

        X = np.hstack((X, np.ones((len(X), 1))*i))

        train = np.vstack((train, X))

        target = np.concatenate((target, y))

    return train, target
%%time

train0, target0 = gen_data_0()
def gen_data(N=512, M=255):

    train = np.zeros((N**2, M + 1,), dtype=np.float)

    target = np.zeros((N**2,), dtype=np.float)

    for i in range(N):

        X, y = make_classification(N, M, n_informative=np.random.randint(33, 47), n_redundant=0, flip_y=0.08)

        X = np.hstack([X, i * np.ones((N, 1,))])



        start, stop = i * N, (i + 1) * N

        train[start: stop] = X

        target[start: stop] = y

    return train, target
%%time

train, target = gen_data()
np.random.seed(2019)

train0, target0 = gen_data_0()



np.random.seed(2019)

train, target = gen_data()
np.allclose(train, train0) and np.allclose(target, target0)