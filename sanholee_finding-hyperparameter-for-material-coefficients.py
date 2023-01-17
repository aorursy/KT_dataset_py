import numpy as np

import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
# objective target...

def tensile_strength(c,n,x):

    return c*(x)**n
tensile_test_x = np.linspace(0,0.1, 100)

tensile_test_y = tensile_strength(c=123, n=0.4, x=tensile_test_x)
plt.plot(tensile_test_x, tensile_test_y)
# make objective funtion returning subtraction with original funtion above.

# It should be minimize and it has two unknown parameter.

# It returns a sqrt value for subtracted values from test data.

def target_function(C, n):

    x = np.linspace(0,0.1, 100)

    target_y = tensile_strength(c=123, n=0.4, x=x)

    

    y = tensile_strength(c=C,n=n,x=x)

    

    squaring_of_subtractionY = np.square(target_y - y)

    sqrtY = np.sqrt(np.sum(squaring_of_subtractionY))

    

    return -sqrtY



# For using maxmize method, need to multiply minus 1 for returning value.
# initial bounds setting.

pbounds = {

    "C":(50,400),

    "n":(0.1,0.90)

}
# make optimizaer instance

optimizer = BayesianOptimization(

    f=target_function,

    pbounds=pbounds,

    verbose=2,

    random_state=1

)
optimizer.maximize(

    init_points=3,

    n_iter=100

)
optimizer.max
C = optimizer.max["params"]["C"]

n = optimizer.max["params"]["n"]



plt.scatter(tensile_test_x,tensile_test_y, c='r', label="TEST DATA, C : 123, n : 0.4")

plt.plot(tensile_test_x, tensile_strength(c=C,n=n,x=tensile_test_x), label="Searched by BO, C : {:5.3f}, n : {:5.3f} ".format(C,n))

plt.legend()

# 초기에 설정한 함수와 근사한 계수를 탐색해보라고 지도해본다.

optimizer.probe(

    params={"C":120,"n":0.35},

    lazy=True

)
optimizer.maximize(

    init_points=0,

    n_iter=10

)
C = optimizer.max["params"]["C"]

n = optimizer.max["params"]["n"]



plt.scatter(tensile_test_x,tensile_test_y, c='r', label="TEST DATA, C : 123, n : 0.4")

plt.plot(tensile_test_x, tensile_strength(c=C,n=n,x=tensile_test_x), label="Searched by BO, C : {:5.3f}, n : {:5.3f} ".format(C,n))

plt.legend()