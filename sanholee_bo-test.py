from bayes_opt import BayesianOptimization
def black_box_function(x,y):

    return -x**2 - (y-1)**2 + 1
# 파라미터 범위를 지정한다

pbound = {

    "x":(2,4),

    "y":(-3,3)

}
# optimizer 객체를 만들어줌.

optimizer = BayesianOptimization(

    f=black_box_function,

    pbounds = pbound,

    verbose = 2,

    random_state = 1

)
optimizer.maximize(

    init_points=2,

    n_iter=3

)
print(optimizer.max)
for i, res in enumerate(optimizer.res):

    print("Iteration {} : \n\t {}".format(i, res))    
optimizer.set_bounds(new_bounds={

    "x":(-2,3)

})
optimizer.maximize(

    init_points=0,

    n_iter=5

)
optimizer.probe(

    params={"x":0.5,"y":0.7},

    lazy=True

)
optimizer.space.keys
optimizer.probe(

    params=[-0.3,0.1],

    lazy=True

)
optimizer.maximize(

    init_points=0,

    n_iter=0

)
from bayes_opt.logger import JSONLogger

from bayes_opt.event import Events
logger = JSONLogger(path="./logs.json")

optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
optimizer.maximize(

    init_points=2,

    n_iter=3

)
from bayes_opt.util import load_logs
# making new optimizer

new_optimizer = BayesianOptimization(

    f=black_box_function,

    pbounds={"x":(-2,2),"y":(-2,2)},

    verbose=2,

    random_state=7

)

print(len(new_optimizer.space))
load_logs(new_optimizer, logs=["./logs.json"])
print("New optimizer is now aware of {} points".format(len(new_optimizer.space)))
for i, res in enumerate(new_optimizer.res):

    print("Iteration {} : \n\t {}".format(i, res))  
new_optimizer.max
new_optimizer.maximize(

    init_points=0,

    n_iter=10

)