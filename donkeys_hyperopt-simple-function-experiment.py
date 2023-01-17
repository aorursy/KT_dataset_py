import random



history_v1 = []

history_v2 = []

history_v3 = []



# define an objective function

# normally, you would put something like a machine learning algorithm here

# and the parameters would be its hyperparameters

# returned value would be the loss or whatever you are trying to optimize

def objective(args):

    v1 = args['v1']

    v2 = args['v2']

    v3 = args['v3']

    history_v1.append(v1)

    history_v2.append(v2)

    history_v3.append(v3)

    result = random.uniform(v2,v3)/v1

    return result



# define a search space

from hyperopt import hp



space = {

    'v1': hp.uniform('v1', 0.5,1.5),

    'v2': hp.uniform('v2', 0.5,1.5),

    'v3': hp.uniform('v3', 0.5,1.5),

}



# minimize the objective over the space

from hyperopt import fmin, tpe, space_eval

best = fmin(objective, space, algo=tpe.suggest, max_evals=1000)



print(best)
import pandas as pd



df_histories = pd.DataFrame()

df_histories["v1"] = history_v1

df_histories["v2"] = history_v2

df_histories["v3"] = history_v3

df_histories.head()

df_histories.plot()
df_histories["v1"].plot()
df_histories["v2"].plot()
df_histories["v3"].plot()