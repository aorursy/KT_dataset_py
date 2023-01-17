%%writefile submission.py

from numba import njit



# WORKS | Testcase 1 - native python - return static value

def python_constant():

    return 3





# WORKS | Testcase 2 - @njit - return static value

@njit

def njit_constant():

    return 3



# BROKEN | Testcase 3 - @njit - passing in Structs arguments

# BROKEN | cannot determine Numba type of <class 'kaggle_environments.utils.Struct'>

@njit

def njit_struct_arguments(observation, configuration):

    return configuration.columns - configuration.rows







# The last function defined in the file run by Kaggle in submission.py

def agent(observation, configuration) -> int:

    print('python_constant() = ', python_constant(), type(python_constant()) )    # WORKS

    print('njit_constant() = ',   njit_constant(),   type(njit_constant())   )    # WORKS

    # print('observation, configuration() = ', njit_struct_arguments(observation, configuration),   type(njit_constant(observation, configuration)) ) # BROKEN

    

    action = njit_constant()

    return int(action)
%run submission.py
from kaggle_environments import make

env = make("connectx", debug=True)

env.render()

env.reset()



observation   = env.state[0].observation

configuration = env.configuration

action = agent(observation, configuration)

print(action, type(action))