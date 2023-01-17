# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.4 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# ConnectX environment was defined in v0.1.4

!pip install 'kaggle-environments>=0.1.4'
from kaggle_environments import evaluate, make



env = make("connectx", debug=True)
env.agents
env.configuration
env.specification
# This agent random chooses a non-empty column.

def my_agent(observation, configuration):

    from random import choice

    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
# Play as first position against random agent.

trainer = env.train([None, "random"])



observation = trainer.reset()



print("Observation contains:\t", observation)

print("Configuration contains:\t", env.configuration)



my_action = my_agent(observation, env.configuration)

print("My Action", my_action)

observation, reward, done, info = trainer.step(my_action)

# env.render(mode="ipython", width=100, height=90, header=False, controls=False)

env.render(mode="ipython", width=100, height=90, header=False, controls=False)

print("Observation after:\t", observation)

#env.render()
def my_comatose_agent(observation, configuration):

    from random import choice

    from time import sleep

    sleep(2)

    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])



def my_sleepy_agent(observation, configuration):

    from random import choice

    from time import sleep

    sleep(1)

    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
print(evaluate("connectx", [my_comatose_agent, "random"], num_episodes=1))

print(evaluate("connectx", [my_sleepy_agent, "random"], num_episodes=1))

print(evaluate("connectx", [my_agent, "random"], num_episodes=1))
def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=3)))
import inspect

import os



print(inspect.getsource(env.agents['negamax']))
neg_v_neg = evaluate("connectx", [env.agents['negamax'], "negamax"], num_episodes=10)

print(neg_v_neg)

print(mean_reward(neg_v_neg))
def try_not_to_loose_agent(observation, configuration):

    from random import choice

    from kaggle_environments import make

    env = make("connectx", debug=True)

    trainer = env.train([None, "negamax"])

    

    cols = list(range(configuration.columns))

    while cols:

        # We set the state of the environment, so we can experiment on it.

        env.state[0]['observation'] = observation

        env.state[1]['observation'] = observation

        # Take a random column that is not full

        my_action = choice([c for c in cols if observation.board[c] == 0])

        # Simulate the next step

        out = env.train([None, "negamax"]).step(my_action)

        # If the next step makes us lose, take a different step!

        if out[2]:

            cols.pop(my_action)

        else:

            return my_action

    else:

        # If we run out of steps to take, we just loose with one step.

        return 1
stupid_v_random = evaluate("connectx", [try_not_to_loose_agent, "random"], num_episodes=10)

print(stupid_v_random)

print(mean_reward(stupid_v_random))



stupid_v_neg = evaluate("connectx", [try_not_to_loose_agent, "negamax"], num_episodes=10)

print(stupid_v_neg)

print(mean_reward(stupid_v_neg))
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(try_not_to_loose_agent, "submission.py")