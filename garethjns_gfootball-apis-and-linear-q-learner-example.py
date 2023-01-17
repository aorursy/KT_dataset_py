import matplotlib.pyplot as plt

import pprint

import glob 

import imageio

import pathlib

import numpy as np

from typing import Tuple

from tqdm import tqdm

import os

import sys

from IPython.display import Image
# GFootball environment.

!pip install kaggle_environments

!apt-get update -y

!apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev

!git clone -b v2.3 https://github.com/google-research/football.git

!mkdir -p football/third_party/gfootball_engine/lib

!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.3.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so

!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .



# Some helper code

!git clone https://github.com/garethjns/kaggle-football.git

sys.path.append("/kaggle/working/kaggle-football/")
import gym

import gfootball  # Required as envs registered on import



simple_env = gym.make("GFootball-11_vs_11_kaggle-simple115v2-v0")

pixels_env = gym.make("GFootball-11_vs_11_kaggle-Pixels-v0")

smm_env = gym.make("GFootball-11_vs_11_kaggle-SMM-v0")



print(f"simple115v2:\n {simple_env.__str__()}\n")

print(f"Pixels:\n {pixels_env.__str__()}\n")

print(f"SMM:\n {smm_env.__str__()}\n")
from gfootball.env.football_env import FootballEnv



env_name = "GFootballBase-v0"

gym.envs.register(id=env_name,

                  entry_point="gfootball.env.football_env:FootballEnv",

                  max_episode_steps=10000)
from gfootball.env.config import Config



base_env = gym.make(env_name, config=Config())
obs = base_env.reset()



pprint.pprint(obs[0])
obs = simple_env.reset()



print(obs.shape)



pprint.pprint(obs)
from kaggle_football.viz import generate_gif, plot_smm_obs



smm_env = gym.make("GFootball-11_vs_11_kaggle-SMM-v0")

print(smm_env.reset().shape)



generate_gif(smm_env, n_steps=200)

Image(filename='smm_env_replay.gif', format='png')
from gfootball.env import create_environment



# (These are the args set by the kaggle_environments package)

COMMON_KWARGS = {"stacked": False, "representation": 'raw', "write_goal_dumps": False,

                 "write_full_episode_dumps": False, "write_video": False, "render": False,

                 "number_of_left_players_agent_controls": 1, "number_of_right_players_agent_controls": 0}



create_environment(env_name='11_vs_11_kaggle')
chk_reward_env = create_environment(env_name='11_vs_11_kaggle', rewards='scoring,checkpoints')



_ = chk_reward_env.reset()

for s in range(100):

    _, r, _, _ = chk_reward_env.step(5)

    if r > 0:

        print(f"Step {s} checkpoint reward recieved: {r}")
run_to_score_env = create_environment(env_name='academy_run_to_score')
%%writefile random_agent.py

  

from typing import Any

from typing import List



import numpy as np





class RandomAgent:

    def get_action(self, obs: Any) -> int:

        return np.random.randint(19)





AGENT = RandomAgent()





def agent(obs) -> List[int]:

    return [AGENT.get_action(obs)]
from kaggle_environments import make  

env = make("football", configuration={"save_video": True,

                                      "scenario_name": "11_vs_11_kaggle"})



# Define players

left_player = "random_agent.py"  # A custom agent, eg. random_agent.py or example_agent.py

right_player = "run_right"  # eg. A built in 'AI' agent



# Run the whole sim

# Output returned is a list of length n_steps. Each step is a list containing the output for each player as a dict.

# steps

output = env.run([left_player, right_player])



for s, (left, right) in enumerate(output):

    

    # Just print the last few steps of the output

    if s > 2990:

        print(f"\nStep {s}")



        print(f"Left player ({left_player}): \n"

              f"actions taken: {left['action']}, "

              f"reward: {left['reward']}, "

              f"status: {left['status']}, "

              f"info: {left['info']}")



        print(f"Right player ({right_player}): \n"

              f"actions taken: {right['action']}, "

              f"reward: {right['reward']}, "

              f"status: {right['status']}, "

              f"info: {right['info']}\n")



print(f"Final score: {sum([r['reward'] for r in output[0]])} : {sum([r['reward'] for r in output[1]])}")



env.render(mode="human", width=800, height=600)
print(output[-1][0].keys())

print(f"Left player: {output[-1][0]['status']}: {output[-1][0]['info']}")

print(f"Right player: {output[-1][0]['status']}: {output[-1][1]['info']}")
%%writefile broken_agent.py

  

from typing import Any

from typing import List



class DeliberateException(Exception):

    pass





class BrokenAgent:

    def get_action(self, obs: Any) -> int:

        raise DeliberateException(f"I am broken.")





AGENT = BrokenAgent()





def agent(obs) -> List[int]:

    return [AGENT.get_action(obs)]
env = make("football", configuration={"save_video": True,

                                      "scenario_name": "11_vs_11_kaggle"})



output = env.run(["random_agent.py", "broken_agent.py"])



print(len(output))

print(f"Left player: {output[-1][0]['status']}: {output[-1][0]['info']}")

print(f"Right player: {output[-1][0]['status']}: {output[-1][1]['info']}")
env = make("football", debug=True,

           configuration={"save_video": True,

                          "scenario_name": "11_vs_11_kaggle"})



try:

    output = env.run(["random_agent.py", "broken_agent.py"])

except DeliberateException as e:

    print(e)
from random_agent import agent  





env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle"})

env.reset()



# This is the observation that is passed to agent function

obs_kag_env = env.state[0]['observation']



for _ in range(3000):

    action = agent(obs_kag_env)



    # Environment step is list of agent actions, ie [[agent_1], [agent_2]], 

    # here there is 1 action per agent.

    other_agent_action = [0]

    full_obs = env.step([action, other_agent_action])

    obs_kag_env = full_obs[0]['observation']
!pip install reinforcement_learning_keras
import gym

from reinforcement_learning_keras.agents.components.history.training_history import TrainingHistory

from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy

from reinforcement_learning_keras.agents.q_learning.linear_q_agent import LinearQAgent

from sklearn.exceptions import DataConversionWarning



import warnings





agent = LinearQAgent(name="linear_q",

                     env_spec="GFootball-11_vs_11_kaggle-simple115v2-v0",

                     eps=EpsilonGreedy(eps_initial=0.9, decay=0.001, eps_min=0.01, 

                                       decay_schedule='linear'),

                     training_history=TrainingHistory(agent_name='linear_q', 

                                                      plotting_on=True, plot_every=25, 

                                                      rolling_average=1))



with warnings.catch_warnings():

    warnings.simplefilter('ignore', DataConversionWarning)

    agent.train(verbose=True, render=False,

                n_episodes=25, max_episode_steps=2000)