# Install:

# Kaggle environments.

!git clone https://github.com/Kaggle/kaggle-environments.git

!cd kaggle-environments && pip install .



# GFootball environment.

!apt-get update -y

!apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev



# Make sure that the Branch in git clone and in wget call matches !!

!git clone -b v2.3 https://github.com/google-research/football.git

!mkdir -p football/third_party/gfootball_engine/lib



!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.3.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so

!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .
# Set up the Environment.

from kaggle_environments import make

env = make("football", configuration={"save_video": True, 

                                      "scenario_name": "1_vs_1_easy", 

                                      "running_in_notebook": True})
env.agents
env.configuration
# Let run_left agent play against eachother and see what happens

output = env.run(["run_left", "run_left"])



# show the game

env.render(mode="human", width=800, height=600)
import inspect

import os



print(inspect.getsource(env.agents['run_left']))
def run_left_agent(obs):

    # keep running left.

    return [1] * obs.controlled_players



run_left_agent(output[0][1]['observation'])
run_left_agent(output[80][1]['observation'])