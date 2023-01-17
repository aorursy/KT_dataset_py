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
%%writefile submission.py

from kaggle_environments.envs.football.helpers import *



@human_readable_agent

def agent(obs):

    return Action.Top
# Set up the Environment.

from kaggle_environments import make

env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle", "running_in_notebook": True})

output = env.run(["/kaggle/working/submission.py", "/kaggle/working/submission.py"])[-1]

print('Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))

print('Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))

env.render(mode="human", width=800, height=600)
import pandas as pd



log = pd.DataFrame(env.steps).rename({0: "left_team", 1: "right_team"}, axis=1)
def print_boundaries(team):

    max_y, min_y = -1, 1

    for step in range(3000):

        ball_y = log[team][step]['observation']['players_raw'][0]['ball'][1]

        min_y = min(min_y, ball_y)

        max_y = max(max_y, ball_y)



    print(f"{team} y in [{round(min_y, 3)}, {round(max_y, 3)}]")
print_boundaries("left_team")
print_boundaries("right_team")