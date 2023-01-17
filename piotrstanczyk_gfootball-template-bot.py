# Install:

# Kaggle environments.

!git clone https://github.com/Kaggle/kaggle-environments.git

!cd kaggle-environments && pip install .



# GFootball environment.

!apt-get update -y

!apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev



# Make sure that the Branch in git clone and in wget call matches !!

!git clone -b v2.7 https://github.com/google-research/football.git

!mkdir -p football/third_party/gfootball_engine/lib



!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.7.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so

!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .
%%writefile submission.py

from kaggle_environments.envs.football.helpers import *



# @human_readable_agent wrapper modifies raw observations 

# provided by the environment:

# https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#raw-observations

# into a form easier to work with by humans.

# Following modifications are applied:

# - Action, PlayerRole and GameMode enums are introduced.

# - 'sticky_actions' are turned into a set of active actions (Action enum)

#    see usage example below.

# - 'game_mode' is turned into GameMode enum.

# - 'designated' field is removed, as it always equals to 'active'

#    when a single player is controlled on the team.

# - 'left_team_roles'/'right_team_roles' are turned into PlayerRole enums.

# - Action enum is to be returned by the agent function.

@human_readable_agent

def agent(obs):

    # Make sure player is running.

    if Action.Sprint not in obs['sticky_actions']:

        return Action.Sprint

    # We always control left team (observations and actions

    # are mirrored appropriately by the environment).

    controlled_player_pos = obs['left_team'][obs['active']]

    # Does the player we control have the ball?

    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:

        # Shot if we are 'close' to the goal (based on 'x' coordinate).

        if controlled_player_pos[0] > 0.5:

            return Action.Shot

        # Run towards the goal otherwise.

        return Action.Right

    else:

        # Run towards the ball.

        if obs['ball'][0] > controlled_player_pos[0] + 0.05:

            return Action.Right

        if obs['ball'][0] < controlled_player_pos[0] - 0.05:

            return Action.Left

        if obs['ball'][1] > controlled_player_pos[1] + 0.05:

            return Action.Bottom

        if obs['ball'][1] < controlled_player_pos[1] - 0.05:

            return Action.Top

        # Try to take over the ball if close to the ball.

        return Action.Slide
# Set up the Environment.

from kaggle_environments import make

env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle", "running_in_notebook": True})

output = env.run(["/kaggle/working/submission.py", "do_nothing"])[-1]

print('Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))

print('Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))

env.render(mode="human", width=800, height=600)