# Install:

# Kaggle environments.

!git clone -q https://github.com/Kaggle/kaggle-environments.git

!cd kaggle-environments && pip install -q .

# GFootball environment.

!apt-get update -qy 

!apt-get install -qy libsdl2-gfx-dev libsdl2-ttf-dev

# Make sure that the Branch in git clone and in wget call matches !!

!git clone -b v2.3 https://github.com/google-research/football.git

!mkdir -p football/third_party/gfootball_engine/lib

!wget -q --show-progress https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.3.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so

!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install -q .
%%writefile submission.py

import numpy as np

from kaggle_environments.envs.football.helpers import *



@human_readable_agent

def agent(obs):

    

    # Global param

    goal_threshold = 0.5

    gravity = 0.098

    pick_height = 0.5

    step_length = 0.015 # As we always sprint

    body_radius = 0.012

    slide_threshold = step_length + body_radius

    

    # Ignore drag to estimate the landing point

    def ball_landing(ball, ball_direction):

        start_height = ball[2]

        end_height = pick_height

        start_speed = ball_direction[2]

        time = np.sqrt(start_speed**2/gravity**2 - 2/gravity*(end_height-start_height)) + start_speed/gravity

        return [ball[0]+ball_direction[0]*time, ball[1]+ball_direction[1]*time]

    

    # Check whether pressing on direction buttons and take action if so

    # Else press on direction first

    def sticky_check(action, direction):

        if direction in obs['sticky_actions']:

            return action

        else:

            return direction

    

    # Find right team positions

    def_team_pos = obs['right_team']

    # Fix goalkeeper index here as PlayerRole has issues

    # Default PlayerRole [0, 7, 9, 2, 1, 1, 3, 5, 5, 5, 6]

    def_keeper_pos = obs['right_team'][0]

    

    # We always control left team (observations and actions

    # are mirrored appropriately by the environment).

    controlled_player_pos = obs['left_team'][obs['active']]

    # Get team size

    N = len(obs['left_team'])

    

    # Does the player we control have the ball?

    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:

        # Kickoff strategy: short pass to teammate

        if obs['game_mode'] == GameMode.KickOff:

            return sticky_check(Action.ShortPass, Action.Top) if controlled_player_pos[1] > 0 else sticky_check(Action.ShortPass, Action.Bottom)

        # Goalkick strategy: high pass to front

        if obs['game_mode'] == GameMode.GoalKick:

            return sticky_check(Action.LongPass, Action.Right)

        # Freekick strategy: make shot when close to goal, high pass when in back field, and short pass in mid field

        if obs['game_mode'] == GameMode.FreeKick:

            if controlled_player_pos[0] > goal_threshold:

                if abs(controlled_player_pos[1]) < 0.1:

                    return sticky_check(Action.Shot, Action.Right)

                if abs(controlled_player_pos[1]) < 0.3:

                    return sticky_check(Action.Shot, Action.TopRight) if controlled_player_pos[1]>0 else sticky_check(Action.Shot, Action.BottomRight)

                return sticky_check(Action.HighPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.HighPass, Action.Bottom)

            

            if controlled_player_pos[0] < -goal_threshold:

                if abs(controlled_player_pos[1]) < 0.3:

                    return sticky_check(Action.HighPass, Action.Right)

                return sticky_check(Action.HighPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.HighPass, Action.Bottom)

            

            if abs(controlled_player_pos[1]) < 0.3:

                return sticky_check(Action.ShortPass, Action.Right)

            return sticky_check(Action.ShortPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.ShortPass, Action.Bottom)

        # Corner strategy: high pass to goal area

        if obs['game_mode'] == GameMode.Corner:

            return sticky_check(Action.HighPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.HighPass, Action.Bottom)

        # Throwin strategy: short pass into field

        if obs['game_mode'] == GameMode.ThrowIn:

            return sticky_check(Action.ShortPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.ShortPass, Action.Bottom)

        # Penalty strategy: make a shot

        if obs['game_mode'] == GameMode.Penalty:

            right_actions = [Action.TopRight, Action.BottomRight, Action.Right]

            for action in right_actions:

                if action in obs['sticky_actions']:

                    return Action.Shot

            return np.random.choice(right_actions)

            

        # Defending strategy

        if controlled_player_pos[0] < -goal_threshold:

            if abs(controlled_player_pos[1]) < 0.3:

                return sticky_check(Action.HighPass, Action.Right)

            return sticky_check(Action.HighPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.HighPass, Action.Bottom)

            

        # Make sure player is running.

        if Action.Sprint not in obs['sticky_actions']:

            return Action.Sprint

        

        # Shot if we are 'close' to the goal (based on 'x' coordinate).

        if controlled_player_pos[0] > goal_threshold:

            if abs(controlled_player_pos[1]) < 0.1:

                return sticky_check(Action.Shot, Action.Right)

            if abs(controlled_player_pos[1]) < 0.3:

                return sticky_check(Action.Shot, Action.TopRight) if controlled_player_pos[1]>0 else sticky_check(Action.Shot, Action.BottomRight)

            elif controlled_player_pos[0] < 0.85:

                return Action.Right

            else:

                return sticky_check(Action.HighPass, Action.Top) if controlled_player_pos[1]>0 else sticky_check(Action.HighPass, Action.Bottom)

        

        # Run towards the goal otherwise.

        return Action.Right

    else:

        # when the ball is generally on the ground not flying

        if obs['ball'][2] <= pick_height:

            # Run towards the ball's left position.

            if obs['ball'][0] > controlled_player_pos[0] + slide_threshold:

                if obs['ball'][1] > controlled_player_pos[1] + slide_threshold:

                    return Action.BottomRight

                elif obs['ball'][1] < controlled_player_pos[1] - slide_threshold:

                    return Action.TopRight

                else:

                    return Action.Right

            elif obs['ball'][0] < controlled_player_pos[0] + slide_threshold:

                if obs['ball'][1] > controlled_player_pos[1] + slide_threshold:

                    return Action.BottomLeft

                elif obs['ball'][1] < controlled_player_pos[1] - slide_threshold:

                    return Action.TopLeft

                else:

                    return Action.Left

            # When close to the ball, try to take over.

            else:

                return Action.Slide

        # when the ball is flying

        else:

            landing_point = ball_landing(obs['ball'], obs['ball_direction'])

            # Run towards the landing point's left position.

            if landing_point[0] - body_radius > controlled_player_pos[0] + slide_threshold:

                if landing_point[1] > controlled_player_pos[1] + slide_threshold:

                    return Action.BottomRight

                elif landing_point[1] < controlled_player_pos[1] - slide_threshold:

                    return Action.TopRight

                else:

                    return Action.Right

            elif landing_point[0] - body_radius < controlled_player_pos[0] + slide_threshold:

                if landing_point[1] > controlled_player_pos[1] + slide_threshold:

                    return Action.BottomLeft

                elif landing_point[1] < controlled_player_pos[1] - slide_threshold:

                    return Action.TopLeft

                else:

                    return Action.Left

            # Try to take over the ball if close to the ball.

            elif controlled_player_pos[0] > goal_threshold:

                # Keep making shot when around landing point

                return sticky_check(Action.Shot, Action.Right) if ['ball'][2] <= pick_height else Action.Idle

            else:

                return sticky_check(Action.Slide, Action.Right) if ['ball'][2] <= pick_height else Action.Idle
# Set up the Environment.

from kaggle_environments import make

env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle", "running_in_notebook": True})

output = env.run(["/kaggle/working/submission.py", "do_nothing"])[-1]

print('Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))

print('Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))

env.render(mode="human", width=800, height=600)
# Validation

from datetime import datetime

from kaggle_environments import make

start = datetime.now()

env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle", "running_in_notebook": True})

output = env.run(["/kaggle/working/submission.py", "/kaggle/working/submission.py"])[-1]

print('Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))

print('Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))

print(datetime.now()-start)

env.render(mode="human", width=800, height=600)
import pandas as pd

log = pd.DataFrame(env.steps)
log[0].head()
log.iloc[0,0]
ball_log = pd.DataFrame()

ball_log['ball'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball'])

ball_log['ball_direction'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball_direction'])

ball_log.head(20)
print('Ball position at step 9 is ', ball_log.iloc[9,0])

print('Ball position at step 10 is ', ball_log.iloc[10,0])

print('Ball speed at step 9 is ', ball_log.iloc[9,1])

print('Ball speed at step 10 is ', ball_log.iloc[10,1])

print('Ball position change between step 9 and 10 is ',[b - a for a, b in zip(ball_log.iloc[9,0], ball_log.iloc[10,0])])

print('Ball speed change between step 9 and 10 is ',[b - a for a, b in zip(ball_log.iloc[9,1], ball_log.iloc[10,1])])
print('Ball position at step 9 is ', ball_log.iloc[8,0])

print('Ball position at step 10 is ', ball_log.iloc[9,0])

print('Ball speed at step 9 is ', ball_log.iloc[8,1])

print('Ball speed at step 10 is ', ball_log.iloc[9,1])

print('Ball position change between step 9 and 10 is ',[b - a for a, b in zip(ball_log.iloc[8,0], ball_log.iloc[9,0])])

print('Ball speed change between step 9 and 10 is ',[b - a for a, b in zip(ball_log.iloc[8,1], ball_log.iloc[9,1])])
right1 = pd.DataFrame()

right1['position'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['right_team'][1])

right1['speed'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['right_team_direction'][1])
print('Right team player 1 position at step 35 is ',right1['position'][35])

print('Right team player 1 position at step 36 is ',right1['position'][36])

print('Right team player 1 speed at step 35 is ',right1['speed'][35])

print('Right team player 1 speed at step 36 is ',right1['speed'][36])

print('Right team player 1 position change at step 35 is ',[b - a for a, b in zip(right1.iloc[35,0], right1.iloc[36,0])])

print('Right team player 1 speed change at step 35 is ',[b - a for a, b in zip(right1.iloc[35,1], right1.iloc[36,1])])
step = 70

player = pd.DataFrame()

player['position'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['left_team'][8])

player['speed'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['left_team_direction'][8])

ball = pd.DataFrame()

ball['position'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball'])

ball['speed'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball_direction'])

print('Player position at step ',step,' is ',player['position'][step])

print('Ball position at step ',step,' is ',ball['position'][step])

print('Player position at step ',step+1,' is ',player['position'][step+1])

print('Ball position at step ',step+1,' is ',ball['position'][step+1])

print('Player position at step ',step+2,' is ',player['position'][step+2])

print('Ball position at step ',step+2,' is ',ball['position'][step+2])

print('Player position at step ',step+3,' is ',player['position'][step+3])

print('Ball position at step ',step+3,' is ',ball['position'][step+3])
step = 150

player = pd.DataFrame()

player['position'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['left_team'][9])

player['speed'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['left_team_direction'][9])

ball = pd.DataFrame()

ball['position'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball'])

ball['speed'] = log[0].apply(lambda x: x['observation']['players_raw'][0]['ball_direction'])

for i in range(5):

    print('Player position at step ',step+i,' is ',player['position'][step+i])

    print('Ball position at step ',step+i,' is ',ball['position'][step+i])

    print('Player speed at step ',step+i,' is ',player['speed'][step+i])

    print('Ball speed at step ',step+i,' is ',ball['speed'][step+i])