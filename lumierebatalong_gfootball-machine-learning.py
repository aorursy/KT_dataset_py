# Install:

# Kaggle environments.

!git clone https://github.com/Kaggle/kaggle-environments.git

!cd kaggle-environments && pip install .



# GFootball environment.

!apt-get update -y

!apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev



# Make sure that the Branch in git clone and in wget call matches !!

!git clone -b v2.6 https://github.com/google-research/football.git

!mkdir -p football/third_party/gfootball_engine/lib



!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.6.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so

!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .
%%writefile AlbertEinsteinAcademic.py

from kaggle_environments.envs.football.helpers import *

import numpy



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

    

    

    if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:

        

        # Shot if we are 'close' to the goal (based on 'x-y' coordinate).

        if controlled_player_pos[0] >= 0.7 and controlled_player_pos[1] > 0.3:

            return numpy.random.choice([Action.Shot, Action.TopRight, Action.Right])

        

        if controlled_player_pos[0] >= 0.7 and controlled_player_pos[1] < -0.3:

            return  numpy.random.choice([Action.Shot, Action.BottomRight, Action.Right])

        

        if controlled_player_pos[0] > 0.5 and (controlled_player_pos[1] >= -0.5 and controlled_player_pos[1] <= 0.5):

            return numpy.random.choice([Action.Shot, Action.TopRight, Action.BottomRight]) 

        

        if controlled_player_pos[0] >= 0.7:

            if controlled_player_pos[1] >= -0.3 and controlled_player_pos[1] <= 0.3:

                return Action.Shot 

            

        

        if controlled_player_pos[0] < 0.0:

            return numpy.random.choice([Action.ShortPass, Action.LongPass, Action.HighPass])

        

        #go forward and remove ball in dangerous place.

        if controlled_player_pos[0] < -0.5:

            return Action.Right

        

        if controlled_player_pos[0] >= 0.0:

            return Action.Right



        

        # Run towards the goal otherwise.

        return Action.Right 

        

    else:

        

        right_player_pos = obs['right_team'][obs['active']]

        

        # Run towards the ball.

        if obs['ball'][0] > controlled_player_pos[0] + 0.05:

            return numpy.random.choice([Action.Right, Action.BottomRight, Action.TopRight])



        if obs['ball'][0] < controlled_player_pos[0] - 0.05:

            return numpy.random.choice([Action.Left, Action.TopLeft, Action.BottomLeft])



        if obs['ball'][1] > controlled_player_pos[1] + 0.05:

            return numpy.random.choice([Action.Bottom, Action.BottomLeft, Action.BottomRight])



        if obs['ball'][1] < controlled_player_pos[1] - 0.05:

            return numpy.random.choice([Action.Top, Action.TopLeft, Action.TopRight])

        

        #run toward the opposite player

        if right_player_pos[0] > controlled_player_pos[0] + 0.05:

            return numpy.random.choice([Action.Right, Action.BottomRight, Action.TopRight])

        

        if right_player_pos[0] < controlled_player_pos[0] - 0.05:

            return numpy.random.choice([Action.Left, Action.BottomLeft, Action.TopLeft])

        

        if right_player_pos[1] > controlled_player_pos[1] + 0.05:

            return numpy.random.choice([Action.Bottom, Action.BottomRight, Action.BottomLeft])

        

        if right_player_pos[1] < controlled_player_pos[1] - 0.05:

            return numpy.random.choice([Action.Top, Action.TopLeft, Action.TopRight])



        # Try to take over the ball if close to the ball.

        return Action.Slide
%%writefile NeilsBohrAcademic.py

from kaggle_environments.envs.football.helpers import *

import numpy



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

    controlled_player_pos = obs['left_team'][obs['active'] - 5]

    

    # Does the player we control have the ball?

    right_player_pos = obs['right_team'][obs['active']]

    goalKeeperR = obs['right_team'][0] #opponent goal keeper

    

################################### Team Strategy ###############################################################   



    if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:

        

        # single player have a ball.

        player_pos = obs['left_team'][obs['active']]

        u_vector = numpy.zeros((2, 1)) # 

        w_vector = u_vector.copy()

        

        u_vector[0] = goalKeeperR[0] - player_pos[0]

        u_vector[1] = goalKeeperR[1] - player_pos[1]

        distanceG_P = numpy.linalg.norm(u_vector, 2) # distance between active players and goal keeper oppenent

        

        w_vector[0] = player_pos[0] - right_player_pos[0]

        w_vector[1] = player_pos[1] - right_player_pos[0]

        distanceP_P = numpy.linalg.norm(w_vector, 2) # distance between active player and opponent

        

        #left goal keeper kicks a ball in dangerous place

        if obs['active'] == 0: 

            x = numpy.random.choice([Action.Shot, Action.LongPass])

            print('GoalKeeper: ', x, ' Active players: ', obs['active'])

            return x

        

        

        #*************************** controlling only one players *******************************#

        

        # player's prepares  to score.

        if player_pos[0] > 0.5 and (player_pos[1] >= -0.5 and player_pos[1] <= 0.5):

            x = numpy.random.choice([Action.TopRight, Action.BottomRight, Action.Right, 

                                     Action.Shot, Action.Top, Action.Bottom]) 

            return x

        

        #player echap or dribble

        if distanceP_P < 0.025:

            return numpy.random.choice([Action.ReleaseSprint, Action.BottomRight, Action.TopRight, 

                                        Action.ReleaseDribble])

        

        # Shot if we are 'close' to the goal (based on 'x-y' coordinate).

        if player_pos[0] >= 0.7:

            

            #player tends to adjust a ball to go to score

            if player_pos[1] > 0.5:

                x = numpy.random.choice([Action.TopRight, Action.Top, Action.Shot])

                return x

            

            if player_pos[1] < -0.5:

                x = numpy.random.choice([Action.BottomRight, Action.Bottom, Action.Shot])

                return  x

            

            if distanceG_P < 0.05:

                return numpy.random.choice([Action.BottomRight, Action.TopRight, Action.Shot])

            

            if player_pos[1] >= -0.5 and player_pos[1] <= 0.5:

                x = Action.Shot

                return x

        #******************************************************************************************#

            

        # make some passing ball forward

        if (controlled_player_pos[0] - player_pos[0]) > 0: 

            x = numpy.random.choice([Action.ShortPass, Action.LongPass, Action.HighPass])

            return x

        

        # Run towards the goal otherwise.

        return Action.Right 

        

    else:

        

        # Run towards the ball.

        if (obs['ball'][0] > controlled_player_pos[0] + 0.05) and (right_player_pos[0] > controlled_player_pos[0] + 0.05):

            return numpy.random.choice([Action.Right, Action.BottomRight, Action.TopRight])



        if (obs['ball'][0] < controlled_player_pos[0] - 0.05) and (right_player_pos[0] < controlled_player_pos[0] - 0.05):

            return numpy.random.choice([Action.Left, Action.TopLeft, Action.BottomLeft])



        if (obs['ball'][1] > controlled_player_pos[1] + 0.05) and (right_player_pos[1] > controlled_player_pos[1] + 0.05):

            return numpy.random.choice([Action.Bottom, Action.BottomLeft, Action.BottomRight])



        if (obs['ball'][1] < controlled_player_pos[1] - 0.05) and (right_player_pos[1] < controlled_player_pos[1] - 0.05):

            return numpy.random.choice([Action.Top, Action.TopLeft, Action.TopRight])

        

        # Run towards the ball.

        if obs['ball'][0] > controlled_player_pos[0] + 0.05:

            return numpy.random.choice([Action.Right, Action.BottomRight, Action.TopRight])



        if obs['ball'][0] < controlled_player_pos[0] - 0.05:

            return numpy.random.choice([Action.Left, Action.TopLeft, Action.BottomLeft])



        if obs['ball'][1] > controlled_player_pos[1] + 0.05:

            return numpy.random.choice([Action.Bottom, Action.BottomLeft, Action.BottomRight])



        if obs['ball'][1] < controlled_player_pos[1] - 0.05:

            return numpy.random.choice([Action.Top, Action.TopLeft, Action.TopRight])



        # Try to take over the ball if close to the ball.

        return Action.Slide
print('Welcome to this beautiful Kaggle stadium. We are going to watch a big derby between Einstein Academic Physics football and Bohr Academic Physics football. Goodluck!')
from kaggle_environments import make

env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle", "running_in_notebook": True})

output = env.run(["/kaggle/working/AlbertEinsteinAcademic.py", "/kaggle/working/NeilsBohrAcademic.py"])[-1]

print('Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))

print('Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))

env.render(mode="human", width=300, height=400)