# Install:

# GFootball environment (https://github.com/google-research/football/)



!apt-get update

!apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev



# Update kaggle-environments to the newest version.

!pip3 install kaggle-environments -U



# Make sure that the Branch in git clone and in wget call matches !!

!git clone -b v2.3 https://github.com/google-research/football.git

!mkdir -p football/third_party/gfootball_engine/lib



!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.3.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so

!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .
from gfootball.env.wrappers import Simple115StateWrapper

from kaggle_environments import make

env = make("football", 

           configuration={"save_video": False, 

                          "scenario_name": "11_vs_11_kaggle", 

                          "running_in_notebook": True,

                         })

obs = env.reset()
# all game information

obs
# get raw obs for the first player we control.

obs[0]['observation']['players_raw']
from gfootball.env import observation_preprocessing

raw_obs = obs[0]['observation']['players_raw']

obs_smm = observation_preprocessing.generate_smm(raw_obs)[0]

print(obs_smm)

print(obs_smm.shape)
from gfootball.env.wrappers import Simple115StateWrapper

raw_obs = obs[0]['observation']['players_raw']

# Note: simple115v2 enables fixed_positions option.

# Source code in https://github.com/google-research/football/blob/3603de77d2bf25e53a1fbd52bc439f1377397b3b/gfootball/env/wrappers.py#L119

obs_115 = Simple115StateWrapper.convert_observation(raw_obs, fixed_positions=True)[0]

print(obs_115)

print(obs_115.shape)
%%writefile ./test.py



from gfootball.env import observation_preprocessing

from gfootball.env.wrappers import Simple115StateWrapper

import random



def agent(obs):

    

    # error:

    # raw_obs = obs[0]['observation']['players_raw']

    # obs115 = Simple115StateWrapper.convert_observation(raw_obs, True)[0]

    # obs_smm = observation_preprocessing.generate_smm([raw_obs])[0]

    

    # correct:

    raw_obs = obs['players_raw'][0]

    obs_115 = Simple115StateWrapper.convert_observation([raw_obs], True)[0]

    obs_smm = observation_preprocessing.generate_smm([raw_obs])[0]

    

    agent_output = random.randint(1, 18)

    

    # you need return a list contains your single action(a int type number from [1, 18])

    # be ware of your model output might be a float number, so make sure return a int type number.

    return [int(agent_output)]
from kaggle_environments import make



log = []



# you can set debug=True or/and logs to get more information for debug.

env = make("football", 

           configuration={"save_video": True, 

                          "scenario_name": "11_vs_11_kaggle", 

                          "running_in_notebook": True,

                         }, debug=True, logs=log)

output = env.run(["./test.py", "./test.py"])[-1]

print('Left player: reward = %s, status = %s, info = %s' % (output[0]['reward'], output[0]['status'], output[0]['info']))

print('Right player: reward = %s, status = %s, info = %s' % (output[1]['reward'], output[1]['status'], output[1]['info']))



# you can print detailed log

# print(log)



env.render(mode="human", width=800, height=600)