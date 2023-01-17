# Install:

# Kaggle environments.

!git clone https://github.com/Kaggle/kaggle-environments.git

!cd kaggle-environments && pip install .



# GFootball environment.

!apt-get update -y

!apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev



# Make sure that the Branch in git clone and in wget call matches !!

!git clone -b v2.5 https://github.com/google-research/football.git

!mkdir -p football/third_party/gfootball_engine/lib



!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.5.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so

!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .
import gfootball.env as football_env



env = football_env.create_environment(

    env_name="11_vs_11_kaggle",

    representation='extracted',

    stacked=False,

    rewards='scoring',

    logdir='.',

    write_goal_dumps=False,

    write_full_episode_dumps=False,

    render=False,

    number_of_right_players_agent_controls=1,

    dump_frequency=0)



obs0, obs1 = env.reset()

actions = [0, 0]

print(f'before: actions={actions}')

env.step(actions)

print(f'after: actions={actions}')