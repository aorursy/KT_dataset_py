# Kaggle environment

!git clone https://github.com/Kaggle/kaggle-environments.git

!cd kaggle-environments && pip3 install -q .



# GFootball requirement

!apt-get update -y

!apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev



# GFootball environment

# Make sure that the Branch in git clone and in wget call matches !!

!git clone -b v2.5 https://github.com/google-research/football.git

!mkdir -p football/third_party/gfootball_engine/lib

!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.5.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so

# Custom files

!wget https://gist.githubusercontent.com/RaffaeleMorganti/04192739d0a5a518ac253889eb83c6f1/raw/c09f3d602ea89e66daeda96574d966949a2896ce/11_vs_11_deterministic.py -O football/gfootball/scenarios/11_vs_11_deterministic.py 

!wget https://gist.githubusercontent.com/RaffaeleMorganti/04192739d0a5a518ac253889eb83c6f1/raw/c09f3d602ea89e66daeda96574d966949a2896ce/football_action_set.py -O football/gfootball/env/football_action_set.py

!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install -q .