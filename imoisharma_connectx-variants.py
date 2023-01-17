# We must use an internet-connected kernel to download the latest version of the environment

!pip install "kaggle_environments==0.1.6"
import kaggle_environments as ke
config_9x7 = {

    'columns': 9,

    'rows': 8,

    'inarow': 5

}

env_9x7 = ke.make('connectx', configuration=config_9x7)

env_9x7.render()
config_5x4 = {

    'columns': 5,

    'rows': 4,

    'inarow': 3

}

env_5x4 = ke.make('connectx', configuration=config_5x4)

env_5x4.render()
config_8x8 = {

    'columns': 8,

    'rows': 8,

    'inarow': 4

}

env_8x8 = ke.make('connectx', configuration=config_8x8)

env_8x8.render()