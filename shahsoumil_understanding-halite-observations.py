# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# Halite environment was defined in v0.2.1

!pip install 'kaggle-environments>=0.2.1'
from kaggle_environments import evaluate, make



env = make("halite", debug=True)

env.render(mode="ipython", width=800, height=600,controls=True)

print(list(env.agents))
env.run(["random", "random","random","random"])

env.render(mode="ipython", width=800, height=600)
%%writefile submission.py



from random import choice

import numpy as np

def agent(obs):

    action = {}

    

    board=np.reshape(np.float32(obs.halite),(15,15))

    me=0

    enemy=1

    if obs.player==me:

        current_player="me"

    else:

        current_player="enemy"

    my_total_halite=obs.players[me][0]

    enemy_total_halite=obs.players[enemy][0]

    my_shipyards=obs.players[me][1]

    enemy_shipyards=obs.players[enemy][1]

    my_ships=obs.players[me][2]

    enemy_ships=obs.players[enemy][2]

    print("*"*10)

    print("At timestep",obs.step)

    print("Curent player is",obs.player,"This will be important when playing against yourself")

    print("I have",my_total_halite,"halites")

    print("Enemy has",enemy_total_halite,"halites")

    for agent_id in my_shipyards.keys():

        print("My shipyard",agent_id,"is at",my_shipyards[agent_id],str((my_shipyards[agent_id]%15,my_shipyards[agent_id]//15)))

    for agent_id in my_ships.keys():

        print("My ship",agent_id,"is at",my_ships[agent_id][0],str((my_ships[agent_id][0]%15,my_ships[agent_id][0]//15)),"and has",my_ships[agent_id][1],"halite")

    for agent_id in enemy_shipyards.keys():

        print("Enemy shipyard",agent_id,"is at",enemy_shipyards[agent_id],str((enemy_shipyards[agent_id]%15,enemy_shipyards[agent_id]//15)))

    for agent_id in enemy_ships.keys():

        print("Enemy ship",agent_id,"is at",enemy_ships[agent_id][0],str((enemy_ships[agent_id][0]%15,enemy_ships[agent_id][0]//15)),"and has",enemy_ships[agent_id][1],"halite")

        

    ship_action=choice(["NORTH", "SOUTH", "EAST", "WEST", None])

    ship_id=choice(list(my_ships.keys())+list(my_shipyards.keys()))

    if ship_action is not None:

        action[ship_id] = ship_action

    print("Action taken:",action)

    return action
# Play as the first agent against default "shortest" agent.

env.run(["/kaggle/working/submission.py", "random"])

env.render(mode="ipython", width=800, height=600)