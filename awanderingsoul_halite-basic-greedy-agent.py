!pip install 'kaggle-environments>=0.2.1'



import kaggle_environments

print("Kaggle Environments version:", kaggle_environments.version)
%%writefile agent.py



DIRS = ["NORTH", "SOUTH", "EAST", "WEST"]



# Each ship id will be assigned a state, one of COLLECT or DEPOSIT, which decides what it will do on a turn.

states = {}



COLLECT = "collect"

DEPOSIT = "deposit"





def argmax(arr, key=None):

  return arr.index(max(arr, key=key)) if key else arr.index(max(arr))





# This function will not hold up in practice

# E.g. cell getAdjacent(224) includes position 0, which is not adjacent

def getAdjacent(pos):

  return [

    (pos - 15) % 225,

    (pos + 15) % 225,

    (pos +  1) % 225,

    (pos -  1) % 225

  ]



def getDirTo(fromPos, toPos):

  fromY, fromX = divmod(fromPos, 15)

  toY,   toX   = divmod(toPos,   15)



  if fromY < toY: return "SOUTH"

  if fromY > toY: return "NORTH"

  if fromX < toX: return "EAST"

  if fromX > toX: return "WEST"



    

def agent(obs):

  action = {}



  player_halite, shipyards, ships = obs.players[obs.player]



  for uid, shipyard in shipyards.items():

    # Maintain one ship always

    if len(ships) == 0:

      action[uid] = "SPAWN"



  for uid, ship in ships.items():

    # Maintain one shipyard always

    if len(shipyards) == 0:

      action[uid] = "CONVERT"

      continue



    # If a ship was just made

    if uid not in states: states[uid] = COLLECT



    pos, halite = ship



    if states[uid] == COLLECT:

      if halite > 2500:

        states[uid] = DEPOSIT



      elif obs.halite[pos] < 100:

        best = argmax(getAdjacent(pos), key=obs.halite.__getitem__)

        action[uid] = DIRS[best]



    if states[uid] == DEPOSIT:

      if halite < 200: states[uid] = COLLECT



      direction = getDirTo(pos, list(shipyards.values())[0])

      if direction: action[uid] = direction

      else: states[uid] = COLLECT





  return action
# Sparring Partner

def null_agent(*_): return {}



for _ in range(3):

    env = kaggle_environments.make("halite", debug=True)

    env.run(["agent.py", null_agent])

    env.render(mode="ipython", width=800, height=600)
def mean_reward(rewards):

    wins = 0

    ties = 0

    loses = 0



    for r in rewards:

        r0 = r[0] or 0

        r1 = r[1] or 0



        if   r0 > r1: wins  += 1

        elif r1 > r0: loses += 1

        else:         ties  += 1



    return [wins / len(rewards), ties / len(rewards), loses / len(rewards)]





import inspect

def test_against(enemy, n=25):

    results = mean_reward(kaggle_environments.evaluate(

        "halite",

        ["agent.py", enemy],

        num_episodes=n

    ))



    enemy_name = enemy.__name__ if inspect.isfunction(enemy) else enemy

    print("My Agent vs {}: wins={}, ties={}, loses={}".format(enemy_name, *results))



test_against(null_agent)

test_against("random")