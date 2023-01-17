# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# Battle Geese environment was defined in v0.2.1

!pip install 'kaggle-environments>=0.2.1'
from kaggle_environments import evaluate, make



env = make("battlegeese", debug=True)

env.render()
%%writefile submission.py



# Silly agent which circles the perimeter clockwise.

def act(observation, configuration):

    cols = configuration.columns

    rows = configuration.rows

    goose_head = observation.geese[observation.index][0]

    if goose_head < cols - 1:

        return "E"

    elif goose_head % cols == 0:

        return "N"

    elif goose_head >= cols * (rows - 1):

        return "W"

    else:

        return "S"
# Play against yourself without an ERROR or INVALID.

# Note: The first episode in the competition will run this to weed out erroneous agents.

env.run(["/kaggle/working/submission.py", "/kaggle/working/submission.py"])

print("VALID SUBMISSION!" if env.toJSON()["statuses"] == ["INACTIVE", "INACTIVE"] else "INVALID SUBMISSION!")



# Play as the first agent against default "shortest" agent.

env.run(["/kaggle/working/submission.py", "shortest"])

env.render(mode="ipython", width=800, height=600)
# Play as first position against random agent.

trainer = env.train([None, "random"])



observation = trainer.reset()



def my_agent(observation, configuration):

    cols = configuration.columns

    rows = configuration.rows

    goose_head = observation.geese[observation.index][0]

    if goose_head < cols - 1:

        return "E"

    elif goose_head % cols == 0:

        return "N"

    elif goose_head >= cols * (rows - 1):

        return "W"

    else:

        return "S"



while not env.done:

    my_action = my_agent(observation, env.configuration)

    print("My Action", my_action)

    observation, reward, done, info = trainer.step(my_action)

    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)

env.render()
def mean_reward(rewards):

    wins = 0

    ties = 0

    loses = 0

    for r in rewards:

        r0 = 0 if r[0] is None else r[0]

        r1 = 0 if r[1] is None else r[1]

        if r0 > r1:

            wins += 1

        elif r1 > r0:

            loses += 1

        else:

            ties += 1

    return f'wins={wins/len(rewards)}, ties={ties/len(rewards)}, loses={loses/len(rewards)}'



# Run multiple episodes to estimate its performance.

# Setup agentExec as LOCAL to run in memory (runs faster) without process isolation.

print("My Agent vs Random Agent:", mean_reward(evaluate(

    "battlegeese",

    ["/kaggle/working/submission.py", "random"],

    num_episodes=20, configuration={"agentExec": "LOCAL"}

)))

print("My Agent vs Shortest Agent:", mean_reward(evaluate(

    "battlegeese",

    ["/kaggle/working/submission.py", "shortest"],

    num_episodes=20, configuration={"agentExec": "LOCAL"}

)))
# "None" represents which agent you'll manually play as.

env.play([None, "shortest", "shortest", "shortest"], width=800, height=600)