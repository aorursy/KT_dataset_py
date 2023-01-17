# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# ConnectX environment was defined in v0.1.6
!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
env.render()
def peijen_agent(observation, configuration):
    import numpy as np

    def can_I_win(column, board):
        for r in range(configuration.rows):
            if board[r][column] != 0 and board[r][column] != observation.mark:
                if r >= 4:
                    return True
                else:
                    return False
            else:
                # can't really win but block the other player
                if r > 4 and board[r][column] != observation.mark:
                    return True
        return True

    # setup
    plays = [3,2,4,5,1,6,0]
    playable_columns = [c for c in range(configuration.columns) if observation.board[c] == 0]
    board = np.reshape(observation.board, (configuration.rows, configuration.columns))

    for column in plays:
        if column in playable_columns and can_I_win(column, board):
            return column

    # cannot win? play a valid column
    return playable_columns[0]
# Play as the first agent against default "negamax" agent.
env.reset()
env.run([peijen_agent, "negamax"])
env.render(mode="ipython", width=500, height=450)

# Play as the second agent against default "negamax" agent.
env.reset()
env.run(["negamax", peijen_agent])
env.render(mode="ipython", width=500, height=450)
# Play as first position against random agent.
trainer = env.train([None, "negamax"])

observation = trainer.reset()

while not env.done:
    my_action = peijen_agent(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()
#def mean_reward(rewards):
#    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

# Run multiple episodes to estimate its performance.
#print("Peijen Agent vs Random Agent:", mean_reward(evaluate("connectx", [peijen_agent, "random"], num_episodes=10)))
#print("Peijen Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [peijen_agent, "negamax"], num_episodes=10)))
#print("Peijen Agent vs Black Agent:", mean_reward(evaluate("connectx", [peijen_agent, black_agent], num_episodes=10)))
#print("Black Agent vs Peijen Agent:", mean_reward(evaluate("connectx", [black_agent, peijen_agent], num_episodes=10)))
# "None" represents which agent you'll manually play as (first or second player).
env.play([None, "negamax"], width=500, height=450)
import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(peijen_agent, "submission.py")
# Note: Stdout replacement is a temporary workaround.
import sys
out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")