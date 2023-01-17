import sys

sys.path.insert(0, "/kaggle/input/connectx/kaggle-environments-0.1.4")
%%writefile submission.py

def a(o,c):return(__import__("kaggle_environments").envs.connectx.connectx.negamax_agent(o,c),3)[sum(o.board)<7 and sum(o.board[-6:-1])<2]
!stat --printf="%s bytes" submission.py
%run submission.py
from kaggle_environments import evaluate



def mean_reward(rewards):

    return sum(float(r[0] or 0) for r in rewards) / sum(float(r[0] or 0) + r[1] for r in rewards)



print("Agent vs  Random:", mean_reward(evaluate("connectx", [a, "random"], num_episodes=10)))

print("Random vs  Agent:", mean_reward(evaluate("connectx", ["random", a], num_episodes=10)))

print("Agent vs Negamax:", mean_reward(evaluate("connectx", [a, "negamax"], num_episodes=3)))

print("Negamax vs Agent:", mean_reward(evaluate("connectx", ["negamax", a], num_episodes=3)))