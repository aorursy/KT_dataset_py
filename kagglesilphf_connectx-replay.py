#coding: utf-8

import numpy as np

from kaggle_environments import evaluate, make, utils

import re
def msg2replay(msg):

    replay = msg.split(", ")

    output = [int(s) for s in replay]

    return np.array(output)
def replay_process(replay):

    replay = re.findall(r'\d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d, \d', replay)

    state, act = [], []

    for i in range(len(replay)): replay[i] = msg2replay(replay[i])

    for i in range(len(replay)):

        if any(replay[i] != replay[i-1]):

            if i != 0:

                state.append(replay[i-1])

                act.append(np.mod(np.where(replay[i] != replay[i-1]), 7))

    state_list = np.array(state).reshape(-1,42)

    act_list = np.array(act).reshape(-1,1)

    return state_list, act_list
replay = ''

replay = ''

replay = '{"configuration": {"columns": 7, "inarow": 4, "rows": 6, "steps": 1000, "timeout": 5}, "description": "Classic Connect in a row but configurable.", "name": "connectx", "rewards": [0, 1], "schema_version": 1, "specification": {"action": {"default": 0, "description": "Column to drop a checker onto the board.", "minimum": 0, "type": "integer"}, "agents": [2], "configuration": {"columns": {"default": 7, "description": "The number of columns on the board", "minimum": 1, "type": "integer"}, "inarow": {"default": 4, "description": "The number of checkers in a row required to win.", "minimum": 1, "type": "integer"}, "rows": {"default": 6, "description": "The number of rows on the board", "minimum": 1, "type": "integer"}, "steps": {"default": 1000, "description": "Maximum number of steps the environment can run.", "minimum": 1, "type": "integer"}, "timeout": {"default": 5, "description": "Seconds an agent can run before timing out.", "minimum": 1, "type": "integer"}}, "info": {}, "observation": {"board": {"default": [], "description": "Serialized grid (rows x columns). 0 = Empty, 1 = P1, 2 = P2", "type": "array"}, "mark": {"default": 0, "description": "Which checkers are the agents.", "enum": [1, 2]}}, "reset": {"observation": [{"mark": 1}, {"mark": 2}], "status": ["ACTIVE", "INACTIVE"]}, "reward": {"default": 0.5, "description": "0 = Lost, 0.5 = Draw, 1 = Won", "enum": [0, 0.5, 1], "type": ["number", "null"]}}, "statuses": ["DONE", "DONE"], "steps": [[{"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "mark": 1}, "reward": 0.5, "status": "ACTIVE"}, {"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "mark": 2}, "reward": 0.5, "status": "INACTIVE"}], [{"action": 1, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], "mark": 1}, "reward": 0.5, "status": "INACTIVE"}, {"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], "mark": 2}, "reward": 0.5, "status": "ACTIVE"}], [{"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0], "mark": 1}, "reward": 0.5, "status": "ACTIVE"}, {"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0], "mark": 2}, "reward": 0.5, "status": "INACTIVE"}], [{"action": 3, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0], "mark": 1}, "reward": 0.5, "status": "INACTIVE"}, {"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0], "mark": 2}, "reward": 0.5, "status": "ACTIVE"}], [{"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0], "mark": 1}, "reward": 0.5, "status": "ACTIVE"}, {"action": 3, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0], "mark": 2}, "reward": 0.5, "status": "INACTIVE"}], [{"action": 6, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 1, 0, 0, 1], "mark": 1}, "reward": 0.5, "status": "INACTIVE"}, {"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 1, 0, 0, 1], "mark": 2}, "reward": 0.5, "status": "ACTIVE"}], [{"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 1, 0, 0, 1], "mark": 1}, "reward": 0.5, "status": "ACTIVE"}, {"action": 3, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 1, 0, 0, 1], "mark": 2}, "reward": 0.5, "status": "INACTIVE"}], [{"action": 6, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 2, 1, 0, 1, 0, 0, 1], "mark": 1}, "reward": 0.5, "status": "INACTIVE"}, {"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 2, 1, 0, 1, 0, 0, 1], "mark": 2}, "reward": 0.5, "status": "ACTIVE"}], [{"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 2, 1, 0, 1, 0, 0, 1], "mark": 1}, "reward": 0.5, "status": "ACTIVE"}, {"action": 3, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 2, 1, 0, 1, 0, 0, 1], "mark": 2}, "reward": 0.5, "status": "INACTIVE"}], [{"action": 6, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 2, 1, 0, 1, 0, 0, 1], "mark": 1}, "reward": 0.5, "status": "INACTIVE"}, {"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 2, 1, 0, 1, 0, 0, 1], "mark": 2}, "reward": 0.5, "status": "ACTIVE"}], [{"action": 0, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 2, 1, 0, 1, 0, 0, 1], "mark": 1}, "reward": 0, "status": "DONE"}, {"action": 3, "info": {}, "observation": {"board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 2, 1, 0, 1, 0, 0, 1], "mark": 2}, "reward": 1, "status": "DONE"}]], "title": "ConnectX", "version": "1.0.0"}'



env = make("connectx", debug=True)

#env.render()



state_list, act_list = replay_process(replay)



def agent_replay(observation, configuration):

    for i in range(len(act_list)):

        if all(np.array(observation.board)==state_list[i]): action = act_list[i]

    return int(action)



env.reset()

trainer = env.train([None, agent_replay])

observation = trainer.reset()

while not env.done:

    act = agent_replay(observation, env.configuration)

    observation, reward, done, info = trainer.step(act)

    #env.render()

env.render(mode="ipython", width=256, height=256, header=False, controls=False)