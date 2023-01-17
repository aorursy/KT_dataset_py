!pip install 'kaggle-environments'
from kaggle_environments import evaluate

from operator import itemgetter



def grade_agent(game_name, agent, opponent, episodes):

    as_p1 = evaluate(game_name, [agent, opponent], num_episodes=episodes)

    as_p1_reward = sum(map(itemgetter(0), as_p1))

    as_p1_total = sum(map(itemgetter(1), as_p1)) + as_p1_reward

    

    as_p2 = evaluate(game_name, [opponent, agent], num_episodes=episodes)

    as_p2_reward = sum(map(itemgetter(1), as_p2))

    as_p2_total = sum(map(itemgetter(0), as_p2)) + as_p2_reward

    

    return 100 * (as_p1_reward + as_p2_reward) / (as_p1_total + as_p2_total)

import random



def combined_agent(default_agent, alternate_agent, epsilon):

    def updated_agent(obs, config):

        if (random.random() < epsilon):

            return alternate_agent(obs, config)

        return default_agent(obs,config)

    return updated_agent



from kaggle_environments.envs.connectx.connectx import negamax_agent

from kaggle_environments.envs.connectx.connectx import random_agent



e_greedy_negamax = combined_agent(negamax_agent, random_agent, 0.2)