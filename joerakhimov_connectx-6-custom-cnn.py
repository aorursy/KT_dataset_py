!pip install stable-baselines3
import gym

from kaggle_environments import make, evaluate



import os

import numpy as np

import torch as th

from torch import nn as nn

import torch.nn.functional as F



from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.monitor import load_results

from stable_baselines3.common.torch_layers import NatureCNN

from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# ConnectX wrapper from Alexis' notebook.

# Changed shape, channel first.

# Changed obs/2.0

class ConnectFourGym(gym.Env):

    def __init__(self, agent2="random"):

        ks_env = make("connectx", debug=True)

        self.env = ks_env.train([None, agent2])

        self.rows = ks_env.configuration.rows

        self.columns = ks_env.configuration.columns

        # Learn about spaces here: http://gym.openai.com/docs/#spaces

        self.action_space = gym.spaces.Discrete(self.columns)

        self.observation_space = gym.spaces.Box(low=0, high=1, 

                                            shape=(1,self.rows,self.columns), dtype=np.float)

        # Tuple corresponding to the min and max possible rewards

        self.reward_range = (-10, 1)

        # StableBaselines throws error if these are not defined

        self.spec = None

        self.metadata = None

    def reset(self):

        self.obs = self.env.reset()

        return np.array(self.obs['board']).reshape(1,self.rows,self.columns)/2

    def change_reward(self, old_reward, done):

        if old_reward == 1: # The agent won the game

            return 1

        elif done: # The opponent won the game

            return -1

        else: # Reward 1/42

            return 1/(self.rows*self.columns)

    def step(self, action):

        # Check if agent's move is valid

        is_valid = (self.obs['board'][int(action)] == 0)

        if is_valid: # Play the move

            self.obs, old_reward, done, _ = self.env.step(int(action))

            reward = self.change_reward(old_reward, done)

        else: # End the game and penalize agent

            reward, done, _ = -10, True, {}

        return np.array(self.obs['board']).reshape(1,self.rows,self.columns)/2, reward, done, _
env = ConnectFourGym()

env
# Create directory for logging training information

log_dir = "log/"

os.makedirs(log_dir, exist_ok=True)



# Logging progress

env = Monitor(env, log_dir, allow_early_resets=True)

env
env = DummyVecEnv([lambda: env])

env
env.observation_space.sample()
class Net(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):

        super(Net, self).__init__(observation_space, features_dim)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.fc3 = nn.Linear(384, features_dim)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = nn.Flatten()(x)

        x = F.relu(self.fc3(x))

        return x
policy_kwargs = {

    'activation_fn':th.nn.ReLU, 

    'net_arch':[64, dict(pi=[32, 16], vf=[32, 16])],

    'features_extractor_class':Net,

}

learner = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs)



learner.policy
%%time

learner.learn(total_timesteps=100_000)
df = load_results(log_dir)['r']

df.rolling(window=1000).mean().plot()
learner.predict(env.reset())
def testagent(obs, config):

    import numpy as np

    obs = np.array(obs['board']).reshape(1, config.rows, config.columns)/2

    action, _ = learner.predict(obs)

    return int(action)
def get_win_percentages(agent1, agent2, n_rounds=100):

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time      

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
get_win_percentages(agent1=testagent, agent2="random")
env = make("connectx", debug=True)



# Two random agents play one game round

env.run([testagent, "random"])



# Show the game

env.render(mode="ipython")
%%writefile submission.py

def agent(obs, config):

    import numpy as np

    import torch as th

    from torch import nn as nn

    import torch.nn.functional as F

    from torch import tensor

    

    class Net(nn.Module):

        def __init__(self):

            super(Net, self).__init__()

            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

            self.fc3 = nn.Linear(384, 512)

            self.shared1 = nn.Linear(512, 64)

            self.policy1 = nn.Linear(64, 32)

            self.policy2 = nn.Linear(32, 16)

            self.action = nn.Linear(16, 7)



        def forward(self, x):

            x = F.relu(self.conv1(x))

            x = F.relu(self.conv2(x))

            x = nn.Flatten()(x)

            x = F.relu(self.fc3(x))

            x = F.relu(self.shared1(x))

            x = F.relu(self.policy1(x))

            x = F.relu(self.policy2(x))

            x = self.action(x)

            x = x.argmax()

            return x
learner.policy.state_dict().keys()
th.set_printoptions(profile="full")



agent_path = 'submission.py'



state_dict = learner.policy.to('cpu').state_dict()

state_dict = {

    'conv1.weight': state_dict['features_extractor.conv1.weight'],

    'conv1.bias': state_dict['features_extractor.conv1.bias'],

    'conv2.weight': state_dict['features_extractor.conv2.weight'],

    'conv2.bias': state_dict['features_extractor.conv2.bias'],

    'fc3.weight': state_dict['features_extractor.fc3.weight'],

    'fc3.bias': state_dict['features_extractor.fc3.bias'],

    

    'shared1.weight': state_dict['mlp_extractor.shared_net.0.weight'],

    'shared1.bias': state_dict['mlp_extractor.shared_net.0.bias'],

    

    'policy1.weight': state_dict['mlp_extractor.policy_net.0.weight'],

    'policy1.bias': state_dict['mlp_extractor.policy_net.0.bias'],

    'policy2.weight': state_dict['mlp_extractor.policy_net.2.weight'],

    'policy2.bias': state_dict['mlp_extractor.policy_net.2.bias'],

    

    'action.weight': state_dict['action_net.weight'],

    'action.bias': state_dict['action_net.bias'],

}



with open(agent_path, mode='a') as file:

    #file.write(f'\n    data = {learner.policy._get_data()}\n')

    file.write(f'    state_dict = {state_dict}\n')
%%writefile -a submission.py



    model = Net()

    model = model.float()

    model.load_state_dict(state_dict)

    model = model.to('cpu')

    model = model.eval()

    obs = tensor(obs['board']).reshape(1, 1, config.rows, config.columns).float()

    obs = obs / 2

    action = model(obs)

    return int(action)
# load submission.py

f = open(agent_path)

source = f.read()

exec(source)
agent(env.reset()[0]['observation'], env.configuration)
get_win_percentages(agent1=agent, agent2="random")
env = make("connectx", debug=True)



# Two random agents play one game round

env.run([agent, "random"])



# Show the game

env.render(mode="ipython")