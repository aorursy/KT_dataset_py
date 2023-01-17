!pip3 install kaggle --upgrade > /dev/null 2>&1
!pip install 'kaggle-environments>=0.1.6' > /dev/null 2>&1
!pip install 'tensorflow-gpu == 1.14.0' > /dev/null 2>&1
!pip install stable-baselines > /dev/null 2>&1
from kaggle_environments import make

import gym

import numpy as np



class ConnectX(gym.Env):

    """Custom Environment that follows gym interface"""

    

    def __init__(self, opponent_type):

        self.env = make("connectx", debug=True)

        self.trainer = self.env.train([None, opponent_type])

        self.obs = None

        self.action_space = gym.spaces.Discrete(self.env.configuration.columns)

        self.observation_space = gym.spaces.Box(0, 2, shape=(self.env.configuration.rows, self.env.configuration.columns), dtype=np.float32)



    def get_kaggle_env(self):

        return self.env



    def step(self, action):

        # Wrap kaggle environment.step()

        if self.obs[0][action] != 0:

          r = -1 # punish illegal move

          d = False

          o = self.obs

        else:

          o, r, d, _ = self.trainer.step(int(action))

          o = np.reshape(np.array(o['board']), (self.env.configuration.rows, self.env.configuration.columns))

          self.obs = o



        return o, float(r), bool(d), {}

    

    def reset(self):        

        o = self.trainer.reset()

        self.obs = np.reshape(np.array(o['board']), (self.env.configuration.rows, self.env.configuration.columns))

        return self.obs



    def render(self, **kwargs):

        return self.env.render(**kwargs)

    
import gym



from stable_baselines.common.policies import MlpPolicy

from stable_baselines import PPO2



# from stable_baselines.deepq.policies import MlpPolicy

# from stable_baselines import DQN



from stable_baselines.common.vec_env import DummyVecEnv



from stable_baselines.bench import Monitor

from stable_baselines.results_plotter import load_results, ts2xy



# Create log dir

import os

log_dir = "/kaggle/working/"

# os.makedirs(log_dir, exist_ok=True)



def callback(_locals, _globals):

    """

    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)

    :param _locals: (dict)

    :param _globals: (dict)

    """

    global best_mean_reward

    # Evaluate policy training performance

    x, y = ts2xy(load_results(log_dir), 'timesteps')

    if len(x) > 0:

        mean_reward = np.mean(y[-100:])

        print(x[-1], 'timesteps')

        print("Best / last mean reward per episode: {:.2f} / {:.2f}".format(best_mean_reward, mean_reward))



        # New best model, you could save the agent here

        if mean_reward > best_mean_reward:

            best_mean_reward = mean_reward

            # Example for saving best model

            print("*** Saving new best model ***")

            _locals['self'].save(log_dir + 'best_model.pkl')

    return True



best_mean_reward = -1000000



# gym_env = ConnectX('random')

gym_env = ConnectX('negamax')



env = Monitor(gym_env, log_dir, allow_early_resets=True)



env = DummyVecEnv([lambda: env])

model = PPO2('MlpPolicy', env)

# model = DQN('MlpPolicy', env)



# model.learn(total_timesteps=100000, callback=callback) #, seed=42)

model.learn(total_timesteps=10000, callback=callback) #, seed=42)
model = PPO2.load(log_dir + 'best_model.pkl', env)



done = False

obs = env.reset()

step_cnt = 0



max_moves = gym_env.get_kaggle_env().configuration.columns * gym_env.get_kaggle_env().configuration.rows



while (not done) and step_cnt <= max_moves:

      step_cnt += 1

      action, _states = model.predict(obs, deterministic=True)

      print('action:', action)

      if obs[0][0][action] != 0:

        print('skipping illegal move')

      else:

        obs, reward, done, info = env.step(action)

        gym_env.render()

        print(reward, done)

      print()

for key, value in model.get_parameters().items():

    print(key, value.shape)
import torch as th

import torch.nn as nn



class PyTorchMlpPolicy(nn.Module):

    def __init__(self):

        super(PyTorchMlpPolicy, self).__init__()

        self.pi_fc0 = nn.Linear(42, 64)

        self.pi_fc1 = nn.Linear(64, 64)

        self.pi = nn.Linear(64, 7)

        

        self.tanh = th.tanh

        self.out_activ = nn.Softmax(dim=0)



    def forward(self, x):

        x = self.tanh(self.pi_fc0(x))

        x = self.tanh(self.pi_fc1(x))

        x = self.pi(x)

        x = self.out_activ(x)

        return x
def copy_mlp_weights(baselines_model):

    torch_mlp = PyTorchMlpPolicy()

    model_params = baselines_model.get_parameters()

    # Get only the policy parameters

    policy_keys = [key for key in model_params.keys() if "pi" in key] # or "c" in key]

    policy_params = [model_params[key] for key in policy_keys]

    

    for (th_key, pytorch_param), key, policy_param in zip(torch_mlp.named_parameters(), policy_keys, policy_params):

        param = policy_param.copy()

        # Copy parameters from stable baselines model to pytorch model



        # weight of fully connected layer

        if len(param.shape) == 2:

            param = param.T



        # bias

        if 'b' in key:

            param = param.squeeze()



        param = th.from_numpy(param)

        pytorch_param.data.copy_(param.data.clone())

        

    return torch_mlp

th_model = copy_mlp_weights(model)
import torch

from torch.autograd import Variable



episode_reward = 0

done = False

obs = env.reset()

step_cnt = 0

max_moves = gym_env.get_kaggle_env().configuration.columns * gym_env.get_kaggle_env().configuration.rows



while (not done) and step_cnt <= max_moves:

      step_cnt += 1

      th_obs = Variable(torch.from_numpy(obs.flatten()))

      action = th.argmax(th_model(th_obs)).item()



      print('action:', action)

      if obs[0][0][action] != 0:

        print('skipping illegal move')

      else:

        obs, reward, done, info = env.step([action])

        gym_env.render()

        episode_reward += reward

      print()

torch.save(th_model.state_dict(), 'thmodel')
import base64

with open('thmodel', 'rb') as f:

    raw_bytes = f.read()

    encoded_weights = base64.encodebytes(raw_bytes)
print(encoded_weights)
import io

import base64

import torch

from torch.autograd import Variable

import random



agent_th_model = PyTorchMlpPolicy()

# encoded_weights =b'gAKKCmz8n ..... [long string]

decoded = base64.b64decode(encoded_weights)

buffer = io.BytesIO(decoded)

agent_th_model.load_state_dict(torch.load(buffer))
def my_agent(observation, configuration):

      obs = np.array(observation['board'])

      th_obs = Variable(torch.from_numpy(obs)).float()

      y = agent_th_model(th_obs)

      action = th.argmax(agent_th_model(th_obs)).item()

      if observation.board[action] == 0:

          return action

      else:

          return random.choice([c for c in range(configuration.columns) if observation.board[c] == 0])

kaggle_env = gym_env.get_kaggle_env()

kaggle_env.reset()

kaggle_env.run([my_agent, "negamax"])

kaggle_env.render(mode="ipython", width=500, height=450)
from kaggle_environments import evaluate



def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



# Run multiple episodes to estimate its performance.

print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))

print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
import inspect

import os



def write_agent_to_file(file):

#     with open(file, "a" if os.path.exists(file) else "w") as f:

    with open(file, "w") as f:

        f.write('import numpy as np\n')

        f.write('import random\n')

        f.write('import torch as th\n')

        f.write('import torch.nn as nn\n')

        f.write('import io\n')

        f.write('import base64\n')

        f.write('import torch\n')

        f.write('from torch.autograd import Variable\n')



        f.write('class PyTorchMlpPolicy(nn.Module):\n')

        f.write('    def __init__(self):\n')

        f.write('        super(PyTorchMlpPolicy, self).__init__()\n')

        f.write('        self.pi_fc0 = nn.Linear(42, 64)\n')

        f.write('        self.pi_fc1 = nn.Linear(64, 64)\n')

        f.write('        self.pi = nn.Linear(64, 7)\n') 

        f.write('        self.tanh = th.tanh\n')

        f.write('        self.out_activ = nn.Softmax(dim=0)\n')

        f.write('    def forward(self, x):\n')

        f.write('        x = self.tanh(self.pi_fc0(x))\n')

        f.write('        x = self.tanh(self.pi_fc1(x))\n')

        f.write('        x = self.pi(x)\n')

        f.write('        x = self.out_activ(x)\n')

        f.write('        return x\n')



        f.write('agent_th_model = PyTorchMlpPolicy()\n')

        f.write('encoded_weights =' + str(encoded_weights) + '\n')

        f.write('decoded = base64.b64decode(encoded_weights)\n')

        f.write('buffer = io.BytesIO(decoded)\n')

        f.write('agent_th_model.load_state_dict(torch.load(buffer))\n')

        

        f.write(inspect.getsource(my_agent))



write_agent_to_file("submission.py")

# Note: Stdout replacement is a temporary workaround.

import sys

out = sys.stdout

from kaggle_environments import utils

submission = utils.read_file("/kaggle/working/submission.py")

submission_agent = utils.get_last_callable(submission)

sys.stdout = out



kaggle_env.run([submission_agent, submission_agent])

print("Success!" if kaggle_env.state[0].status == kaggle_env.state[1].status == "DONE" else "Failed...")



kaggle_env.play([submission_agent, None])