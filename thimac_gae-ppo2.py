%load_ext autoreload

%autoreload
from IPython.display import clear_output



! apt update

! apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb

! pip install -U git+https://github.com/openai/gym.git

! pip install -U git+https://github.com/cgoldberg/xvfbwrapper.git    



! mkdir logs

! pip install tensorboardX



clear_output()
import torch

import torch.nn as nn

import random

import numpy as np

from tensorboardX import SummaryWriter

import gym

import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace as st



random.seed(42)

torch.manual_seed(42)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

np.random.seed(42)
writer = SummaryWriter(log_dir="./logs/1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



env = gym.make('CartPole-v0')

env.seed(42)

env.reset()
from multiprocessing import Process

import matplotlib.pyplot as plt



! pkill Xvfb

! nohup Xvfb -screen 0 800x600x24 &



import os

os.environ['DISPLAY'] = ":0"



def show(env):

    obs = env.render("rgb_array")

    print(" ")

    plt.imshow(obs)

    plt.axis("off")

    plt.show()

    

p = Process(target=show, args=(env, ) )

p.start()

p.join()    
# env.action_space.low, env.action_space.high

env.action_space.sample()
env.step( env.action_space.sample() )
%load_ext tensorboard.notebook 

%tensorboard --logdir ./logs
class SoftmaxPolicyNet(nn.Module):

    def __init__(self, input_size, num_actions):

        super().__init__()

        

        self.num_actions = num_actions

        self.input_size = input_size

        

        self.π = nn.Sequential(

            nn.Linear(input_size, 32),

            nn.Tanh(),

            nn.Linear(32, 32),

            nn.Tanh(),

            nn.Linear(32, num_actions),

            nn.LogSoftmax(dim=1),

        )

            

    def forward(self, x):

        x = self.π(x)

        return x
class ValueNet(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        

        self.V_fn = nn.Sequential(

            nn.Linear(input_size, 32),

            nn.Tanh(),

            nn.Linear(32, 32),

            nn.Tanh(),

            nn.Linear(32, 1),

        )

            

    def forward(self, x):

        v = self.V_fn(x)

        return v
class RLAgent(nn.Module):

    def __init__(self, input_size, num_actions):

        super().__init__()

        

        self.policy_network = SoftmaxPolicyNet(input_size, num_actions)

        self.value_network = ValueNet(input_size)

        

    def forward(self, state):

        pr = self.policy_network(state)

        vl = self.value_network(state)

        

        return pr, vl

        
env.observation_space
num_recent_frames=4



γ = 0.99

λ = 0.95
agent = RLAgent(input_size=4*num_recent_frames, num_actions=2).to(device)
agent
def add_new_obs(recent_obs, obs):

    recent_obs.append(obs)

    recent_obs.pop(0)

    

def obs_to_input(obs):

    return torch.from_numpy(obs).float()



def frames_to_state(recent_obs):

    obs = list(map(obs_to_input, recent_obs))

    

    return torch.cat(obs, dim=0).view(-1)



def run_agent(agent, state):

    pr, vl = agent(state.unsqueeze(0))

    

    cat = torch.distributions.categorical.Categorical(pr)     

    action = cat.sample()

    

    logp = pr.gather(index=action.unsqueeze(0), dim=1)

    

    return action.item(), logp.item(), vl.item()

    

import itertools



def roll_out(agent, env):

    agent.eval()    

    

    rewards = []

    actions = []

    states  = [] 

    logps   = []

    values  = []

    

    recent_obs = [ np.zeros(4) for _ in range(num_recent_frames) ]    

    V = []

    

    with torch.no_grad():    

        obs = env.reset()

        add_new_obs(recent_obs, obs)



        for i in itertools.count():

            state = frames_to_state(recent_obs)

            states.append(state)



            action, logp, value = run_agent(agent, state.cpu())



            obs, reward, is_done, info = env.step(action)



            add_new_obs(recent_obs, obs)



            state = frames_to_state(recent_obs)

            rewards.append(reward)

            actions.append(action)

            values.append(value)

            logps.append(logp)



            if is_done:

                break



        acc = 0.0

        

        for r in reversed(rewards):

            acc = r + γ * acc

            V.insert(0, acc)

            

    return list( zip( map(lambda x: x.numpy(), states ), actions, V, logps, values) )


for i in range(100):

    

    exp = []



    cpu_agent = agent.cpu()

    

    for eps in range(10):

        exp.extend(roll_out(cpu_agent, env))

        

        

#     dataloader = torch.ut

    print(len(exp))
exp
def render_frames_with_env(agent, env):



    frames = []

    

    recent_frames = []

    

    π.eval()

    for i in range(10):

        

        recent_obs = [ np.zeros(4) for _ in range(num_recent_frames) ]    

        

        obs = env.reset()

        with torch.no_grad():

            for i in itertools.count():

                

                state = frames_to_state(recent_obs) 

                act, _, _ = run_agent(agent, state)

                obs, reward, is_done, info = env.step(act)



                frames.append(  env.render(mode="rgb_array") )



                if is_done:

                    break

                

    return frames





def create_animation(frames):

    

    from matplotlib import animation, rc

    from IPython.display import Math, HTML





    from pylab import rcParams



    rcParams['figure.figsize'] = 5, 3

    

    rc('animation', html='jshtml')

    fig = plt.figure()

    plt.axis("off")

    im = plt.imshow(frames[0], animated=True)



    def updatefig(i):

        im.set_array(frames[i])

        return im,



    ani = animation.FuncAnimation(fig, updatefig, frames=len(frames), interval=20, blit=True)

    print(" ")

    display(HTML(ani.to_html5_video()))    

    plt.close()    

    

    return ani





from multiprocessing import Process

import matplotlib.pyplot as plt



! pkill Xvfb

! nohup Xvfb -screen 0 800x600x24 &



import os

os.environ['DISPLAY'] = ":0"



def create_vid(agent, env):

    from IPython.display import Math, HTML

    ani = create_animation(render_frames_with_env(agent.cpu() , env))

#     display(ani)

    

p = Process(target=create_vid, args=(agent, env) )

p.start()

p.join()    
