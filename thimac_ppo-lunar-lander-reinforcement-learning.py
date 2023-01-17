! apt update

! apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb

! pip install git+https://github.com/openai/gym.git#egg=gym[box2d]

! pip install xvfbwrapper



from IPython.display import clear_output



clear_output()



from xvfbwrapper import Xvfb

vdisplay = Xvfb(width=1280, height=740)

vdisplay.start()



import numpy as np

import random

import matplotlib.pyplot as plt

import matplotlib.animation as animation

from matplotlib import animation, rc

from IPython.display import Math, HTML



from pylab import rcParams



rcParams['figure.figsize'] = 5, 3



import gym



def render_frames(env, num_frame=50):

    env.reset()

    frames = []

    for i in range(num_frame):

        _, _, done, _ = env.step( env.action_space.sample() )

        if done:

            env.reset()        

        frames.append(  env.render(mode="rgb_array") )

        

    return frames



def create_animation(frames):

    rc('animation', html='jshtml')

    fig = plt.figure()

    plt.axis("off")

    im = plt.imshow(frames[0], animated=True)



    def updatefig(i):

        im.set_array(frames[i])

        return im,



    ani = animation.FuncAnimation(fig, updatefig, frames=len(frames), interval=len(frames)/10, blit=True)

    display(HTML(ani.to_html5_video()))    

    plt.close()    

    

    return ani
env = gym.make('LunarLander-v2')

game_obs = env.reset()



ani = create_animation(render_frames(env, 300))
%load_ext tensorboard.notebook 



! mkdir logs

! pip install tensorboardX



clear_output()
%tensorboard --logdir ./logs
γ = 0.99
env.observation_space.low, env.observation_space.high
import torch

import torch.nn as nn
class ValueFunction(nn.Module):

    def __init__(self, input_size):

        super().__init__()       

        

        self.V = nn.Sequential(

            nn.Linear(input_size, 128),

            nn.SELU(),

            nn.Linear(128, 256),

            nn.SELU(),

            nn.Linear(256, 1),

        )        

        

    def forward(self, x):

        v = self.V(x)

        return v
class SoftmaxPolicy(nn.Module):

    def __init__(self, input_size, num_actions):

        super().__init__()

        

        self.input_size = input_size

        self.num_actions = num_actions

        

        self.π = nn.Sequential(

            nn.Linear(input_size, 128),

            nn.SELU(),

            nn.Linear(128, 256),

            nn.SELU(),

            nn.Linear(256, 256),

            nn.SELU(),

            nn.Linear(256, num_actions),

            nn.LogSoftmax(dim=1)

        )

        

    def forward(self, x):

        logits = self.π(x)

        

        return logits
cc = 0
from torch.distributions.categorical import Categorical

def sample_action(logits):

    return Categorical(logits=logits).sample()



obs = env.reset()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



import itertools



from tensorboardX import SummaryWriter



cc = cc + 1

writer = SummaryWriter("./logs/{}".format(cc))



step = 0



my_π = SoftmaxPolicy(input_size=8, num_actions=4).to(device)

my_π_old = SoftmaxPolicy(input_size=8, num_actions=4).to(device)

my_V = ValueFunction( input_size=8 ).to(device)





state = torch.from_numpy(obs).float()

logits = my_π(state.unsqueeze(0).to(device))

logits.squeeze(0)



sample_action(logits.squeeze(0).cpu())
def roll_out(env, π):

    obs = env.reset()

    

    memory = []

    rewards = []

    states = []

    actions = []

    logps = []

    π.eval()

    with torch.no_grad():

        for i in itertools.count():

            state = torch.from_numpy(obs).float()

            logits = π(state.unsqueeze(0).to(device))

            act = sample_action(logits.cpu().squeeze())

            obs, reward, is_done, info = env.step(act.item())

            

#             reward = 1.0

            states.append(state)

            actions.append(act)

            rewards.append(torch.tensor(reward/200.0, dtype=torch.float)) # .tanh())

            logps.append(logits[0,act.item()])

            

            if is_done:

                break

                

    values = []

    acc = 0.0

#     c = 0

    for r in reversed(rewards):

#         c += 1

        acc = γ*acc + r

        values.insert(0, acc )



    return zip(states, actions, values, logps)



from torch.utils.data import DataLoader



params = itertools.chain(my_π.parameters(), my_V.parameters() )

optim = torch.optim.Adam(params,  lr=1e-4)



import math



from torch.nn.utils import clip_grad_value_
eps = 0.2



for loop in range(800):

    experiences = []



    for episole in range(30):

        experiences.extend( roll_out(env, my_π) )

    

    dataloader = DataLoader(experiences, batch_size=32, shuffle=True)



    my_V.train()

        

    my_π.train()

    

    my_π_old.load_state_dict( my_π.state_dict() )

    my_π_old.eval()

    

    for state, action, value, logp_old in dataloader:

        step += 1

        ii = state.to(device)

        ### V loss

        vv = my_V(ii)

        vv_loss = nn.functional.mse_loss(input=vv, target =value.unsqueeze(1).to(device) )



        writer.add_scalar("V_loss", vv_loss.item(), step)        

        

        ## Policy gradient

        

        logits = my_π(ii)

        logp = logits.gather(dim=1, index=action.unsqueeze(1).to(device)) 

        with torch.no_grad():

#             logits_old = my_π_old(ii)

#             logp_old = logits_old.gather(dim=1, index=action.unsqueeze(1).to(device)) 

            advantage = value.unsqueeze(1).to(device) - vv

            

        ratio = (logp - logp_old.unsqueeze(1).to(device) ).exp()

        

        entropy = - logits * logits.exp()

        pi_loss = - torch.min( ratio * advantage, torch.clamp(ratio, min=1.0 - eps, max=1.0 + eps) * advantage ) - 1e-4 * entropy



        pi_loss = pi_loss.mean()

                

        loss = pi_loss + vv_loss

    

        writer.add_scalar("mean reward", value.mean().item(), step)

        writer.add_scalar("pi_loss", pi_loss.item(), step)

        writer.add_scalar("loss", loss.item(), step)

        

        optim.zero_grad()

        loss.backward()



        clip_grad_value_( params , 1.0)

        optim.step()

loop
obs = env.reset()
def render_frames_with_env(env, π):



    frames = []

    

    π.eval()

    for i in range(10):

        obs = env.reset()

        with torch.no_grad():

            for i in itertools.count():

                state = torch.from_numpy(obs).float()

                logits = π(state.unsqueeze(0).to(device))

                act = sample_action(logits.cpu().squeeze())

                obs, reward, is_done, info = env.step(act.item())



                frames.append(  env.render(mode="rgb_array") )



                if is_done:

                    break

                

    return frames



def create_animation(frames):

    rc('animation', html='jshtml')

    fig = plt.figure()

    plt.axis("off")

    im = plt.imshow(frames[0], animated=True)



    def updatefig(i):

        im.set_array(frames[i])

        return im,



    ani = animation.FuncAnimation(fig, updatefig, frames=len(frames), interval=20, blit=True)

    display(HTML(ani.to_html5_video()))    

    plt.close()    

    

    return ani
ani = create_animation(render_frames_with_env(env, my_π))