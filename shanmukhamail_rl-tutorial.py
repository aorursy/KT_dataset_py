# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shutil
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install 'gym[box2d]'
!apt-get install python-opengl -y
!apt install xvfb -y
!pip install pyvirtualdisplay
!pip install https://github.com/pyglet/pyglet/archive/pyglet-1.5-maintenance.zip
!apt-get install ffmpeg -y
from pyvirtualdisplay import Display
import gym
from gym import wrappers
from gym import envs
import matplotlib.pyplot as plt
display = Display(visible=0,size=(600,600))
display.start()
env = gym.make('MountainCar-v0')
# state = env.reset()
monitor_dir = os.getcwd()
env = wrappers.Monitor(env,monitor_dir,video_callable=lambda ep_id: ep_id%1000 == 0,force=True)
print(env.action_space.n)
L_R = 0.1
Disc = 0.95
Epis = 4001
epsilon = 1
start_decay = 1
end_decay = Epis//2
epsilon_decay = epsilon/(end_decay - start_decay)
stats_for = 100
epi_rewards = []
reward_stats = {'epi':[],'avg_r':[],'max_r':[],'min_r':[]}
print(env.observation_space.high)
print(env.observation_space.low)
buckets_shape = [20,20]
bucket_range = (env.observation_space.high - env.observation_space.low)/buckets_shape
print(bucket_range)
q_table = np.random.uniform(low=-2,high=0,size=(buckets_shape + [env.action_space.n]))
print(q_table.shape)
def dis_from_cont(cont_state):
    state = (cont_state - env.observation_space.low)/bucket_range
    return tuple(state.astype(np.int))
# env.reset()
for ep in range(Epis):
    epi_r = 0
    dis_state = dis_from_cont(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[dis_state])
        else:
            action = np.random.randint(0,env.action_space.n)
    #     action = env.action_space.sample()
        new_state,reward,done,_ = env.step(action)
        epi_r += reward
        new_dis_state = dis_from_cont(new_state)
        if not done:
            #update q
            max_future_q = np.max(q_table[new_dis_state])
            curr_q = q_table[dis_state + (action,)]
            new_q = (1 - L_R)*curr_q + L_R*(reward + Disc*max_future_q)
            q_table[dis_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[dis_state + (action,)] = reward
    #     print(action,state,reward,done,_)
        dis_state = new_dis_state
    epi_rewards.append(epi_r)
    #track metrics for rewards
    if ep%stats_for == 0:
        reward_stats['epi'].append(ep)
        avg_r = sum(epi_rewards[-stats_for:])/stats_for
        reward_stats['avg_r'].append(avg_r)
        reward_stats['max_r'].append(max(epi_rewards[-stats_for:]))
        reward_stats['min_r'].append(min(epi_rewards[-stats_for:]))
        print(f"Episode {ep}, avg reward {avg_r:.1f}, epsilon {epsilon:.2f}")
    if start_decay <= ep <= end_decay:
        epsilon -= epsilon_decay
    
env.close()
from IPython.display import HTML
from base64 import b64encode

video = [v for v in os.listdir('./') if 'mp4' in v]
video.sort()
print(len(video))
# print(video[:26])
vid_1 = open(video[0],'rb').read()
data_url_1 = "data:video/mp4;base64," + b64encode(vid_1).decode()
HTML("""
<video width=600 height=600 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url_1)
vid_2 = open(video[-1],'rb').read()
data_url_2 = "data:video/mp4;base64," + b64encode(vid_2).decode()
HTML("""
<video width=600 height=600 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url_2)
plt.plot(reward_stats['epi'],reward_stats['avg_r'],label='average reward')
plt.plot(reward_stats['epi'],reward_stats['max_r'],label='max reward')
plt.plot(reward_stats['epi'],reward_stats['min_r'],label='min reward')
plt.grid(True)
plt.legend(loc=2)
plt.show()