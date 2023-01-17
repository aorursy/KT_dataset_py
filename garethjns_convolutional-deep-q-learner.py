# GFootball environment.

!pip install kaggle_environments

!apt-get update -y

!apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev

!git clone -b v2.3 https://github.com/google-research/football.git

!mkdir -p football/third_party/gfootball_engine/lib

!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.3.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so

!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .



# Some helper code

!git clone https://github.com/garethjns/kaggle-football.git

!pip install reinforcement_learning_keras==0.6.0
import collections

from typing import Union, Callable, List, Tuple, Iterable, Any, Dict

from dataclasses import dataclass

from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np

from tensorflow import keras

import tensorflow as tf

import seaborn as sns

import gym

import gfootball

import glob 

import imageio

import pathlib

import zlib

import pickle

import tempfile

import os

import sys

from IPython.display import Image, display



sns.set()



# In TF > 2, training keras models in a loop with eager execution on causes memory leaks and terrible performance.

tf.compat.v1.disable_eager_execution()



sys.path.append("/kaggle/working/kaggle-football/")
from kaggle_football.viz import generate_gif, plot_smm_obs





smm_env = gym.make("GFootball-11_vs_11_kaggle-SMM-v0")

print(smm_env.reset().shape)



generate_gif(smm_env, n_steps=500, expected_min=0, expected_max=255)

Image(filename='smm_env_replay.gif', format='png')
class SMMFrameProcessWrapper(gym.Wrapper):

    """

    Wrapper for processing frames from SMM observation wrapper from football env.



    Input is (72, 96, 4), where last dim is (team 1 pos, team 2 pos, ball pos, 

    active player pos). Range 0 -> 255.

    Output is (72, 96, 4) as difference to last frame for all. Range -1 -> 1

    """



    def __init__(self, env: gym.Env = None,

                 obs_shape: Tuple[int, int] = (72, 96, 4)) -> None:

        """

        :param env: Gym env, or None. Allowing None here is unusual,

                    but we'll reuse the buffer functunality later in

                    the submission, when we won't be using the gym API.

        :param obs_shape: Expected shape of single observation.

        """

        if env is not None:

            super().__init__(env)

        self._buffer_length = 2

        self._obs_shape = obs_shape

        self._prepare_obs_buffer()



    @staticmethod

    def _normalise_frame(frame: np.ndarray):

        return frame / 255.0



    def _prepare_obs_buffer(self) -> None:

        """Create buffer and preallocate with empty arrays of expected shape."""



        self._obs_buffer = collections.deque(maxlen=self._buffer_length)



        for _ in range(self._buffer_length):

            self._obs_buffer.append(np.zeros(shape=self._obs_shape))



    def build_buffered_obs(self) -> np.ndarray:

        """

        Iterate over the last dimenion, and take the difference between this obs 

        and the last obs for each.

        """

        agg_buff = np.empty(self._obs_shape)

        for f in range(self._obs_shape[-1]):

            agg_buff[..., f] = self._obs_buffer[1][..., f] - self._obs_buffer[0][..., f]



        return agg_buff



    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:

        """Step env, add new obs to buffer, return buffer."""

        obs, reward, done, info = self.env.step(action)



        obs = self._normalise_frame(obs)

        self._obs_buffer.append(obs)



        return self.build_buffered_obs(), reward, done, info



    def reset(self) -> np.ndarray:

        """Add initial obs to end of pre-allocated buffer.



        :return: Buffered observation

        """

        self._prepare_obs_buffer()

        obs = self.env.reset()

        self._obs_buffer.append(obs)



        return self.build_buffered_obs()
smm_env = gym.make("GFootball-11_vs_11_kaggle-SMM-v0")

wrapped_smm_env = SMMFrameProcessWrapper(smm_env)

print(wrapped_smm_env.reset().shape)
generate_gif(wrapped_smm_env, n_steps=500, suffix="wrapped_smm_env_", expected_min=-1, expected_max=1)

Image(filename='wrapped_smm_env_replay.gif', format='png')
class SplitLayer(keras.layers.Layer):

    def __init__(self, split_dim: int = 3) -> None:

        super().__init__()

        self.split_dim = split_dim



    def call(self, inputs) -> tf.Tensor:

        """Split a given dim into seperate tensors."""

        return [tf.expand_dims(inputs[..., i], self.split_dim) 

                for i in range(inputs.shape[self.split_dim])]
class SplitterConvNN:



    def __init__(self, observation_shape: List[int], n_actions: int, 

                 output_activation: Union[None, str] = None,

                 unit_scale: int = 1, learning_rate: float = 0.0001, 

                 opt: str = 'Adam') -> None:

        """

        :param observation_shape: Tuple specifying input shape.

        :param n_actions: Int specifying number of outputs

        :param output_activation: Activation function for output. Eg. 

                                  None for value estimation (off-policy methods).

        :param unit_scale: Multiplier for all units in FC layers in network 

                           (not used here at the moment).

        :param opt: Keras optimiser to use. Should be string. 

                    This is to avoid storing TF/Keras objects here.

        :param learning_rate: Learning rate for optimiser.



        """

        self.observation_shape = observation_shape

        self.n_actions = n_actions

        self.unit_scale = unit_scale

        self.output_activation = output_activation

        self.learning_rate = learning_rate

        self.opt = opt



    @staticmethod

    def _build_conv_branch(frame: keras.layers.Layer, name: str) -> keras.layers.Layer:

        conv1 = keras.layers.Conv2D(16, kernel_size=(8, 8), strides=(4, 4),

                                    name=f'conv1_frame_{name}', padding='same', 

                                    activation='relu')(frame)

        conv2 = keras.layers.Conv2D(24, kernel_size=(4, 4), strides=(2, 2),

                                    name=f'conv2_frame_{name}', padding='same', 

                                    activation='relu')(conv1)

        conv3 = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),

                                    name=f'conv3_frame_{name}', padding='same', 

                                    activation='relu')(conv2)



        flatten = keras.layers.Flatten(name=f'flatten_{name}')(conv3)



        return flatten



    def _model_architecture(self) -> Tuple[keras.layers.Layer, keras.layers.Layer]:

        n_units = 512 * self.unit_scale



        frames_input = keras.layers.Input(name='input', shape=self.observation_shape)

        frames_split = SplitLayer(split_dim=3)(frames_input)

        conv_branches = []

        for f, frame in enumerate(frames_split):

            conv_branches.append(self._build_conv_branch(frame, name=str(f)))



        concat = keras.layers.concatenate(conv_branches)

        fc1 = keras.layers.Dense(units=int(n_units), name='fc1', 

                                 activation='relu')(concat)

        fc2 = keras.layers.Dense(units=int(n_units / 2), name='fc2', 

                                 activation='relu')(fc1)

        action_output = keras.layers.Dense(units=self.n_actions, name='output',

                                           activation=self.output_activation)(fc2)



        return frames_input, action_output



    def compile(self, model_name: str = 'model', 

                loss: Union[str, Callable] = 'mse') -> keras.Model:

        """

        Compile a copy of the model using the provided loss.



        :param model_name: Name of model

        :param loss: Model loss. Default 'mse'. Can be custom callable.

        """

        # Get optimiser

        if self.opt.lower() == 'adam':

            opt = keras.optimizers.Adam

        elif self.opt.lower() == 'rmsprop':

            opt = keras.optimizers.RMSprop

        else:

            raise ValueError(f"Invalid optimiser {self.opt}")



        state_input, action_output = self._model_architecture()

        model = keras.Model(inputs=[state_input], outputs=[action_output], 

                            name=model_name)

        model.compile(optimizer=opt(learning_rate=self.learning_rate), 

                      loss=loss)



        return model



    def plot(self, model_name: str = 'model') -> None:

        keras.utils.plot_model(self.compile(model_name), 

                               to_file=f"{model_name}.png", show_shapes=True)

        plt.show()





mod = SplitterConvNN(observation_shape=wrapped_smm_env.observation_space.shape, 

                     n_actions=wrapped_smm_env.action_space.n)

mod.compile()

mod.plot()

Image(filename='model.png') 
from reinforcement_learning_keras.agents.components.history.training_history import TrainingHistory

from reinforcement_learning_keras.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer

from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent

from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy



agent = DeepQAgent(

    name='deep_q',

    model_architecture=SplitterConvNN(observation_shape=(72, 96, 4), 

                                      n_actions=19),

    replay_buffer=ContinuousBuffer(buffer_size=300),

    env_spec="GFootball-11_vs_11_kaggle-SMM-v0",

    env_wrappers=[SMMFrameProcessWrapper],

    eps=EpsilonGreedy(eps_initial=0.5, 

                      decay=0.001, 

                      eps_min=0.01, 

                      decay_schedule='linear'),

    training_history=TrainingHistory(agent_name='deep_q', 

                                     plotting_on=True, 

                                     plot_every=5, 

                                     rolling_average=5)

)



agent.train(verbose=True, render=False,

            n_episodes=2, max_episode_steps=100, 

            update_every=10, checkpoint_every=10)
agent._action_model.save_weights("saved_model")

!ls
%%writefile main.py





import collections

import pickle

import zlib

from typing import Tuple, Dict, Any, Union, Callable, List



import gym

import numpy as np

import tensorflow as tf

from gfootball.env import observation_preprocessing

from tensorflow import keras





class SMMFrameProcessWrapper(gym.Wrapper):

    """

    Wrapper for processing frames from SMM observation wrapper from football env.



    Input is (72, 96, 4), where last dim is (team 1 pos, team 2 pos, ball pos,

    active player pos). Range 0 -> 255.

    Output is (72, 96, 4) as difference to last frame for all. Range -1 -> 1

    """



    def __init__(self, env: Union[None, gym.Env] = None,

                 obs_shape: Tuple[int, int] = (72, 96, 4)) -> None:

        """

        :param env: Gym env.

        :param obs_shape: Expected shape of single observation.

        """

        if env is not None:

            super().__init__(env)

        self._buffer_length = 2

        self._obs_shape = obs_shape

        self._prepare_obs_buffer()



    @staticmethod

    def _normalise_frame(frame: np.ndarray):

        return frame / 255.0



    def _prepare_obs_buffer(self) -> None:

        """Create buffer and preallocate with empty arrays of expected shape."""



        self._obs_buffer = collections.deque(maxlen=self._buffer_length)



        for _ in range(self._buffer_length):

            self._obs_buffer.append(np.zeros(shape=self._obs_shape))



    def build_buffered_obs(self) -> np.ndarray:

        """

        Iterate over the last dimenion, and take the difference between this obs

        and the last obs for each.

        """

        agg_buff = np.empty(self._obs_shape)

        for f in range(self._obs_shape[-1]):

            agg_buff[..., f] = self._obs_buffer[1][..., f] - self._obs_buffer[0][..., f]



        return agg_buff



    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:

        """Step env, add new obs to buffer, return buffer."""

        obs, reward, done, info = self.env.step(action)



        obs = self._normalise_frame(obs)

        self._obs_buffer.append(obs)



        return self.build_buffered_obs(), reward, done, info



    def reset(self) -> np.ndarray:

        """Add initial obs to end of pre-allocated buffer.



        :return: Buffered observation

        """

        self._prepare_obs_buffer()

        obs = self.env.reset()

        self._obs_buffer.append(obs)



        return self.build_buffered_obs()



    

    

class SplitLayer(keras.layers.Layer):

    def __init__(self, split_dim: int = 3) -> None:

        super().__init__()

        self.split_dim = split_dim



    def call(self, inputs) -> tf.Tensor:

        """Split a given dim into seperate tensors."""

        return [tf.expand_dims(inputs[..., i], self.split_dim) 

                for i in range(inputs.shape[self.split_dim])]



    

class SplitterConvNN:



    def __init__(self, observation_shape: List[int], n_actions: int, 

                 output_activation: Union[None, str] = None,

                 unit_scale: int = 1, learning_rate: float = 0.0001, 

                 opt: str = 'Adam') -> None:

        """

        :param observation_shape: Tuple specifying input shape.

        :param n_actions: Int specifying number of outputs

        :param output_activation: Activation function for output. Eg. 

                                  None for value estimation (off-policy methods).

        :param unit_scale: Multiplier for all units in FC layers in network 

                           (not used here at the moment).

        :param opt: Keras optimiser to use. Should be string. 

                    This is to avoid storing TF/Keras objects here.

        :param learning_rate: Learning rate for optimiser.



        """

        self.observation_shape = observation_shape

        self.n_actions = n_actions

        self.unit_scale = unit_scale

        self.output_activation = output_activation

        self.learning_rate = learning_rate

        self.opt = opt



    @staticmethod

    def _build_conv_branch(frame: keras.layers.Layer, name: str) -> keras.layers.Layer:

        conv1 = keras.layers.Conv2D(16, kernel_size=(8, 8), strides=(4, 4),

                                    name=f'conv1_frame_{name}', padding='same', 

                                    activation='relu')(frame)

        conv2 = keras.layers.Conv2D(24, kernel_size=(4, 4), strides=(2, 2),

                                    name=f'conv2_frame_{name}', padding='same', 

                                    activation='relu')(conv1)

        conv3 = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),

                                    name=f'conv3_frame_{name}', padding='same', 

                                    activation='relu')(conv2)



        flatten = keras.layers.Flatten(name=f'flatten_{name}')(conv3)



        return flatten



    def _model_architecture(self) -> Tuple[keras.layers.Layer, keras.layers.Layer]:

        n_units = 512 * self.unit_scale



        frames_input = keras.layers.Input(name='input', shape=self.observation_shape)

        frames_split = SplitLayer(split_dim=3)(frames_input)

        conv_branches = []

        for f, frame in enumerate(frames_split):

            conv_branches.append(self._build_conv_branch(frame, name=str(f)))



        concat = keras.layers.concatenate(conv_branches)

        fc1 = keras.layers.Dense(units=int(n_units), name='fc1', 

                                 activation='relu')(concat)

        fc2 = keras.layers.Dense(units=int(n_units / 2), name='fc2', 

                                 activation='relu')(fc1)

        action_output = keras.layers.Dense(units=self.n_actions, name='output',

                                           activation=self.output_activation)(fc2)



        return frames_input, action_output



    def compile(self, model_name: str = 'model', 

                loss: Union[str, Callable] = 'mse') -> keras.Model:

        """

        Compile a copy of the model using the provided loss.



        :param model_name: Name of model

        :param loss: Model loss. Default 'mse'. Can be custom callable.

        """

        # Get optimiser

        if self.opt.lower() == 'adam':

            opt = keras.optimizers.Adam

        elif self.opt.lower() == 'rmsprop':

            opt = keras.optimizers.RMSprop

        else:

            raise ValueError(f"Invalid optimiser {self.opt}")



        state_input, action_output = self._model_architecture()

        model = keras.Model(inputs=[state_input], outputs=[action_output], 

                            name=model_name)

        model.compile(optimizer=opt(learning_rate=self.learning_rate), 

                      loss=loss)



        return model



    def plot(self, model_name: str = 'model') -> None:

        keras.utils.plot_model(self.compile(model_name), 

                               to_file=f"{model_name}.png", show_shapes=True)

        plt.show()

    

   

tf_mod = SplitterConvNN(observation_shape=(72, 96, 4), n_actions=19).compile()

try:

    # For evaulation The .tar.gz will be extracted to /kaggle_simulations/agent/

    tf_mod.load_weights("/kaggle_simulations/agent/saved_model")

except (FileNotFoundError, ValueError):

    # In notebook

    tf_mod.load_weights("saved_model")

    

obs_buffer = SMMFrameProcessWrapper()





def agent(obs):



    # Use the existing model and obs buffer on each call to agent

    global tf_mod

    global obs_buffer



    # Get the raw observations return by the environment

    obs = obs['players_raw'][0]

    # Convert these to the same output as the SMMWrapper we used in training

    obs = observation_preprocessing.generate_smm([obs])



    # Use the SMMFrameProcessWrapper to do the buffering, but not enviroment

    # stepping or anything related to the Gym API.

    obs_buffer._obs_buffer.append(obs)



    # Predict actions from keras model

    actions = tf_mod.predict(obs)

    action = np.argmax(actions)



    return [action]
!tar -czvf submission.tar.gz main.py saved_model*
from typing import Tuple, Dict, List, Any



from kaggle_environments import make



env = make("football", debug=True,configuration={"save_video": True,

                                                 "scenario_name": "11_vs_11_kaggle"})



# Define players

left_player = "main.py"  # A custom agent, eg. random_agent.py or example_agent.py

right_player = "run_right"  # eg. A built in 'AI' agent or the agent again





output: List[Tuple[Dict[str, Any], Dict[str, Any]]] = env.run([left_player, right_player])



print(f"Final score: {sum([r['reward'] for r in output[0]])} : {sum([r['reward'] for r in output[1]])}")

env.render(mode="human", width=800, height=600)