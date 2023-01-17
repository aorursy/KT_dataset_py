!pip install tensorflow==1.13.1
# import modules

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import gym

%pylab inline
# hyperparam

gamma = 0.9

actor_lr = 1e-4

critic_lr = 2e-4

batch_size = 32

max_length_eps = 200

total_eps = 800

epsilon = 0.2

actor_update_steps = 10

critic_update_steps = 10
# config

env_name = 'Pendulum-v0'

env = gym.make(env_name).unwrapped

if_render = False

print(env.observation_space)

print(env.action_space)
dim_s = 3

dim_a = 1
class Agent:

    def __init__(self, actor_lr, critic_lr, batch_size, gamma, dim_s, dim_a, epsilon):

        self.actor_lr = actor_lr

        self.critic_lr = critic_lr

        self.batch_size = batch_size

        self.gamma = gamma

        self.dim_s = dim_s

        self.dim_a = dim_a

        self.epsilon = epsilon



        self.sess = tf.Session()

        self.build()

        self.sess.run(tf.global_variables_initializer())



    def build(self):

        # placeholders

        self.input_state = tf.placeholder(

            tf.float32, [None, self.dim_s], name='state')

        self.input_action = tf.placeholder(

            tf.float32, [None, self.dim_a], name='action')

        self.discounted_r = tf.placeholder(

            tf.float32, [None, 1], name='discounted_r')

        self.input_advantage = tf.placeholder(tf.float32, [None, 1], 'adv')





        # critic

        self.critic_layer_0 = tf.layers.dense(self.input_state, 64, tf.nn.relu)

        self.v = tf.layers.dense(self.critic_layer_0, 1)

        self.advantage = self.discounted_r - self.v

        self.loss_critic = tf.reduce_mean(tf.square(self.advantage))

        self.train_op_critic = tf.train.AdamOptimizer(self.critic_lr).minimize(self.loss_critic)



        # actor

        pi, pi_params = self.action_net('pi', True)

        pi_old, pi_old_params = self.action_net('pi_old', False)



        self.action_sample = tf.squeeze(pi.sample(1), axis=0)

        self.update_policy_op = [oldp.assign(

            p) for p, oldp in zip(pi_params, pi_old_params)]



        ratio = pi.prob(self.input_action) / pi_old.prob(self.input_action)

        surr = ratio * self.input_advantage

        self.loss_actor = -tf.reduce_mean(tf.minimum(

            surr,

            tf.clip_by_value(ratio, 1.-self.epsilon, 1.+self.epsilon)*self.input_advantage))

        self.train_op_actor = tf.train.AdamOptimizer(self.actor_lr).minimize(self.loss_actor)



    def learn(self, state, action, reward):

        state = np.vstack(state)

        action = np.vstack(action)

        reward = np.array(reward)[:, np.newaxis]

        self.sess.run(self.update_policy_op)

        adv = self.sess.run(self.advantage, feed_dict={

            self.input_state: state,

            self.discounted_r: reward

        })



        # update actor

        for _ in range(actor_update_steps):

            self.sess.run(self.train_op_actor, feed_dict={

                self.input_state: state,

                self.input_advantage: adv,

                self.input_action: action

            })

        

        # update critic

        for _ in range(critic_update_steps):

            self.sess.run(self.train_op_critic, feed_dict={

                self.input_state: state,

                self.discounted_r: reward

            })





    def act(self, state):

        state = state[np.newaxis, :]

        action = self.sess.run(self.action_sample, feed_dict={

            self.input_state: state

        })

        action = np.clip(action[0], -2, 2)

        return action



    def return_v(self, state):

        if state.ndim < 2:

            state = state[np.newaxis, :]

        return self.sess.run(self.v, {self.input_state: state})[0, 0]



    def action_net(self, name, trainable):

        with tf.variable_scope(name):

            l1 = tf.layers.dense(

                self.input_state, 100, tf.nn.relu, trainable=trainable)

            mu = 2 * tf.layers.dense(l1, self.dim_a,

                                     tf.nn.tanh, trainable=trainable)

            sigma = tf.layers.dense(

                l1, self.dim_a, tf.nn.softplus, trainable=trainable)

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return norm_dist, params
agent = Agent(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, gamma=gamma, dim_s=dim_s, dim_a=dim_a, epsilon=epsilon)

rewards_arr = []
for i in range(total_eps):

    # a new episode start

    s_t = env.reset()

    s_buffer, a_buffer, r_buffer = [], [], []

    eps_reward = 0



    for step in range(max_length_eps):

        if if_render:

            env.render()



        action = agent.act(s_t)

        s_tpo, r, done, _ = env.step(action)

        s_buffer.append(s_t)

        a_buffer.append(action)

        r_buffer.append((r + 8) / 8)

        s_t = s_tpo

        eps_reward += r



        # learn

        if step == max_length_eps-1 or (step+1) % batch_size == 0:

            value_s_tpo = agent.return_v(s_tpo)

            discounted_r = []

            for reward in r_buffer[::-1]:

                value_s_tpo = reward + gamma * value_s_tpo

                discounted_r.append(value_s_tpo)

            discounted_r.reverse()



            agent.learn(state=s_buffer, action=a_buffer, reward=discounted_r)

            s_buffer, a_buffer, r_buffer = [], [], []

    if i == 0:

        rewards_arr.append(eps_reward)

    else:

        rewards_arr.append(rewards_arr[-1]*0.9+eps_reward*0.1)

    if i % 10 == 1:

        print('now episode: %d, reward: %f'%(i, eps_reward))



plt.plot(np.arange(len(rewards_arr)),rewards_arr)

plt.xlabel('episode')

plt.ylabel('rewards')

plt.show()