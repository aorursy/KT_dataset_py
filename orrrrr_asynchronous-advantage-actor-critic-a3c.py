!pip install tensorflow==1.13.1
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import threading

import multiprocessing

import gym

import os

import shutil
game_name = 'CartPole-v0'

log_dir = './tmp/log'

output_graph = False

n_workers = multiprocessing.cpu_count()



_gamma = 0.9

_epsilon = 1e-6

_entropy_param = 0.001

_max_global_episode = 3000

_global_episode = 0

_update_global_iter = 10

_render_flag = False

_global_reward_records = []

_actor_lr = 0.001

_critic_lr = 0.001

_global_net_scope = 'Global_Net'



env = gym.make(game_name)

num_state = env.observation_space.shape[0]

num_action = env.action_space.n





class A3CNet(object):

    def __init__(self, scope, global_net=None):

        if scope == _global_net_scope:   # global network

            with tf.variable_scope(scope):

                self.state = tf.placeholder(tf.float32, [None, num_state], 'state')

                self.actor_variables, self.critic_variables = self._build_AC_net(scope)[-2:]

        else:   # local net, calculate losses

            with tf.variable_scope(scope):

                self.state = tf.placeholder(tf.float32, [None, num_state], 'state')

                self.action = tf.placeholder(tf.int32, [None, ], 'action')

                self.target_v = tf.placeholder(tf.float32, [None, 1], 'target_v')



                self.action_prob, self.v, self.actor_variables, self.critic_variables = self._build_AC_net(scope)



                td_error = tf.subtract(self.target_v, self.v, name='td_error')

                with tf.name_scope('critic_loss'):

                    self.critic_loss = tf.reduce_mean(tf.square(td_error))



                with tf.name_scope('actor_loss'):

                    log_action_prob = tf.reduce_sum(tf.one_hot(self.action, num_action, dtype=tf.float32) * tf.log(self.action_prob + _epsilon), axis=1, keep_dims=True)

                    entropy = -tf.reduce_sum(self.action_prob * tf.log(self.action_prob + _epsilon),

                                             axis=1, keep_dims=True)  # for exploration

                    self.actor_loss = tf.reduce_mean(-(log_action_prob * tf.stop_gradient(td_error) + _entropy_param * entropy))



                with tf.name_scope('local_gradient'):

                    self.actor_grads = tf.gradients(self.actor_loss, self.actor_variables)

                    self.critic_grads = tf.gradients(self.critic_loss, self.critic_variables)



            with tf.name_scope('sync'):

                with tf.name_scope('pull'):

                    self.pull_actor_variables_op = [l_p.assign(g_p) for l_p, g_p in zip(self.actor_variables, global_net.actor_variables)]

                    self.pull_critic_variables_op = [l_p.assign(g_p) for l_p, g_p in zip(self.critic_variables, global_net.critic_variables)]

                with tf.name_scope('push'):

                    self.update_actor_op = actor_optimizer.apply_gradients(zip(self.actor_grads, global_net.actor_variables))

                    self.update_critic_op = critic_optimizer.apply_gradients(zip(self.critic_grads, global_net.critic_variables))



    def _build_AC_net(self, scope):

        init_weights = tf.random_normal_initializer(0., .1)

        with tf.variable_scope('actor'):

            policy_fc_1 = tf.layers.dense(self.state, 200, tf.nn.relu6, kernel_initializer=init_weights, name='policy_fc1')

            action_prob = tf.layers.dense(policy_fc_1, num_action, tf.nn.softmax, kernel_initializer=init_weights, name='action_prob')

        with tf.variable_scope('critic'):

            v_fc_1 = tf.layers.dense(self.state, 100, tf.nn.relu6, kernel_initializer=init_weights, name='v_fc1')

            v = tf.layers.dense(v_fc_1, 1, kernel_initializer=init_weights, name='state_value')  # state value

        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')

        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return action_prob, v, actor_variables, critic_variables



    def update_global_variables(self, feed_dict):  # run by a local

        sess.run([self.update_actor_op, self.update_critic_op], feed_dict)  # local grads applies to global net



    def pull_global_variables(self):  # run by a local

        sess.run([self.pull_actor_variables_op, self.pull_critic_variables_op])



    def act(self, state):  # run by a local

        prob_distribution = sess.run(self.action_prob, feed_dict={self.state: state[np.newaxis, :]})

        action = np.random.choice(range(prob_distribution.shape[1]),

                                  p=prob_distribution.ravel())  # select action w.r.t the actions prob

        return action
class Worker(object):

    def __init__(self, name, global_net):

        self.env = gym.make(game_name).unwrapped

        self.name = name

        self.A3C = A3CNet(name, global_net)



    def work(self):

        global _global_episode, _global_reward_records

        state_buffer, action_buffer, reward_buffer = [], [], []

        while _global_episode < _max_global_episode and not coordinator.should_stop():

            state = self.env.reset()

            episode_reward = 0

            step = 0

            while True:

                if _render_flag:

                    if self.name == 'worker_0':   # visualization of worker 0 

                        self.env.render()

                action = self.A3C.act(state)

                next_state, reward, terminal, info = self.env.step(action)

                step += 1

                if step == 200:

                    terminal = True

                state_buffer.append(state)

                action_buffer.append(action)

                reward_buffer.append(reward)

                episode_reward += reward



                if terminal or step % _update_global_iter == 0:   # update global and assign to local net

                    if terminal:

                        v_next_state = 0   # terminal

                    else:

                        v_next_state = sess.run(self.A3C.v, {self.A3C.state: next_state[np.newaxis, :]})[0, 0]

                    target_v_buffer = []

                    for reward in reward_buffer[::-1]:    # reverse buffer r

                        v_next_state = reward + _gamma * v_next_state

                        target_v_buffer.append(v_next_state)

                    target_v_buffer.reverse()



                    state_buffer, action_buffer, target_v_buffer = np.vstack(state_buffer), np.array(action_buffer), np.vstack(target_v_buffer)

                    feed_dict = {

                        self.A3C.state: state_buffer,

                        self.A3C.action: action_buffer,

                        self.A3C.target_v: target_v_buffer,

                    }

                    self.A3C.update_global_variables(feed_dict)



                    state_buffer, action_buffer, reward_buffer = [], [], []

                    self.A3C.pull_global_variables()



                state = next_state

                

                if terminal:

                    if len(_global_reward_records) == 0:  # record running episode reward

                        _global_reward_records.append(episode_reward)

                    else:

                        _global_reward_records.append(0.99 * _global_reward_records[-1] + 0.01 * episode_reward)

                    print(

                        self.name,

                        "episode:", _global_episode,

                        "| episode_reward: %i" % _global_reward_records[-1],

                          )

                    _global_episode += 1

                    break
if __name__ == "__main__":

    sess = tf.Session()

    with tf.device("/cpu:0"):

        actor_optimizer = tf.train.RMSPropOptimizer(_actor_lr, name='RMSPropActor')

        critic_optimizer = tf.train.RMSPropOptimizer(_critic_lr, name='RMSPropCritic')

        global_a3c_net = A3CNet(_global_net_scope)  # we only need its params

        workers = []

        # Create worker

        for i in range(n_workers):

            worker_name = 'worker_%i' % i   # worker name

            workers.append(Worker(worker_name, global_a3c_net))

    coordinator = tf.train.Coordinator()

    sess.run(tf.global_variables_initializer())



    if output_graph:

        if os.path.exists(log_dir):

            shutil.rmtree(log_dir)

        tf.summary.FileWriter(log_dir, sess.graph)



    worker_threads = []

    for worker in workers:   # create threads for workers

        job = lambda: worker.work()

        thread = threading.Thread(target=job)

        thread.start()

        worker_threads.append(thread)

    coordinator.join(worker_threads)

    # plot results

    plt.plot(np.arange(len(_global_reward_records)), _global_reward_records)

    plt.xlabel('step')

    plt.ylabel('weighted moving average reward')

    plt.show()