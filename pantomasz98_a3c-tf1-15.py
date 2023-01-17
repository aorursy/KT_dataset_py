!pip install 'tensorflow==1.15'

!pip install 'gym[atari]'
# use with tf1 virtual env

import tensorflow as tf

# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()





def build_feature_extractor(input_):

    # create weights once



    input_ = tf.to_float(input_)/255.0



    # conv layers

    conv1 = tf.contrib.layers.conv2d(input_,

                                     16,  # output feature maps

                                     4,  # kernel size

                                     2,  # stride

                                     activation_fn=tf.nn.relu,

                                     scope="conv1"

                                     )



    conv2 = tf.contrib.layers.conv2d(input_,

                                     32,  # output feature maps

                                     4,  # kernel size

                                     2,  # stride

                                     activation_fn=tf.nn.relu,

                                     scope="conv2"

                                     )



    # image -> feature vector

    flat = tf.contrib.layers.flatten(conv2)



    # dense

    fc1 = tf.contrib.layers.fully_connected(

        inputs=flat,

        num_outputs=256,

        scope="fc1"

    )



    return fc1





class PolicyNetwork:

    def __init__(self, num_outputs, reg=0.01):

        self.num_outputs = num_outputs



        # graph inputs

        # use 4 consecutive frames

        self.states = tf.placeholder(

            shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")

        # Adv = G - V(s)

        self.advantage = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        # selected actions

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")



        # reuse = False so we must create Policy before the Value network

        # ValueNet will have reuse = True

        with tf.variable_scope("shared", reuse=False):

            fc1 = build_feature_extractor(self.states)



        # separate scope for output and loss

        with tf.variable_scope("policy_network"):

            self.logits = tf.contrib.layers.fully_connected(

                fc1, num_outputs, activation_fn=None)

            self.probs = tf.nn.softmax(self.logits)



            # sample action

            cdist = tf.distributions.Categorical(logits=self.logits)

            self.sample_action = cdist.sample()



            # add regularization

            self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), axis=1)



            # get preds for the chosen action

            batch_size = tf.shape(self.states)[0]

            gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions

            self.selected_action_probs = tf.gather(

                tf.reshape(self.probs, [-1]), gather_indices)



            self.loss = tf.log(self.selected_action_probs) * self.advantage + reg * self.entropy

            self.loss = -tf.reduce_sum(self.loss, name="loss")



            # training

            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)



            # needed later

            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]





class ValueNetwork:

    def __init__(self):

        # input placeholders

        self.states = tf.placeholder(

            shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")

        # TD target

        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")



        with tf.variable_scope("shared", reuse=True):

            fc1 = build_feature_extractor(self.states)



        with tf.variable_scope("value_network"):

            self.vhat = tf.contrib.layers.fully_connected(

                inputs=fc1, num_outputs=1, activation_fn=None)

            self.vhat = tf.squeeze(self.vhat, squeeze_dims=[1], name="vhat")



            self.loss = tf.squared_difference(self.vhat, self.targets)

            self.loss = tf.reduce_sum(self.loss, name="loss")



            # training

            self.optimizer = tf.train.RMSPropOptimizer(

                0.00025, 0.99, 0.0, 1e-6)



            # needed later for grad desc

            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]



# create nets in the correct order

def create_networks(num_outputs):

    policy_network = PolicyNetwork(num_outputs = num_outputs)

    value_network = ValueNetwork()

    return policy_network, value_network
# from nets import create_networks

import gym

import sys

import os

import numpy as np

import time



# use with tf1 virtual env

import tensorflow as tf

# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()





# for storage

class Step:

    def __init__(self, state, action, reward, next_state, done):

        self.state = state

        self.action = action

        self.reward = reward

        self.next_state = next_state

        self.done = done





# transform raw images

# grayscale

# resize

# crop

class ImageTransformer:

    def __init__(self):

        with tf.variable_scope("image_transformer"):

            self.input_state = tf.placeholder(

                shape=[210, 160, 3], dtype=tf.uint8)

            self.output = tf.image.rgb_to_grayscale(self.input_state)

            self.output = tf.image.crop_to_bounding_box(

                self.output, 34, 0, 160, 160)

            self.output = tf.image.resize_images(

                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            self.output = tf.squeeze(self.output)



    def transform(self, state, sess=None):

        sess = sess or tf.get_default_session()

        return sess.run(self.output, {self.input_state: state})



# create initial state by repeating first frame 4 times





def repeat_frame(frame):

    return np.stack([frame]*4, axis=2)





# create next state by shifting each frame by 1

# throw out the oldest

# concat newest

def shift_frames(state, next_frame):

    return np.append(state[:, :, 1:], np.expand_dims(next_frame, 2), axis=2)



# make tf op to copy weights





def get_copy_params_op(src_vars, dst_vars):

    src_vars = list(sorted(src_vars, key=lambda v: v.name))

    dst_vars = list(sorted(dst_vars, key=lambda v: v.name))



    ops = []

    for s, d in zip(src_vars, dst_vars):

        op = d.assign(s)

        ops.append(op)



    return ops





def make_train_op(local_net, global_net):

    """

    Use grads from the local net to update the global net.

    """

    # We want a list of gradients and corresponding variables

    # e.g. [[g1, g2, g3], [v1, v2, v3]]

    # Since that's what the optimizer expects.

    # But we would like the gradients to come from the local network

    # And the variables to come from the global network

    # So we want to make a list like this:

    # [[local_g1, local_g2, local_g3], [global_v1, global_v2, global_v3]]



    # get local grads

    local_grads, _ = zip(*local_net.grads_and_vars)



    # clip grads

    local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)



    # get global vars

    _, global_vars = zip(*global_net.grads_and_vars)

    local_grads_global_vars = list(zip(local_grads, global_vars))



    # run gradient descent step

    # var = var - lr * grad

    return global_net.optimizer.apply_gradients(local_grads_global_vars, global_step=tf.train.get_global_step())



# Worker object to be run in a thread

# name (String) should be unique for each thread

# env (OpenAI Gym Environment) should be unique for each thread

# policy_net (PolicyNetwork) should be a global passed to every worker

# value_net (ValueNetwork) should be a global passed to every worker

# returns_list (List) should be a global passed to every worker





class Worker:

    def __init__(self, name, env, policy_net, value_net, global_counter, returns_list, max_time, discount_factor=0.99, max_global_steps=None, start_time = time.time()):

        self.name = name

        self.env = env

        self.global_policy_net = policy_net

        self.global_value_net = value_net

        self.global_counter = global_counter

        self.discount_factor = discount_factor

        self.max_global_steps = max_global_steps

        self.global_step = tf.train.get_global_step()

        self.image_transformer = ImageTransformer()

        self.start_time = start_time

        self.max_time = max_time



        # create local policy and val nets

        with tf.variable_scope(name):

            self.policy_net, self.value_net = create_networks(

                policy_net.num_outputs)



        # ops to train global nets

        self.copy_params_op = get_copy_params_op(

            tf.get_collection(

                tf.GraphKeys.TRAINABLE_VARIABLES, scope="global"),

            tf.get_collection(

                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"/")

        )



        self.vnet_train_op = make_train_op(self.value_net, self.global_value_net)

        self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net)



        self.state = None

        self.total_reward = 0.

        self.returns_list = returns_list



    def run(self, sess, coord, t_max, render = False):

        with sess.as_default(), sess.graph.as_default():

            # initial step

            self.state = repeat_frame(

                self.image_transformer.transform(self.env.reset()))



            try:

                while not coord.should_stop():

                    # copy weights

                    sess.run(self.copy_params_op)



                    # collect some experience

                    steps, global_step = self.run_n_steps(t_max, sess, render)



                    # stop when global steps reached or time limit

                    if self.max_global_steps is not None and global_step >= self.max_global_steps:

                        coord.request_stop()

                        print("Max training steps reached.")

                        return

                    if time.time() - self.start_time >= self.max_time:

                        coord.request_stop()

                        print("Max training time reached.")

                        return

                    

                    # update global nets

                    self.update(steps, sess)

            except tf.errors.CancelledError:

                return



    def sample_action(self, state, sess):

        feed_dict = {self.policy_net.states: [state]}

        actions = sess.run(self.policy_net.sample_action, feed_dict)



        return actions[0]



    def get_value_prediction(self, state, sess):

        feed_dict = {self.value_net.states: [state]}

        vhat = sess.run(self.value_net.vhat, feed_dict)

        return vhat[0]



    def run_render(self, sess):

        # steps = []

        done = False

        self.env = gym.wrappers.Monitor(self.env, f'/kaggle/working/vids/{self.env.unwrapped.spec.id}/worker_{self.name}', force = True)

        self.state = repeat_frame(

                self.image_transformer.transform(self.env.reset(), sess))



        while not done:

#             self.env.render()



            # take step

            action = self.sample_action(self.state, sess)

            next_frame, reward, done, _ = self.env.step(action)



            # shift state

            next_state = shift_frames(

                self.state, self.image_transformer.transform(next_frame, sess))



            if done:

                self.state = repeat_frame(

                    self.image_transformer.transform(self.env.reset(), sess))

                break

            else:

                self.state = next_state



    def run_n_steps(self, n, sess, render = False):

        steps = []

        for _ in range(n):

            if render:

                self.env.render()

            # take step

            action = self.sample_action(self.state, sess)

            next_frame, reward, done, _ = self.env.step(action)



            # shift state

            next_state = shift_frames(

                self.state, self.image_transformer.transform(next_frame))



            # save total return

            if done:

                print(f"Total reward: {self.total_reward}\t Worker: {self.name}\t Execution time (from start): {round((time.time()-self.start_time) / 60)} minutes.")

                self.returns_list.append(self.total_reward)

                self.total_reward = 0

                if len(self.returns_list) > 0 and len(self.returns_list) % 10 == 0:

                    print(f"=== Episodes played: {len(self.returns_list)}\t Total avg reward (last 100): {np.mean(self.returns_list[-100:])} ===")

            else:

                self.total_reward += reward



            step = Step(self.state, action, reward, next_state, done)

            steps.append(step)



            # increment local and global counters

            global_step = next(self.global_counter)



            if done:

                self.state = repeat_frame(

                    self.image_transformer.transform(self.env.reset()))

                break

            else:

                self.state = next_state

        return steps, global_step



    def update(self, steps, sess):

        """

        Update global networks using local networks' grads

        """

        # In order to accumulate the total return

        # We will use V_hat(s') to predict the future returns

        # But we will use the actual rewards if we have them

        # Ex. if we have s1, s2, s3 with rewards r1, r2, r3

        # Then G(s3) = r3 + V(s4)

        #      G(s2) = r2 + r3 + V(s4)

        #      G(s1) = r1 + r2 + r3 + V(s4)

        return_ = 0.0



        if not steps[-1].done:

            return_ = self.get_value_prediction(steps[-1].next_state, sess)



        # accumulate minibatch samples

        states = []

        advantages = []

        value_targets = []

        actions = []



        # loop in reverse

        for step in reversed(steps):

            return_ = step.reward + self.discount_factor * return_

            advantage = return_ - self.get_value_prediction(step.state, sess)



            # accumulate updates

            states.append(step.state)

            actions.append(step.action)

            advantages.append(advantage)

            value_targets.append(return_)



        feed_dict = {self.policy_net.states: np.array(states),

                     self.policy_net.advantage: advantages,

                     self.policy_net.actions: actions,

                     self.value_net.states: np.array(states),

                     self.value_net.targets: value_targets

                     }



        # train global estimators using local grads

        global_step, pnet_loss, vnet_loss, _, _ = sess.run([

            self.global_step,

            self.policy_net.loss,

            self.value_net.loss,

            self.pnet_train_op,

            self.vnet_train_op,

        ], feed_dict)



        return pnet_loss, vnet_loss
# from worker import Worker

# from nets import create_networks

import gym

import sys

import os

import numpy as np

# import tensorflow as tf

import matplotlib.pyplot as plt

import itertools

import shutil

import threading

import multiprocessing

import time



# use with tf1 virtual env

import tensorflow as tf

# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior())





# ENV_NAME = "Breakout-v0"

# ENV_NAME = "BeamRider-v0"

# ENV_NAME = "Pong-v0"

# ENV_NAME = "BattleZone-v0"

ENV_NAME = "BeamRiderDeterministic-v4"

MAX_GLOBAL_STEPS = 1e6

MAX_TIME = 8.5*60*60 # 8.5 hours

# MAX_TIME = 15*60 # for tests

STEPS_PER_UPDATE = 5



print("==============")

print(tf.__version__)

print("==============")





def Env():

    return gym.envs.make(ENV_NAME)





# Depending on the game we may have a limited action space

if ENV_NAME == "Pong-v0" or ENV_NAME == "Breakout-v0":

    NUM_ACTIONS = 4  # env.action_space.n returns a bigger number

else:

    env = Env()

    NUM_ACTIONS = env.action_space.n

    env.close()





def smooth(x):

    n = len(x)

    y = np.zeros(n)

    for i in range(n):

        start = max(0, i - 99)

        y[i] = float(x[start:(i+1)].sum())/(i-start+1)

    return y



# what does the model see

def show_image_transformer(sess):

    env = Env()

    im = env.reset()



    # collect images

    im_arr = []

    im_arr.append(im)



    # transform

    image_transformer = ImageTransformer()

    im = image_transformer.transform(im, sess)



    im_arr.append(im)



    # display

    for im in im_arr:

        plt.figure()

        plt.imshow(im)

        



NUM_WORKERS = multiprocessing.cpu_count()

START_TIME = time.time()



with tf.device("/cpu:0"):

    # keep track of number of updates

    global_step = tf.Variable(0, name="global_step", trainable=False)



    # global policy

    with tf.variable_scope("global") as vs:

        policy_net, value_net = create_networks(NUM_ACTIONS)



    # global iterator

    global_counter = itertools.count()



    # save returns

    # list is passed as a reference so the same list is updated by all workers

    returns_list = []



    # create workers

    workers = []

    for worker_id in range(NUM_WORKERS):

        worker = Worker(name=f"worker_{worker_id}", env=Env(), policy_net=policy_net, value_net=value_net,

                        global_counter=global_counter, returns_list=returns_list, max_time = MAX_TIME, discount_factor=0.99, max_global_steps=MAX_GLOBAL_STEPS, start_time = START_TIME)

        workers.append(worker)



with tf.Session() as sess:

#     show_image_transformer(sess)

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()



    # start workers

    worker_threads = []

    for worker in workers:

        worker_fn = lambda:worker.run(sess, coord, STEPS_PER_UPDATE)

        t = threading.Thread(target=worker_fn)

        t.start()

        worker_threads.append(t)



    # wait for the threads to end

    coord.join(worker_threads, stop_grace_period_secs = 300)



    # plot

    x = np.array(returns_list)

    x_smooth = smooth(x)

    plt.plot(x, label = "orig")

    plt.plot(x_smooth, label = "smoothed")

    plt.title(f"Reward over time for {ENV_NAME}")

    plt.xlabel("episode")

    plt.ylabel("reward")

    plt.legend()

    plt.show()



    # render a game

    for worker in workers:

        worker_fn = lambda:worker.run_render(sess)

        t = threading.Thread(target=worker_fn)

        t.start()

        worker_threads.append(t)



    # wait for the threads to end

    coord.join(worker_threads, stop_grace_period_secs = 300)