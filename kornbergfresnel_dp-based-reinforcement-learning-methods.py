import numpy as np



from copy import copy





def policy_eval(env, values, policies, upper_bound):

    print('\n===== Policy Evalution =====')

    delta = upper_bound

    iteration = 0



    while delta >= upper_bound:

        delta = 0.



        for s in env.states:

            v = values.get(s)

            env.set_state(s)



            action_index = policies.sample(s)

            action = env.actions[action_index]

            _, _, rewards, next_states = env.step(action)



            next_values = values.get(list(next_states))

            td_values = list(map(lambda x, y: x + env.gamma * y, rewards, next_values))



            exp_value = np.mean(td_values)

            values.update(s, exp_value)



            # update delta

            delta = max(delta, abs(v - exp_value))

            

        iteration += 1

        print('\r> iteration: {} delta: {}'.format(iteration, delta), flush=True, end="")
def policy_improve(env, values, policies):

    print('\n===== Policy Improve =====')

    policy_stable = True

    

    for state in env.states:

        old_act = policies.sample(state)



        # calculate new policy execution

        actions = env.actions

        value = [0] * len(env.actions)

        

        for i, action in enumerate(actions):

            env.set_state(state)

            _, _, rewards, next_states = env.step(action)

            next_values = values.get(list(next_states))

            td_values = list(map(lambda x, y: x + env.gamma * y, rewards, next_values))

            prob = [1 / len(next_states)] * len(next_states)



            value[i] = sum(map(lambda x, y: x * y, prob, td_values))



        # action selection

        new_act = actions[np.argmax(value)]



        # greedy update policy

        new_policy = [0.] * env.action_space

        new_policy[new_act] = 1.

        policies.update(state, new_policy)



        if old_act != new_act:

            policy_stable = False



    return policy_stable
def value_iter(env, values, upper_bound):

    print('===== Value Iteration =====')

    delta = upper_bound + 1.

    states = copy(env.states)

    

    iteration = 0



    while delta >= upper_bound:

        delta = 0



        for s in states:

            v = values.get(s)



            # get new value

            actions = env.actions

            vs = [0] * len(actions)



            for i, action in enumerate(actions):

                env.set_state(s)

                _, _, rewards, next_states = env.step(action)

                td_values = list(map(lambda x, y: x + env.gamma * y, rewards, values.get(next_states)))



                vs[i] = np.mean(td_values)



            values.update(s, max(vs))

            delta = max(delta, abs(v - values.get(s)))

        

        iteration += 1

        print('\r> iteration: {} delta: {}'.format(iteration, delta), end="", flush=True)

        

    return
class Env:

    def __init__(self):

        self._states = set()

        self._state = None

        self._actions = []

        self._gamma = None

        

    @property

    def states(self):

        return self._states

    

    @property

    def state_space(self):

        return self._state_shape

    

    @property

    def actions(self):

        return self._actions

    

    @property

    def action_space(self):

        return len(self._actions)

    

    @property

    def gamma(self):

        return self._gamma

    

    def _world_init(self):

        raise NotImplementedError

        

    def reset(self):

        raise NotImplementedError

    

    def step(self, state, action):

        """Return distribution and next states"""

        raise NotImplementedError

        

    def set_state(self, state):

        self._state = state





class MatrixEnv(Env):

    def __init__(self, height=4, width=4):

        super().__init__()

        

        self._action_space = 4

        self._actions = list(range(4))

        

        self._state_shape = (2,)

        self._state_shape = (height, width)

        self._states = [(i, j) for i in range(height) for j in range(width)]

        

        self._gamma = 0.9

        self._height = height

        self._width = width



        self._world_init()

        

    @property

    def state(self):

        return self._state

    

    @property

    def gamma(self):

        return self._gamma

    

    def set_gamma(self, value):

        self._gamma = value

        

    def reset(self):

        self._state = self._start_point

        

    def _world_init(self):

        # start_point

        self._start_point = (0, 0)

        self._end_point = (self._height - 1, self._width - 1)

        

    def _state_switch(self, act):

        # 0: h - 1, 1: w + 1, 2: h + 1, 3: w - 1

        if act == 0:  # up

            self._state = (max(0, self._state[0] - 1), self._state[1])

        elif act == 1:  # right

            self._state = (self._state[0], min(self._width - 1, self._state[1] + 1))

        elif act == 2:  # down

            self._state = (min(self._height - 1, self._state[0] + 1), self._state[1])

        elif act == 3:  # left

            self._state = (self._state[0], max(0, self._state[1] - 1))



    def step(self, act):

        assert 0 <= act <= 3

        

        done = False

        reward = 0.



        self._state_switch(act)

        

        if self._state == self._end_point:

            reward = 1.

            done = True



        return None, done, [reward], [self._state]
class ValueTable:

    def __init__(self, env):

        self._values = np.zeros(env.state_space)

        

    def update(self, s, value):

        self._values[s] = value

        

    def get(self, state):

        if type(state) == list:

            # loop get

            res = [self._values[s] for s in state]

            return res

        elif type(state) == tuple:

            # return directly

            return self._values[state]
from collections import namedtuple





Pi = namedtuple('Pi', 'act, prob')





class Policies:

    def __init__(self, env: Env):

        self._actions = env.actions

        self._default_policy = [1 / env.action_space] * env.action_space

        self._policies = dict.fromkeys(env.states, Pi(self._actions, self._default_policy))

    

    def sample(self, state):

        if self._policies.get(state, None) is None:

            self._policies[state] = Pi(self._actions, self._default_policy)



        policy = self._policies[state]

        return np.random.choice(policy.act, p=policy.prob)

    

    def retrieve(self, state):

        return self._policies[state].prob

    

    def update(self, state, policy):

        self._policies[state] = self._policies[state]._replace(prob=policy)
import time



env = MatrixEnv(width=8, height=8)  # TODO(ming): try different word size

policies = Policies(env)

values = ValueTable(env)

upper_bound = 1e-4



stable = False



start = time.time()

while not stable:

    policy_eval(env, values, policies, upper_bound)

    stable = policy_improve(env, values, policies)

end = time.time()



print('\n[time consumpution]: {} s'.format(end - start))



done = False

rewards = 0

env.reset()

step = 0



while not done:

    act_index = policies.sample(env.state)

    _, done, reward, next_state = env.step(env.actions[act_index])

    rewards += sum(reward)

    step += 1



print('Evaluation: [reward] {} [step] {}'.format(rewards, step))
env = MatrixEnv(width=8, height=8)  # try different word size

policies = Policies(env)

values = ValueTable(env)

upper_bound = 1e-4



start = time.time()

value_iter(env, values, upper_bound)

_ = policy_improve(env, values, policies)

end = time.time()



print('\n[time consumption] {}s'.format(end - start))

# print("===== Render =====")

env.reset()

done = False

rewards = 0

step = 0

while not done:

    act_index = policies.sample(env.state)

    _, done, reward, next_state = env.step(env.actions[act_index])

    rewards += sum(reward)

    step += 1



print('Evaluation: [reward] {} [step] {}'.format(rewards, step))