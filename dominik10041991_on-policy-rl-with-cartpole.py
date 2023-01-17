import gym

import numpy as np

import torch

import torch.nn as nn

import matplotlib.pyplot as plt

import time

from torch.optim import Adam

from torch.distributions.categorical import Categorical

class PGAgent(object):

    def __init__(self, input_dim, output_dim, hidden_dims=[32, 32], size=1, lr=1e-2):

        self.input_dim = input_dim

        self.output_dim = output_dim

        self.size = size

        

        self.net = self.__build_network(input_dim, output_dim, hidden_dims, size)

        self.optimizer = Adam(self.net.parameters(), lr=lr)

        #self.__build_train_fn()

    

    def __build_networkConv(self, input_dim, output_dim, hidden_dims=[32, 32], size=1):

        """Create a base network"""

        #activation = nn.ReLU()

        #activation = nn.Sigmoid()

        activation = nn.Tanh()

        

        device = torch.device('cpu')

        if torch.cuda.is_available():

            device = torch.device('cuda')

            

        layers = []

        layers.append(nn.Conv1d(input_dim,hidden_dims[0],1))

        layers.append(activation)

        #print(size)

        for i in range(0, size):

            layers.append(nn.Conv1d(hidden_dims[0], hidden_dims[1], 1))

            layers.append(activation)

        

        layers.append(nn.Conv1d(hidden_dims[1], output_dim, 1))

        layers.append(nn.Softmax())

        net = nn.Sequential(*layers)

        return net

    

    def __build_network(self, input_dim, output_dim, hidden_dims=[32, 32], size=1):

        """Create a base network"""

        #activation = nn.ReLU()

        #activation = nn.Sigmoid()

        activation = nn.Tanh()

        

        device = torch.device('cpu')

        if torch.cuda.is_available():

            device = torch.device('cuda')

            

        layers = []

        layers.append(nn.Linear(input_dim,hidden_dims[0]))

        layers.append(activation)

        #print(size)

        for i in range(0, size):

            layers.append(nn.Linear(hidden_dims[0], hidden_dims[1]))

            layers.append(activation)

        

        layers.append(nn.Linear(hidden_dims[1], output_dim))

        layers.append(nn.Softmax())

        net = nn.Sequential(*layers)

        return net

    def get_name(self):

        return "PG"

    

    def update(self):

        pass

    

    def get_policy(self, state):

        policy = self.net(state)

        return Categorical(logits=policy)

    

    def get_action(self, state):

        #return self.get_policy(state).sample().item()

        policy = self.net(state)

        out_probs = policy.detach().numpy()

        #out_probs.

        #print(out_probs)

        return int(np.random.choice(out_probs.shape[0],p=out_probs)) #self.get_policy(state).sample().item()

    

    # make loss function whose gradient, for the right data, is policy gradient

    def compute_loss(self, state, act, reward):

        #logp = self.get_policy(state).log_prob(act)

        #logp *= -1

        #print(logp)

        p = self.net(state)

        one_hot = nn.functional.one_hot(act,2)

        p = p * one_hot

        p = torch.sum(p,1)

        p = -torch.log(p)

        

        

        

        return (p*reward).mean(), None, None
class network(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2, size):

        super(network, self).__init__()

        self.input = nn.Linear(input_dim,hidden_dim[0])

        self.hidden = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.out1 = nn.Linear(hidden_dim[1],output_dim1)

        self.out2 = nn.Linear(hidden_dim[1],output_dim2)

        self.size = size



    def forward(self, x):

        x = nn.functional.relu(self.input(x))

        for i in range(0, self.size):

            x = nn.functional.relu(self.hidden(x))

        out1 = nn.functional.softmax(self.out1(x))

        out2 = nn.functional.relu(self.out2(x))

        return out1, out2

    

class PGAgentA2C(object):

    def __init__(self, input_dim, output_dim, hidden_dims=[32, 32], size=1, lr=1e-2):

        self.input_dim = input_dim

        self.hidden_dims = hidden_dims

        self.output_dim1 = output_dim

        self.output_dim2 = 1

        self.size = size

        """print(self.input_dim)

        print(self.hidden_dims)

        print(self.output_dim1)

        print(self.output_dim2)

        print(self.size)"""

        self.net = network(self.input_dim, self.hidden_dims, self.output_dim1, self.output_dim2, self.size)

        #print(self.net)

        self.optimizer = Adam(self.net.parameters(), lr=lr)

        #self.__build_train_fn()

    

    def update(self):

        pass

    

    def get_name(self):

        return "PGA2C"

    

    def get_output(self,state):

        policy, v_function = self.net(state)

        #print(policy)

        #print(v_function)

        return policy, v_function

    

    def get_vFunction(self,state):

        policy, v_function = self.get_output(state)

        #print(torch.max(v_function, 1))

        return v_function#Categorical(logits=v_function)

    

    def get_policy(self, state):

        policy, v_function = self.get_output(state)

        #print(torch.max(policy).item())

        return Categorical(logits=policy)

    

    def get_action(self, state):

        policy, _ = self.net(state)

        out_probs = policy.detach().numpy()

        #out_probs.

        #print(out_probs)

        return int(np.random.choice(out_probs.shape[0],p=out_probs)) #self.get_policy(state).sample().item()

    

    def get_value_loss(self, state, reward):

        loss = nn.MSELoss()

        v = self.get_vFunction(state)

        return loss(v, reward)

    

    # make loss function whose gradient, for the right data, is policy gradient

    def compute_loss(self, state, act, reward):

        p, _ = self.net(state)

        one_hot = nn.functional.one_hot(act,2)

        p = p * one_hot

        p = torch.sum(p,1)

        p = torch.log(p)

        v = self.get_vFunction(state)#.logits()

        a = (reward - v)

        #d = v.log_prob(long(L))

        

        value_loss = self.get_value_loss(state, reward)

        print("value loss: {}".format(value_loss))

        policy_loss = (p * a).mean()

        print("Policy loss: {}".format(policy_loss))

        total_loss = policy_loss + value_loss

        #print(ergebnis)

        

        

        return total_loss, policy_loss, value_loss
class network(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2, size):

        super(network, self).__init__()

        self.input = nn.Linear(input_dim,hidden_dim[0])

        self.hidden = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.out1 = nn.Linear(hidden_dim[1],output_dim1)

        self.out2 = nn.Linear(hidden_dim[1],output_dim2)

        self.size = size



    def forward(self, x):

        x = nn.functional.relu(self.input(x))

        for i in range(0, self.size):

            x = nn.functional.relu(self.hidden(x))

        out1 = nn.functional.softmax(self.out1(x))

        out2 = nn.functional.relu(self.out2(x))

        return out1, out2

    

class PPOAgent(object):

    def __init__(self, input_dim, output_dim, hidden_dims=[32, 32], size=1, lr=1e-2):

        self.input_dim = input_dim

        self.hidden_dims = hidden_dims

        self.output_dim1 = output_dim

        self.output_dim2 = 1

        self.size = size

        """print(self.input_dim)

        print(self.hidden_dims)

        print(self.output_dim1)

        print(self.output_dim2)

        print(self.size)"""

        

        self.net = network(self.input_dim, self.hidden_dims, self.output_dim1, self.output_dim2, self.size)

        

        self.net_old = network(self.input_dim, self.hidden_dims, self.output_dim1, self.output_dim2, self.size)

        self.net_old.load_state_dict(self.net.state_dict())

        #print(self.net)

        self.optimizer = Adam(self.net.parameters(), lr=lr)

        #self.__build_train_fn()

    

    def get_name(self):

        return "PPO"

    

    def get_output(self,state):

        policy, v_function = self.net(state)

        return policy, v_function

    

    def get_vFunction(self,state):

        policy, v_function = self.get_output(state)

        #print(torch.max(v_function, 1))

        return v_function#Categorical(logits=v_function)

    

    def get_policy(self, state):

        policy, v_function = self.get_output(state)

        return Categorical(logits=policy)

    

    def get_old_policy(self, state):

        policy, v_function = self.net_old(state)

        return Categorical(logits=policy)

    

    def get_action(self, state):

        policy, _ = self.net(state)

        

        out_probs = policy.detach().numpy()

        #out_probs.

        #print(out_probs)

        return int(np.random.choice(out_probs.shape[0],p=out_probs))

        #print(self.get_policy(state).sample().item())

        #return self.get_policy(state).sample().item()

    

    def get_value_loss(self, state, reward):

        loss = nn.MSELoss()

        p, v = self.net(state)

        return loss(v, reward)

    

    def compute_ratio(self, state,act):

        #ep = 0.00000000000001

        #p = self.get_policy(state)

        #p = p.log_prob(act)

        

        #po = self.get_old_policy(state)

        #po = po.log_prob(act)

        

        policy, _ = self.net(state)

        one_hot = nn.functional.one_hot(act,2)

        policy = policy * one_hot

        policy = torch.sum(policy,1)

        #policy = torch.log(policy)

        

        policy_old, _ = self.net_old(state)

        policy_old = policy_old * one_hot

        one_hot = nn.functional.one_hot(act,2)

        policy_old = policy_old * one_hot

        policy_old = torch.sum(policy_old,1)

        #policy_old = torch.log(policy_old)

        

        #print("Policy : {}".format(policy))

        #print("Policy old: {}".format(policy_old))

        #print(policy)

        

        #r = torch.exp(p - po)

        #r = (p+1)/(po+1)

        r = torch.exp(policy_old - policy)

        #r = (policy + 1) / (policy_old + 1)

        #r = r.log()

        #r = policy * r

        #r = policy.sum() / policy_old.sum()

        #print(r)

        return r

    def compute_advantage(self, state, reward):

        p, v = self.net(state)

        a = (reward - v)

        return a

    

    def compute_entropy(self, state):

        p, _ = self.net(state)

        p = p * torch.log(p)

        return p.mean()

    

    def update(self):

        self.net_old.load_state_dict(self.net.state_dict())

        

    # make loss function whose gradient, for the right data, is policy gradient

    def compute_loss(self, state, act, reward):

        r = self.compute_ratio(state, act)

        a = self.compute_advantage(state, reward)

        #print(a)

        #logp = self.get_policy(state)

        #logp *= -1

        e = 0.2

        beta = 0.01

        #r = r.data.numpy()

        #a = a.data.numpy()

        right = torch.clamp(r, 1-e,1+e) * a #np.(r, 1-e, 1+e) * a

        left = r * a

        #print(left)

        #print(right)

        minimum = torch.min(left, right)

        #v = self.get_vFunction(state)#.logits()

        #d = v.log_prob(long(L))

        

        entropy = self.compute_entropy(state)

        value_loss = self.get_value_loss(state, reward)

        policy_loss = (minimum.mean())

        total_loss = policy_loss + (1 * value_loss) #- (beta * entropy)

        #print(ergebnis)

        print("value loss: {}".format(value_loss))

        print("policy loss: {}".format(policy_loss))

        #print("left: {}".format(left))

        #print("right: {}".format(right))

        #print("minimum: {}".format(minimum))

        

        return total_loss, policy_loss, value_loss
class Environment(object):

    def __init__(self, env, agent, render=False, epochs=2000, title="noName", batchsize=500, lr=0.0001):

        self.env = env

        self.agent = agent

        self.render = render

        self.epochs = epochs

        self.title = title

        self.batchsize = batchsize

        self.lr = lr

        

    def _run_one_epoch(self):

        #make some empty lists

        batch_state = []

        batch_action = []

        batch_return = []

        batch_reward = []

        batch_len = []

        batch_weight = []

        tmp_state = []

        done = False

        state = self.env.reset()

        

        while(True):

            if(self.render):

                self.env.render()

            #save state

            batch_state.append(state.copy())

            tmp_state.append(state.copy())

            

            #act in the environment

            action = self.agent.get_action(torch.as_tensor(state, dtype=torch.float32))

            state, reward, done, info = self.env.step(action)

            

            #save action, reward

            batch_action.append(action)

            batch_reward.append(reward)

            

            if done:

                ret, length = sum(batch_reward), len(batch_reward)

                batch_return.append(ret)

                batch_len.append(length)

                

                

                #print([ret])

                #print(length)

                #print(batch_weight)

                if(self.agent.get_name() == "PG"):

                    tmp = [ret] * length

                    count = 0

                    gamma= 0.99

                    for i in reversed(range(len(tmp))):

                        tmp[i] *= (gamma**count)

                        count += 1

                    batch_weight += tmp

                elif((self.agent.get_name() == "PGA2C") or (self.agent.get_name() == "PPO")):

                    #tmp = self.compute_advantage(torch.as_tensor(batch_reward, dtype=torch.float32), torch.as_tensor(tmp_state, dtype=torch.float32))

                    #batch_weight.extend(tmp)

                    count = 0

                    gamma = 0.99

                    summe = 0

                    for i in reversed(range(len(batch_reward))):

                        #summe = gamma * summe + batch_reward[i]

                        #batch_reward[i] = summe

                        batch_reward[i] = batch_reward[i] * (gamma**count)

                        count += 1

                    #print("rewards: {}".format(batch_reward))

                        

                    batch_weight.extend(batch_reward)

                #print(batch_weight) 

                #while(1):

                    #pass

                #batch_weight.extend(batch_reward)

                state, done = self.env.reset(), False, 

                batch_reward = []

                tmp_state = []

                if(len(batch_state)>self.batchsize):

                    break

        

        self.agent.optimizer.zero_grad()

        """print(len(torch.as_tensor(batch_state, dtype=torch.float32)))

        print(len(torch.as_tensor(batch_action, dtype=torch.int32)))

        print(len(torch.as_tensor(batch_reward, dtype=torch.float32)))

        print(torch.as_tensor(batch_state, dtype=torch.float32))

        print(torch.as_tensor(batch_action, dtype=torch.int32))

        print(torch.as_tensor(batch_reward, dtype=torch.float32))"""

        

       

        #batch_loss, policy_loss, value_loss = self.agent.compute_loss(state=torch.as_tensor(batch_state, dtype=torch.float32),

                                  #act=torch.as_tensor(batch_action, dtype=torch.int32),#act=torch.LongTensor(batch_action[j:m]),#torch.as_tensor(batch_action, dytpe=torch.dtype=torch.int32),

                                  #reward=torch.as_tensor(batch_weight, dtype=torch.float32))

        batch_loss, policy_loss, value_loss = self.agent.compute_loss(state=torch.as_tensor(batch_state, dtype=torch.float32),

                                  act=torch.LongTensor(batch_action),#torch.as_tensor(batch_action, dytpe=torch.dtype=torch.int32),

                                  reward=torch.as_tensor(batch_weight, dtype=torch.float32))

        

        batch_loss.backward()

        self.agent.optimizer.step()

        

        

        return batch_loss, batch_return, batch_len, policy_loss, value_loss

    

    def _plotLoss(self, loss, ret, epoch, policy_loss=None, value_loss=None):

        fig, ax = plt.subplots()

        plt.title(self.title)

        #print(loss)

        #print(ret)

        plt.plot(epoch, loss, label="Total Loss")

        #plt.plot(epoch, ret, label="return")

        if(policy_loss is not None):

             plt.plot(epoch, policy_loss, label="Policy Loss")

        if(value_loss is not None):

             plt.plot(epoch, value_loss, label="Value Loss")

        #plt.ylabel('Loss')"""

        plt.xlabel('Epochs')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        textstr = "α: {}\nbatchsize: {}".format(self.lr, self.batchsize)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords

        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,

        verticalalignment='top', bbox=props)

        plt.show()

    

    def _plot(self, loss, ret, epoch, policy_loss=None, value_loss=None):

        fig, ax = plt.subplots()

        plt.title(self.title)

        #print(loss)

        #print(ret)

        #plt.plot(epoch, loss, label="Total Loss")

        plt.plot(epoch, ret, label="return")

        #plt.ylabel('Loss')"""

        plt.xlabel('Epochs')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        textstr = "α: {}\nbatchsize: {}".format(self.lr, self.batchsize)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords

        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,

        verticalalignment='top', bbox=props)

        plt.show()

        

        

    

    def run(self):

        loss = []

        ret = []

        epoch=[]

        policy_loss=[]

        value_loss=[]

        start_proc = time.process_time()

        for i in range(self.epochs):                

            batch_loss, batch_return, batch_len, policy_los, value_los = self._run_one_epoch()

            

            if(policy_los is None):

                policy_loss = None

            else:

                policy_loss.append(policy_los)

              

            if(value_los is None):

                value_loss = None

            else:

                value_loss.append(value_los)

                

            loss.append(batch_loss)

            ret.append(np.mean(batch_return))

            epoch.append(i)

            print('epoch: %3d \t loss: %f \t return: %.3f \t ep_len: %.3f'%(i, batch_loss, np.mean(batch_return), np.mean(batch_len)))

            if((i % 10 == 0) and (self.agent.get_name() == "PPO")):

                self.agent.update()

            self.agent.update()

        ende_proc = time.process_time()

        print('Systemzeit: {:5.3f}s'.format(ende_proc-start_proc))

        self._plot(loss, ret, epoch, policy_loss, value_loss)

        self._plotLoss(loss, ret, epoch, policy_loss, value_loss)

    

    def compute_advantage(self, reward, state):

        #print(reward)

        #print(state)

        

        v = self.agent.get_vFunction(state)

        a = v - reward

        

        a = (a).mean()

        #print(a)

        

        

        return [a] * len(reward)
def main():

    try:

        env = gym.make("CartPole-v1")

        input_dim = env.observation_space.shape[0]

        output_dim = env.action_space.n

        print(output_dim)

        

        alpha = 0.001

        batchsize = 2000

        epoch = 500

        render = False

        hidden_size = 2

        plotname = "Policy Gradient"

        agent = PGAgent(input_dim, output_dim, [64, 64], hidden_size, lr=alpha)

        environment = Environment(env, agent, render, epoch, plotname, batchsize=batchsize, lr=alpha)

        #environment.run()



        alpha = 0.001

        batchsize = 2000

        epoch = 500

        render = False

        hidden_size = 2

        plotname = "Policy Gradient with A2C"

        agent = PGAgentA2C(input_dim, output_dim, [64, 64], hidden_size, lr=alpha)

        environment = Environment(env, agent, render, epoch, plotname, batchsize=batchsize, lr=alpha)      

        #environment.run()

        

        #print(output_dim)

        alpha = 0.02

        batchsize = 500

        epoch = 100

        render = False

        hidden_size = 2

        plotname = "PPO"

        agent = PPOAgent(input_dim, output_dim, [64, 64], hidden_size, lr=alpha)

        environment = Environment(env, agent, render, epoch, plotname, batchsize=batchsize, lr=alpha)      

        environment.run()

        

        

        #for episode in range(2000):

            #reward = run_episode(env, agent)

            #print(episode, reward)

        #print(input_dim)

        #print(output_dim)

        """state = env.reset()

        a = agent.net(torch.as_tensor(state, dtype=torch.float32))

        act = agent.get_action(torch.as_tensor(state, dtype=torch.float32))

        states = []

        acts = []

        states.append(state)

        acts.append(act)

        statep = torch.as_tensor(states, dtype=torch.float32)

        actsp = torch.as_tensor(acts, dtype=torch.int32)

        logp = agent.get_policy(statep).log_prob(actsp)

        print(logp)"""

        #b = Categorical(logits=a).sample().item()

        #print(agent.get_action(state))

        

        

    finally:

        env.close()





if __name__ == '__main__':

    main()