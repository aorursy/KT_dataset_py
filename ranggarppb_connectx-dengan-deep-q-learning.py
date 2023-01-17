#  ConnectX hanya bisa dijalankan pada kaggle versi 0.1.6



!pip install 'kaggle-environments>=0.1.6' > /dev/null 2>&1
# Import modul yang diperlukan



# Modul umum

import  numpy as np                # perhitungan matematika

from tqdm.notebook import tqdm     # menampilkan progress bar

import matplotlib.pyplot as plt    # menggambar plot



# Modul mempersiapkan environment ConnectX

import gym

from kaggle_environments import evaluate, make



# Modul neural network

import torch

import torch.nn as nn

import torch.optim as optim
# Mendefinisikan class yang akan dipakai sepanjang kernel ini



class ConnectX(gym.Env) :

    

    # constructor method (inisialisasi)

    def __init__(self, switch_prob = 0.5) :

        

        # membuat environment (method-methodnya dapat dilihat pada kernel Getting Started di atas)

        self.env = make('connectx', debug = False)

        # mengambil setting pada environment ConnectX (berisi timeout, columns, rows, inarow atau syarat berhasil, dan steps)   

        config = self.env.configuration

        # mendefinisikan jumlah aksi yang dapat dilakukan (mengisi kolom ke berapa) atau action space dan jumlah state atau obs space

        self.action_space = gym.spaces.Discrete(config.columns)

        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)

        

        # melatih agent kita sebagai player pertama melawan agent Negamax

        # memasukkan parameter [None, "negamax"] ke dalam atribut sendiri agar mudah diakses dan diganti (misal bertukar posisi player) 

        self.pair = [None, 'negamax']     

        self.trainer = self.env.train(self.pair)

        # mendefinisikan atribut peluang untuk berganti posisi player dalam proses pelatihan agent nantinya

        self.switch_prob = switch_prob

        

    # method untuk berganti posisi player    

    def switch_trainer(self) :   

        # mengganti urutan elemen pada self.pair yang telah diinisiasi

        self.pair = self.pair[::-1]

        self.trainer = self.env.train(self.pair)     

    def reset(self) :

        if np.random.random() < self.switch_prob :

            self.switch_trainer()

        return self.trainer.reset()

    

    # method untuk mengakses observasi, reward, status done atau belum, dan info selama pelatihan agent sesuai dengan action yang dilakukan

    def step(self, action) :

        return self.trainer.step(action)

    

    # method untuk merender state

    def render(self, **kwargs) :

        return self.env.render(**kwargs)



# Mendefinisikan kelas untuk neural network untuk proses pelatihan agent (lihat https://pytorch.org/docs/stable/nn.html#containers)



class DeepModel(nn.Module) :

    

    # constructor method (inisialisai)

    def __init__(self, num_states, hidden_units, num_actions) :

         

        super(DeepModel, self).__init__()

        

        # mengkonstruksi hidden layer (perhatikan ilustrasi struktur neural network pada gambar di atas)

        # akan dicoba activation function berupa ReLU 

        self.hidden_layers = []

        for i in range(len(hidden_units)) :

            # untuk hidden layer pertama

            if i == 0 :

                self.hidden_layers.append(nn.Linear(num_states, hidden_units[i]))

            # untuk hidden layer berikutnya

            else :

                self.hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))

        self.output_layer = nn.Linear(hidden_units[-1], num_actions)

    

    def forward(self, x) :

        for layer in self.hidden_layers :

            x = layer(x).clamp(min=0)

        x = self.output_layer(x)

        return x

    

# Mendefinisikan kelas untuk agent

# Secara umum, agent harus punya 3 kemampuan : bermain beberapa permainan, mengingatnya, dan memperkirakan reward dari tiap state

# yang muncul pada permainan



class DQN :



    # constructor method (inisialisasi)

    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr) :

        

        #inisialisasi atribut untuk bermain

        self.num_actions = num_actions   # banyaknya aksi

        self.gamma = gamma               # porsi future reward terhadap immadiate reward

        

        # inisialisasi atribut untuk mekanisme mengingat permainan

        self.batch_size = batch_size

        self.experience = {'s' : [], 'a' : [], 'r' : [], 's2' : [], 'done' : []}

        self.max_experiences = max_experiences

        self.min_experiences = min_experiences

        

        # inisialisasi atribut untuk perkiraan reward dengan neural network di setiap state

        self.model = DeepModel(num_states, hidden_units, num_actions)

        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)

        self.criterion = nn.MSELoss()

        

    # fungsi untuk mengatur permainan yang diingat lalu melakukan training neural network dari sana

    

    # membuang ingatan state paling awal bila sudah melebihi batas memori

    def add_experience(self, exp) :

        if len(self.experience['s']) >= self.max_experiences :

            for key in self.experience.keys() :

                self.experience[key].pop(0)

        for key, value in exp.items() :

            self.experience[key].append(value)

    

    # melakukan training dari neural network 

    # panduannya bisa dilihat di https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    def train(self, TargetNet) :

        

        # hanya melakukan training bila state yang diingat telah melebihi batas minimal

        if len(self.experience['s']) < self.min_experiences :

            return 0

        

        # mengambil ingatan permainan secara random sesuai ukuran batch   

        ids = np.random.randint(low = 0, high = len(self.experience['s']), size = self.batch_size)

        states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])

        actions = np.asarray([self.experience['a'][i] for i in ids])

        rewards = np.asarray([self.experience['r'][i] for i in ids])

        

        # mengambil label

        next_states = np.asarray([self.preprocess(self.experience['s2'][i]) for i in ids])

        dones = np.asarray([self.experience['done'][i] for i in ids])

                

        # untuk semua ingatan yang diambil, kita harus memprediksi maksimal Q-value dari state setelahnya

        value_next = np.max(TargetNet.predict(next_states).detach().numpy(), axis = 1)

        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        

        actions = np.expand_dims(actions, axis = 1)

        actions_one_hot = torch.FloatTensor(self.batch_size, self.num_actions).zero_()

        actions_one_hot = actions_one_hot.scatter_(1, torch.LongTensor(actions), 1)

        selected_action_values = torch.sum(self.predict(states) * actions_one_hot, dim = 1)

        actual_values = torch.FloatTensor(actual_values)

        

        self.optimizer.zero_grad()

        loss = self.criterion(selected_action_values, actual_values)

        loss.backward()

        self.optimizer.step()

    

    def copy_weights(self, TrainNet) :

        self.model.load_state_dict(TrainNet.state_dict())

        

    def save_weights(self, path) :

        torch.save(self.model.state_dict(), path)

        

    def load_weights(self, path) :

        self.model.load_state_dict(torch.load(path))      

        

    # fungsi untuk menentukan aksi yang diambil untuk bermain berdasarkan hasil prediksi dan aturan eksplorasi

    

    # fungsi untuk preprocess state sebelum dimas

    def preprocess(self, state) :

        result = state.board[:]

        result.append(state.mark)

        return result

    

    # fungsi untuk prediksi 

    def predict(self, inputs) :

        return self.model(torch.from_numpy(inputs).float())

    

    def get_action(self, state, epsilon) :

        # mekanisme eksplorasi, melakukan aksi random bila kolom tersebut kosong 

        if np.random.random() < epsilon :

            return int(np.random.choice([c for c in range(self.num_actions) if state.board[c] == 0]))

        else :

            prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].detach().numpy()

            for i in range(self.num_actions) :

                if state.board[i] != 0 :

                    prediction[i] = -1e7

            # melakukan aksi dengan menaruh checker di kolom yang punya hasil prediksi Q value yang paling besar

            return int(np.argmax(prediction)) 



# mendefinisikan aturan permainan yang akan dipelajari agent

def play_game(env, TrainNet, TargetNet, epsilon, copy_step) :

    rewards = 0

    iter = 0

    done = False

    observations = env.reset()

    

    while not done :     

        action = TrainNet.get_action(observations, epsilon)

        prev_observations = observations

        observations, reward, done, _ = env.step(action)  

        if done :

            # menang

            if reward == 1 :

                reward = 20

            # kalah

            elif reward == 0 :

                reward = -50

            # draw

            else :

                reward = 10

        # membuat agent kita berusaha untuk bermain lebih panjang (selama permainan belum berakhir dia mendapat reward 0.5)

        # namun harus dibuat threshold agar bila telah melewati batas tertentu maka agen berusaha menang (bukan berusaha main lebih panjang)    

        else :

            if rewards <= 2.5 : 

                reward = 0.5

            else :

                reward = -0.5

        rewards += reward

        

        # membuat buffer ingatan permainan

        exp = {'s' : prev_observations, 'a' : action, 'r' : reward, 's2' : observations, 'done' : done}

        TrainNet.add_experience(exp)

        TrainNet.train(TargetNet)

        

        # ingat bahwa kita membuat network yang memprediksi nilai sebenarnya dari target akan mengcopy weight dari network satu lagi

        iter+=1

        if iter % copy_step == 0 :

            TargetNet.copy_weights(TrainNet)

            

    return rewards    
# mengaktifkan environment



env = ConnectX()



# mengatur hyperparameter



gamma = 0.99 

copy_step = 25

hidden_units = [100, 200, 200, 100]

max_experiences = 1000

min_experiences = 100

batch_size = 32

lr = 0.01

epsilon = 0.25    # lebih memilih eksploitasi, namun decay epsilon diperlambat

decay = 0.99

min_epsilon = 0.05

episodes =14000

precision = 7
# inisiasi



num_states = env.observation_space.n + 1

num_actions = env.action_space.n



all_total_rewards = np.empty(episodes) 

all_avg_rewards = np.empty(episodes)

all_epsilons = np.empty(episodes)



TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)



progress_bar = tqdm(range(episodes))

for i in progress_bar :

    if i % 10 == 0 :

        epsilon = max(min_epsilon, epsilon * decay)

    else :

        epsilon = max(min_epsilon, epsilon)

    total_reward = play_game(env, TrainNet, TargetNet, epsilon, copy_step)

    all_total_rewards[i] = total_reward

    avg_reward = all_total_rewards[max(0, i-100) : (i+1)].mean()

    all_avg_rewards[i] = avg_reward

    all_epsilons[i] = epsilon

    progress_bar.set_postfix({

        'episode_reward' : total_reward,

        'avg of last 100 reward' : avg_reward,

        'epsilon' : epsilon

    })
# melihat hasil pelatihan agent



plt.plot(all_avg_rewards)

plt.xlabel('Episode')

plt.ylabel('Avg Last 100 Rewards ')

plt.show()



# menyimpan weight dari tiap state



TrainNet.save_weights('./weights.pth')
fc_layers = []



# mengambil weight dan bias dari tiap hidden layer serta output layer

for i in range(len(hidden_units)):

    fc_layers.extend([

        TrainNet.model.hidden_layers[i].weight.T.tolist(), 

        TrainNet.model.hidden_layers[i].bias.tolist() 

    ])

fc_layers.extend([

    TrainNet.model.output_layer.weight.T.tolist(), 

    TrainNet.model.output_layer.bias.tolist() 

])



# mengedit hasil dari fc_layers

fc_layers = list(map(

    lambda x: str(list(np.round(x, precision))) \

        .replace('array(', '').replace(')', '') \

        .replace(' ', '') \

        .replace('\n', ''),

    fc_layers

))

fc_layers = np.reshape(fc_layers, (-1, 2))



# Menuliskan agent

# Agent tersebut harus diinisiasi terlebih dahulu, mempunyai daftar weights tiap state untuk melakukan perhitungan, dan melakukan aksi



# inisiasi agent

my_agent = '''def my_agent(observation, configuration):

    import numpy as np



'''



# memasukkan hasil bobot tiap hidden layer

for i, (w, b) in enumerate(fc_layers[:-1]):

    my_agent += '    hl{}_w = np.array({}, dtype=np.float32)\n'.format(i+1, w)

    my_agent += '    hl{}_b = np.array({}, dtype=np.float32)\n'.format(i+1, b)

my_agent += '    ol_w = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][0])

my_agent += '    ol_b = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][1])



my_agent += '''

    state = observation.board[:]

    state.append(observation.mark)

    out = np.array(state, dtype=np.float32)



'''



# melakukan kalkulasi Q-value berdasarkan weight hidden layer hingga output layer

for i in range(len(fc_layers[:-1])):

    my_agent += '    out = np.matmul(out, hl{0}_w) + hl{0}_b\n'.format(i+1)

    my_agent += '    out = np.maximum(out,0)\n'     # fungsi aktivasi ReLU .clamp(min = 0)

    

my_agent += '    out = np.matmul(out, ol_w) + ol_b\n'



# melakukan aksi

my_agent += '''

    for i in range(configuration.columns):

        if observation.board[i] != 0:

            out[i] = -1e7



    return int(np.argmax(out))

    '''

with open('submission.py', 'w') as f:

    f.write(my_agent)
from submission import my_agent



def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



print("My Agent vs. Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=50)))

print("My Agent vs. Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=50)))

print("Random Agent vs. My Agent:", mean_reward(evaluate("connectx", ["random", my_agent], num_episodes=50)))

print("Negamax Agent vs. My Agent:", mean_reward(evaluate("connectx", ["negamax", my_agent], num_episodes=50)))