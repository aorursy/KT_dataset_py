import numpy as np

from numpy import tanh

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import math

import timeit
pd.set_option('display.max_columns', 50)

afr = pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv').drop(['case','cc3','country','year'], axis=1)

afr.dropna(inplace=True)

afr.banking_crisis = afr.banking_crisis.map({'crisis':1,'no_crisis':-1})

afr
x = afr.drop('banking_crisis', axis=1).values

y = afr.banking_crisis.values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)
labels = ['batch', 'stochastic']

learnings = [0.00001, 0.0001, 0.001, 0.01, 0.1]

# funzione per eseguire i calcoli

# passati un array di percettroni, il tipo di perceptron desiderato e l'errore da raggiungere, addestra i percettroni con i learning_rates

# sopra definiti e restituisce 2 array con i calcoli della precisione e degli errori commessi sul test set

def calc(p, t, eps = 0.2):

    start = timeit.default_timer()

    ris = []; er = []

    for l in learnings:

        ris.append([]); er.append([]); ind = learnings.index(l)

        for i in range(len(p)):    

            p[i].train(x=X_train, y=y_train, x_test=X_test, y_test=y_test, learning_rate=l, no_batch=[i], t=t, eps=eps)

            ris[ind].append(p[i].acc); er[ind].append(p[i].errors)

    stop = timeit.default_timer()

    print('Time: ', stop - start)  

    return ris, er

# funzioni per disegnare

plt.rcParams.update({'font.size': 22})



# disegna le funzioni segno, sigmoid, relu, tanh

def dis_func(t=0, names=[]):

    plt.rcParams["figure.figsize"] = (20, 5)

    if t == 0:

        fun = lambda x:x

    elif t == 1:

        fun = sigmoid

    elif t == 2:

        fun = tanh

    else:

        fun = relu

    deriv = derivative(t)

    l = np.linspace(-5, 5, 100)

    f, axx = plt.subplots(1,2)

    yd = []; yy = []

    for u in l:

        yy.append(fun(u))

        yd.append(deriv(fun(u)))

    axx[0].plot(l, yy)

    axx[0].grid(True)

    axx[0].set_title(names[0])

    axx[1].plot(l, yd); axx[1].grid(True); axx[1].set_title(names[1])

    plt.rcParams["figure.figsize"] = (25, 20)

# disegna accuratezza ed errori commessi

def dis(p, r, e):

    plt.rcParams["figure.figsize"] = (25, 20)

    fig, ax = plt.subplots(len(learnings), 2) 

    for l in range(len(learnings)):

        ax[l][0].set_title('accuratezza con learning_rate = {} '.format(learnings[l]))

        ax[l][1].set_title('errore quadratico medio con learning_rate = {}'.format(learnings[l]))

        for i in range(len(p)):

            ax[l][0].plot(range(1,len(r[l][i]) + 1), r[l][i], label=labels[i])

            ax[l][0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

            ax[l][1].plot(range(1,len(e[l][i]) + 1), e[l][i], label=labels[i])

            ax[l][1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.legend()

    plt.tight_layout()
# Funzione per il calcolo della sigmoide

def sigmoid(gamma):

    if gamma < 0:

        return 1 - 1/(1 + math.exp(gamma))

    else:

        return 1/(1 + math.exp(-gamma))

    

# funzione per il calcolo della relu

def relu(x):

    return max(0.001*x, x)



# Serve per fornire alla classe perceptron la giusta funzione segno in base alla funzione di attivazione 

def sign(t=0):

    if t == 1:

        return lambda x: np.where(x >= 0.5, 1, -1)

    return lambda x: np.where(x >= 0.0, 1, -1)

    

# Fornisce alla classe perceptron la derivata della funzione di attivazione in base alla funzione di attivazione

def derivative(t=0):

    if t == 0:

        return lambda x: 1

    if t == 1:

        return lambda x: x * ( 1 - x )   

    if t == 2:

        return lambda x: 1.0 - x**2

    if t == 3:

        return lambda x: 1 if x >= 0 else 0.001


# classe perceptron

class perceptron:

    

    # In base al parametro definsice la funzione di attivazione

    def act(self, t=0):

        if t == 0:

            return lambda x: np.dot(x, self.w[1:]) + self.w[0]  

        if t == 1:

            return lambda x: sigmoid(np.dot(x, self.w[1:]) + self.w[0])

        if t == 2:

            return lambda x: tanh(np.dot(x, self.w[1:]) + self.w[0])

        if t == 3:

            return lambda x: relu(np.dot(x, self.w[1:]) + self.w[0])

    

    # predice applicando la funzione segno all'attivazione

    def predict(self,x): 

        return self.sign(self.activation(x)) 

    

    # calcolo dell'errore quadratico medio

    def err_calc(self, x, y):

        ris = []; m = 0

        for xi, yi in zip(x, y):

            p = self.sign(self.activation(xi))

            m += ( yi - p ) ** 2

        return m / ( 2 * len(x) )

    

    # calcolo dell' accuratezza

    def accuracy(self, x, y):

        ris = []

        for e in x:

            ris.append(self.predict(e.tolist()))

        return (sum(np.array(ris) == np.array(y)) / len(y))

    

    # inizializza i valori

    # x è il train set, learning_rate = eta, no_batch = 0 crea un perceptron di tipo batch, t il tipo di attivazione

    def initialize(self, x, learning_rate, no_batch, t=0):

        # assegno le funzioni in base al tipo desiderato

        self.sign = sign(t); self.activation = self.act(t); self.derivative = derivative(t)  

        self.errors = []; self.no_batch = no_batch; self.acc = []; self.inputs_dim = len(x[0])

        self.w = np.random.uniform(low=-0.05, high=0.05, size=(self.inputs_dim + 1,))

        self.learning_rate = learning_rate; self.w[0] = 1 # bias iniziale

        

    # aggiorna pesi

    def update(self, xi, yi):

        s = self.activation(xi)

        p = self.sign(s)

        # aggiorno solo se ha sbagliato la predizione

        if p - yi != 0: 

            err = yi - s 

            # se tipo batch aggiorna i delta

            if self.no_batch == 0: 

                self.dw[1:] += self.learning_rate * xi * err * self.derivative(s)

                self.dw[0] += self.learning_rate * err * self.derivative(s)

            else: 

                # se non batch aggiorna i pesi

                self.w[1:] += self.learning_rate * xi * err * self.derivative(s)

                self.w[0] += self.learning_rate * err * self.derivative(s)

                

    # addestra x,y sono il train , x_test, y_test il test, epochs il numero massimo di epoche se non raggiunge l'errore desiderato,

    # eps l'errore da raggiungere, no_batch=0 crea un perceptron di tipo batch, t = tipo di attivazione

    def train(self, x, y, x_test, y_test, learning_rate = 0.001, epochs=10000, eps=0.2, no_batch=1, t=0):

        # inizializza i dati

        self.initialize(x=x, learning_rate=learning_rate, no_batch=no_batch, t=t) 

        # itera sule epoche

        for e in range(epochs): 

            # se batch inizializza i delta

            if self.no_batch == 0:

                self.dw = np.zeros(self.inputs_dim + 1)

            # chiama la funzione di update per ogni esempio    

            for xi, yi in zip(x, y): 

                self.update(xi, yi)

            # se batch somma delta ai pesi

            if self.no_batch == 0: 

                self.w -= self.dw / len(x)

            # crea gli array di accuratezza ed errore    

            self.acc.append(self.accuracy(x=x_test, y=y_test)) 

            er = self.err_calc(x=x, y=y)

            self.errors.append(er) 

            #se raggiunto l'errore interrompe

            if er < eps and e > 10: 

                break
dis_func(t=0,names=['sign(x)', 'sign(x) derivata'])
perc_n = [perceptron(), perceptron()]

ris_n,er_n = calc(perc_n, t=0)

dis(perc_n, ris_n, er_n)
dis_func(t=1,names=['sigmoid(x)', 'sigmoid(x) derivata']) # disegna la funzione
perc_s = [perceptron(), perceptron()]

ris_s,er_s = calc(perc_s, t=1)

dis(perc_s, ris_s, er_s)  
dis_func(t=2,names=['tanh(x)', 'tanh(x) derivata'])
perc_t = [perceptron(), perceptron()]

ris_t,er_t = calc(perc_t, t=2)

dis(perc_t, ris_t, er_t)
dis_func(t=3,names=['relu(x)', 'relu(x) derivata'])
perc_r = [perceptron(), perceptron()]

ris_r,er_r = calc(perc_r, t=3)

dis(perc_r, ris_r, er_r)