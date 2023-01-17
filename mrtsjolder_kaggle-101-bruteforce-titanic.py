# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def guess_missing_features(dataset):

    # guess port

    freq_port = train_df.Embarked.dropna().mode()[0]

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

    # guess fares

    median_fare = dataset['Fare'].dropna().median()

    dataset['Fare'] = dataset['Fare'].fillna(median_fare)

    

    # guess ages

    for i in ['female', 'male']:

        for j in range(3):

            cond = (dataset['Sex'] == i) & (dataset['Pclass'] == j+1)

            guess_df = dataset[cond]['Age'].dropna()

            age_guess = int(guess_df.median() / 0.5 + 0.5) * 0.5

            dataset.loc[(dataset.Age.isnull()) & cond, 'Age'] = age_guess

    dataset['Age'] = dataset['Age'].astype(int)

    return dataset



def invent_features(dataset):

    # use titles from name

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    rare_titles = ['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

    rare_titles += ['Lady', 'Countess', 'Dona']

    dataset['Title'] = dataset['Title'].replace(rare_titles, 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})

    dataset['Title'] = dataset['Title'].fillna(0)

    dataset = dataset.drop(['Name'], axis=1)

    

    return dataset



def wrangle_data(dataset):

    # drop useless features

    dataset = dataset.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)

    

    dataset = guess_missing_features(dataset)

    

    # from categorical to numerical

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    dataset = invent_features(dataset)

    

    return dataset
train_df = pd.read_csv('../input/train.csv')

train = wrangle_data(train_df)

x, y = train.drop("Survived", axis=1), train["Survived"]

train.head()
def get_function(name, alpha=1.):

    if name == 'linear':

        f = lambda x: x

        f1 = lambda a: 1

    elif name == 'sigmoid':

        f = lambda x: 1 / (1 + np.exp(-x))

        f1 = lambda a: a * (1 - a)

    elif name == 'tanh':

        f = lambda x: 2 * np.tanh(x / 2)

        f1 = lambda a: 1 - (a / 2)**2

    elif name == 'relu':

        f = lambda x: np.where(x < 0, 0, x)

        f1 = lambda a: np.where(a < 0, 0, 1)

    elif name == 'elu':

        f = lambda x: np.where(x < 0, alpha * (np.exp(x) - 1), x)

        f1 = lambda a: np.where(a < 0, a + alpha, 1)

    elif name == 'softmax':

        f = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

        f1 = lambda a: a * (1 - a)

    else:

        raise ValueError('Unrecognised function!')

    

    return f, f1



class NeuralNet:

    

    def __init__(self, neurons, f = 'tanh', g = 'softmax'):

        # architecture

        self.n_in, self.n_out = neurons[0], neurons[-1]

        self.n_layers = len(neurons) - 1                        # input is no layer

        

        # activation functions

        self.f, self.g = f, g

        f, p = get_function(f)

        g, _ = get_function(g)

        self.functions = [f] * (self.n_layers - 1) + [g]

        self.prime = p

        

        # parameters

        self.weights = []

        self.biases = []

        

        for n_in, n_out in zip(neurons[:-1], neurons[1:]):

            self.weights.append(np.zeros((n_out, n_in)))

            self.biases.append(np.zeros(n_out))

        

        self.weight_init()

        

        # forward and backward pass variables

        self.activations = []

        self.deltas = []

        self.gradients = []

        

    def weight_init(self):

        self.activations = []

        self.deltas = []

        self.gradients = []

        

        for w,b in zip(self.weights, self.biases):

            bound = (6 / w.shape[-1]) ** .5

            w[:] = np.random.uniform(-bound, bound, w.shape)

            b[:] = np.zeros(b.shape)

        

    def forward(self, x):

        try:

            x = np.reshape(x, (-1, self.n_in))

        except ValueError:

            raise ValueError('Expected x to have {:d} columns!'.format(self.n_in))

        

        self.deltas = []

        self.gradients = []

        

        a = x

        self.activations.append(x)

        for w, b, f in zip(self.weights, self.biases, self.functions):

            s = np.dot(a, w.T) + b

            a = f(s)

            self.activations.append(a)

        

        return np.squeeze(a)

    

    def backward(self, y):

        if not self.activations:

            raise ValueError('Run the forward pass first!')

        if np.size(y) != np.size(self.activations[-1]):

            raise ValueError('Expected y to have {:d} columns!'.format(self.n_out))

            

        y = np.reshape(y, (-1, self.n_out))

        delta = self.activations[-1] - y

        self.deltas.append(delta)

        for x, w in zip(self.activations[-2::-1], self.weights[::-1]):

            dw = np.dot(delta.T, x)

            self.gradients.append(dw)

            

            delta = np.dot(delta, w)

            delta *= self.prime(x)

            self.deltas.append(delta)

        

    def update(self, eta=1e-3):

        if not self.gradients or not self.deltas:

            raise ValueError('Run backward pass first!')

        

        for w, b, dw, d in zip(self.weights, self.biases, self.gradients[::-1], self.deltas[-2::-1]):

            w -= eta * dw

            b -= eta * np.sum(d, axis=0)

        

        self.activations = []

        self.gradients = []

        self.deltas = []

    

    def evaluate(self, x, y):

        pred = self.forward(x)

        y = np.reshape(y, pred.shape)

        

        if self.g == 'linear':

            loss = np.sum((pred - y) ** 2, axis=0)

        elif self.g == 'sigmoid':

            loss = - y * np.log(pred) - (1 - y) * np.log(1 - pred)

        elif self.g == 'softmax':

            loss = - np.sum(y * np.log(pred), axis=0)

        else:

            loss = pred - y

        

        return np.mean(loss)

     

    def learn(self, train, learning_rate=1e-3, batch_size=64, epochs=10):

        accuracies = [[], []]

        

        train = train[0][:-batch_size], train[1][:-batch_size]

        valid = train[0][-batch_size:], train[1][-batch_size:]

        

        data = np.c_[train[0], train[1]]

        if data.shape[1] != self.n_in + self.n_out:

            raise ValueError('Invalid data dimensions!')

            

        mini_batches = [data[i:i+batch_size] for i in range(0, data.shape[0], batch_size)]

        mini_batches = [(batch[:,:self.n_in], batch[:,-self.n_out:]) for batch in mini_batches]

        

        for e in range(epochs):

            np.random.shuffle(data)

            

            for (x_batch, y_batch) in mini_batches:

                self.forward(x_batch)

                self.backward(y_batch)

                self.update(learning_rate / x_batch.shape[0])

            

            acc_train = self.evaluate(train[0], train[1])

            acc_valid = self.evaluate(valid[0], valid[1])

            accuracies[0].append(acc_train)

            accuracies[1].append(acc_valid)

        

        return accuracies
# parameters

lr = 1e-1

b = 128

e = 1000



# preprocessing

_x = np.array(x, dtype='float32')

_y = y.values

shift = np.mean(_x[:-b], axis=0)

scale = np.std(_x[:-b], axis=0)

_x -= shift

_x /= scale

train = (_x, _y)
# train model

model = NeuralNet([8, 64, 32, 16, 8, 1], f='tanh', g='sigmoid')

loss, val_loss = model.learn(train, learning_rate=lr, batch_size=b, epochs=e)



# compute accuracy

pred = model.forward(_x)

y_hat = np.where(pred > .5, 1, 0)

acc = np.mean(y_hat[:-b] == _y[:-b])

val_acc = np.mean(y_hat[-b:] == _y[-b:])

print(acc, val_acc)



# plot losses

from matplotlib.pyplot import plot, show

plot(loss, ls='-')

plot(val_loss, ls='--')

show()
test_df = pd.read_csv('../input/test.csv')

test = wrangle_data(test_df)

_test = (test.values - shift) / scale

submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': np.where(model.forward(_test) > .5, 1, 0)

})

submission.to_csv('.submission.csv', index=False)